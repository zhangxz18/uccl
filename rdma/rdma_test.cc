/**
 * @file rdma_test.cc
 * @brief Test for UCCL RDMA transport
 */
#include "transport.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <signal.h>

#include <chrono>
#include <deque>
#include <thread>

#include "transport_config.h"
#include "util.h"
#include "util_timer.h"

using namespace uccl;

static volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_string(perftype, "basic", "Performance type: basic/lat/tpt.");
DEFINE_bool(warmup, false, "Whether to run warmup.");
DEFINE_uint32(nmsg, 8, "Number of messages within one request to post.");
DEFINE_uint32(nreq, 4, "Outstanding requests to post.");
DEFINE_uint32(msize, 65536, "Size of message.");
DEFINE_uint32(iterations, 100000, "Number of iterations to run.");

static void server_basic(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    for (int i = 0; i < FLAGS_iterations; i++) {
        size_t len = 65536;
        void *recv_data = data;

        auto *poll_ctx = ep.uccl_recv_async(conn_id, &recv_data, &len, 1);

        ep.uccl_poll(poll_ctx);

        // verify data
        for (int i = 0; i < 65536 / 4; i++) {
            assert(((uint32_t *)data)[i] == 0x123456);
        }

        // LOG(INFO) << "Iteration " << i << " done";
        std::cout << "Iteration " << i << " done" << std::endl;
    }
}

static void client_basic(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    // Fill data in a pattern of 0x123456,0x123456,0x123456...
    for (int i = 0; i < 65536 / 4; i++) {
        ((uint32_t *)data)[i] = 0x123456;
    }

    for (int i = 0; i < FLAGS_iterations; i++) {
        void *send_data = data;
        auto *poll_ctx = ep.uccl_send_async(conn_id, send_data, 65536);

        ep.uccl_poll(poll_ctx);

        // LOG(INFO) << "Iteration " << i << " done";
        std::cout << "Iteration " << i << " done" << std::endl;
    }
}

static void server_lat(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    // Latency is measured at server side as it is asynchronous receive
    // c6525-25g, kPortEntropy = 32/1
    // Min: 16/15us
    // P50: 23/23us
    // P90: 33/32us
    // P99: 43/41us
    // Max: 614/635us
    std::vector<uint64_t> lat_vec;

    if (FLAGS_warmup) {
        for (int i = 0; i < 1000; i++) {
            size_t len = 100;
            void *recv_data = data;
            auto *poll_ctx = ep.uccl_recv_async(conn_id, &recv_data, &len, 1);
            ep.uccl_poll(poll_ctx);
        }
    }

    for (int i = 0; i < FLAGS_iterations; i++) {
        size_t len = 100;
        void *recv_data = data;
        auto t1 = rdtsc();
        auto *poll_ctx = ep.uccl_recv_async(conn_id, &recv_data, &len, 1);
        ep.uccl_poll(poll_ctx);
        auto t2 = rdtsc();
        lat_vec.push_back(to_usec(t2 - t1, freq_ghz));
    }
    std::sort(lat_vec.begin(), lat_vec.end());
    std::cout << "Min: " << lat_vec[0] << "us" << std::endl;
    std::cout << "P50: " << lat_vec[FLAGS_iterations / 2] << "us" << std::endl;
    std::cout << "P90: " << lat_vec[FLAGS_iterations * 9 / 10] << "us" << std::endl;
    std::cout << "P99: " << lat_vec[FLAGS_iterations * 99 / 100] << "us" << std::endl;
    std::cout << "Max: " << lat_vec[FLAGS_iterations - 1] << "us" << std::endl;
}

static void client_lat(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    if (FLAGS_warmup) {
        for (int i = 0; i < 1000; i++) {
            void *send_data = data;
            auto *poll_ctx = ep.uccl_send_async(conn_id, send_data, 100);
            ep.uccl_poll(poll_ctx);
        }
    }

    for (int i = 0; i < FLAGS_iterations; i++) {
        void *send_data = data;
        auto *poll_ctx = ep.uccl_send_async(conn_id, send_data, 100);
        ep.uccl_poll(poll_ctx);
    }
}

static void server_tpt(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    if (FLAGS_warmup) {
        for (int i = 0; i < 1000; i++) {
            size_t len = 100;
            void *recv_data = data;
            auto *poll_ctx = ep.uccl_recv_async(conn_id, &recv_data, &len, 1);
            ep.uccl_poll(poll_ctx);
        }
    }

    size_t len[FLAGS_nreq][FLAGS_nmsg];
    void *recv_data[FLAGS_nreq][FLAGS_nmsg];
    std::vector<PollCtx *> poll_ctx_vec(FLAGS_nreq);
    for (int r = 0; r < FLAGS_nreq; r++)
        for (int m = 0; m < FLAGS_nmsg; m++) {
            // First half of the buffer
            len[r][m] = FLAGS_msize;
            recv_data[r][m] = reinterpret_cast<char*>(data) + r * FLAGS_msize * FLAGS_nmsg + m * FLAGS_msize;
        }

    for (int r = 0; r < FLAGS_nreq; r++) {
        poll_ctx_vec[r] = ep.uccl_recv_async(conn_id, recv_data[r], len[r], FLAGS_nmsg);
        FLAGS_iterations--;
    }

    while (FLAGS_iterations) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            if (ep.uccl_poll_once(poll_ctx_vec[r])) {
                poll_ctx_vec[r] = ep.uccl_recv_async(conn_id, recv_data[r], len[r], FLAGS_nmsg);
                FLAGS_iterations--;
            }
        }
    }
}

static void client_tpt(RDMAEndpoint &ep, ConnID conn_id, void *data)
{
    volatile uint64_t prev_sec_bytes = 0;
    volatile uint64_t cur_sec_bytes = 0;
    
    // Create a thread to print throughput every second
    std::thread t([&] {
        while (!quit) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << 
                (cur_sec_bytes - prev_sec_bytes) * 8.0 / 1000 / 1000 / 1000 << " Gbps" << std::endl;
            prev_sec_bytes = cur_sec_bytes;
        }
    });

    if (FLAGS_warmup) {
        for (int i = 0; i < 1000; i++) {
            void *send_data = data;
            auto *poll_ctx = ep.uccl_send_async(conn_id, send_data, 100);
            ep.uccl_poll(poll_ctx);
        }
    }
    
    std::vector<std::vector<PollCtx *>> poll_ctx_vec(FLAGS_nreq);
    for (int i = 0; i < FLAGS_nreq; i++) {
        poll_ctx_vec[i].resize(FLAGS_nmsg);
    }

    for (int r = 0; r < FLAGS_nreq; r++) {
        for (int n = 0; n < FLAGS_nmsg; n++) {
            void *send_data = reinterpret_cast<char*>(data) + r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
            poll_ctx_vec[r][n] = ep.uccl_send_async(conn_id, send_data, FLAGS_msize);
            cur_sec_bytes += FLAGS_msize;
        }
        FLAGS_iterations--;
    }

    int fin_msg = 0;

    while (FLAGS_iterations) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            for (int n = 0; n < FLAGS_nmsg; n++) {
                if (ep.uccl_poll_once(poll_ctx_vec[r][n])) {
                    void *send_data = reinterpret_cast<char*>(data) + r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
                    poll_ctx_vec[r][n] = ep.uccl_send_async(conn_id, send_data, FLAGS_msize);
                    cur_sec_bytes += FLAGS_msize;
                    if (++fin_msg == FLAGS_nreq) {
                        FLAGS_iterations--;
                        fin_msg = 0;
                    }
                }
            }
        }
    }

    t.join();

}

static void server_worker(void)
{
    std::string remote_ip;
    auto ep = RDMAEndpoint(GID_INDEX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_accept(0, remote_ip);

    printf("Server accepted connection from %s\n", remote_ip.c_str());

    void *data = mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(data != MAP_FAILED);

    ep.uccl_regmr(conn_id, data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0);

    if (FLAGS_perftype == "basic") {
        server_basic(ep, conn_id, data);
    } else if (FLAGS_perftype == "lat") {
        server_lat(ep, conn_id, data);
    } else if (FLAGS_perftype == "tpt") {
        server_tpt(ep, conn_id, data);
    } else {
        std::cerr << "Unknown performance type: " << FLAGS_perftype << std::endl;
    }

    while (!quit) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ep.uccl_deregmr(conn_id);
    munmap(data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
}

static void client_worker(void)
{
    auto ep = RDMAEndpoint(GID_INDEX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_connect(0, FLAGS_serverip);

    printf("Client connected to %s\n", FLAGS_serverip.c_str());
    
    void *data = mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(data != MAP_FAILED);

    ep.uccl_regmr(conn_id, data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0);

    if (FLAGS_perftype == "basic") {
        client_basic(ep, conn_id, data);
    } else if (FLAGS_perftype == "lat") {
        client_lat(ep, conn_id, data);
    } else if (FLAGS_perftype == "tpt") {
        client_tpt(ep, conn_id, data);
    } else {
        std::cerr << "Unknown performance type: " << FLAGS_perftype << std::endl;
    }

    ep.uccl_deregmr(conn_id);

    munmap(data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    // google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_v = 1;
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    if (FLAGS_server) {
        server_worker();
    } else {
        client_worker();
    }

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    return 0;
}
