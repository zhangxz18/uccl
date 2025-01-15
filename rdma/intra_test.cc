/**
 * @file rdma_test.cc
 * @brief Test for UCCL RDMA transport
 */

#include <chrono>
#include <deque>
#include <thread>

#include <signal.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "transport.h"
#include "transport_config.h"
#include "util.h"
#include "util_timer.h"

using namespace uccl;

static volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

std::optional<RDMAEndpoint> ep;

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "192.168.25.1", "Server IP address the client tries to connect.");
DEFINE_uint32(nflow, 1, "Number of flows.");
DEFINE_uint32(nmsg, 1, "Number of messages within one request to post.");
DEFINE_uint32(nreq, 1, "Outstanding requests to post.");
DEFINE_uint32(msize, 1000000, "Size of message.");
DEFINE_uint32(iterations, 100000, "Number of iterations to run.");
DEFINE_bool(flush, false, "Whether to flush after receiving.");

static void server_tpt(std::vector<ConnID> &conn_ids, std::vector<struct Mhandle *> &mhandles, std::vector<void *> &datas)
{
    FLAGS_iterations *= FLAGS_nflow;

    int len[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];
    struct Mhandle *mhs[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];
    void *recv_data[FLAGS_nflow][FLAGS_nreq][FLAGS_nmsg];
    
    int flag[FLAGS_nflow][FLAGS_nreq];
    memset(flag, 0, sizeof(int) * FLAGS_nflow * FLAGS_nreq);
    
    std::vector<std::vector<PollCtx *>> poll_ctx_vec(FLAGS_nflow);
    for (int i = 0; i < FLAGS_nflow; i++) {
        poll_ctx_vec[i].resize(FLAGS_nreq);
    }
    
    for (int f = 0; f < FLAGS_nflow; f++) {
        for (int r = 0; r < FLAGS_nreq; r++)
            for (int m = 0; m < FLAGS_nmsg; m++) {
                len[f][r][m] = FLAGS_msize;
                recv_data[f][r][m] = reinterpret_cast<char*>(datas[f]) + r * FLAGS_msize * FLAGS_nmsg + m * FLAGS_msize;
                mhs[f][r][m] = mhandles[f];
            }
    }

    for (int f = 0; f < FLAGS_nflow; f++) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            poll_ctx_vec[f][r] = ep->uccl_recv_async(conn_ids[f], mhs[f][r], recv_data[f][r], len[f][r], FLAGS_nmsg);
            FLAGS_iterations--;
        }
    }

    while (FLAGS_iterations) {
        for (int f = 0; f < FLAGS_nflow; f++) {
            for (int r = 0; r < FLAGS_nreq; r++) {
                if (!ep->uccl_poll_once(poll_ctx_vec[f][r])) continue;

                if (!FLAGS_flush) {
                    poll_ctx_vec[f][r] = ep->uccl_recv_async(conn_ids[f], mhs[f][r], recv_data[f][r], len[f][r], FLAGS_nmsg);
                    FLAGS_iterations--;
                    continue;
                }
                
                if (flag[f][r] == 0) {
                    poll_ctx_vec[f][r] = ep->uccl_flush(conn_ids[f], mhs[f][r], recv_data[f][r], len[f][r], FLAGS_nmsg);
                    flag[f][r] = 1;
                }
                else if (flag[f][r] == 1) {
                    poll_ctx_vec[f][r] = ep->uccl_recv_async(conn_ids[f], mhs[f][r], recv_data[f][r], len[f][r], FLAGS_nmsg);
                    FLAGS_iterations--;
                    flag[f][r] = 0;
                }
            }
        }
    }
}

static void client_tpt(std::vector<ConnID> &conn_ids, std::vector<struct Mhandle *> &mhandles, std::vector<void *> &datas)
{
    volatile uint64_t prev_sec_bytes = 0;
    volatile uint64_t cur_sec_bytes = 0;

    FLAGS_iterations *= FLAGS_nflow;
    
    // Create a thread to print throughput every second
    std::thread t([&] {
        while (!quit) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "(" << FLAGS_nflow << " flows) Throughput: " << std::fixed << std::setprecision(2) << 
                (cur_sec_bytes - prev_sec_bytes) * 8.0 / 1000 / 1000 / 1000 << " Gbps" << std::endl;
            prev_sec_bytes = cur_sec_bytes;
        }
    });
    
    std::vector<std::vector<std::vector<PollCtx *>>> poll_ctx_vec(FLAGS_nflow);
    for (int f = 0; f < FLAGS_nflow; f++) {
        poll_ctx_vec[f].resize(FLAGS_nreq);
        for (int r = 0; r < FLAGS_nreq; r++) {
            poll_ctx_vec[f][r].resize(FLAGS_nmsg);
        }
    }
    for (int f = 0; f < FLAGS_nflow; f++) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            for (int n = 0; n < FLAGS_nmsg; n++) {
                void *send_data = reinterpret_cast<char*>(datas[f]) + r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
                poll_ctx_vec[f][r][n] = ep->uccl_send_async(conn_ids[f], mhandles[f], send_data, FLAGS_msize);
                cur_sec_bytes += FLAGS_msize;
            }
            FLAGS_iterations--;
        }
    }

    int fin_msg = 0;

    while (FLAGS_iterations) {
        for (int f = 0; f < FLAGS_nflow; f++) {
            for (int r = 0; r < FLAGS_nreq; r++) {
                for (int n = 0; n < FLAGS_nmsg; n++) {
                    if (ep->uccl_poll_once(poll_ctx_vec[f][r][n])) {
                        void *send_data = reinterpret_cast<char*>(datas[f]) + r * FLAGS_msize * FLAGS_nmsg + n * FLAGS_msize;
                        poll_ctx_vec[f][r][n] = ep->uccl_send_async(conn_ids[f], mhandles[f], send_data, FLAGS_msize);
                        cur_sec_bytes += FLAGS_msize;
                        if (++fin_msg == FLAGS_nreq) {
                            FLAGS_iterations--;
                            fin_msg = 0;
                        }
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

    std::vector<ConnID> conn_ids;
    std::vector<void *> datas;
    std::vector<struct Mhandle *> mhandles;

    mhandles.resize(FLAGS_nflow);

    for (int i = 0; i < FLAGS_nflow; i++) {
        auto conn_id = ep->uccl_accept(0, i % NUM_ENGINES, remote_ip);
        printf("Server accepted connection from %s (flow#%d)\n", remote_ip.c_str(), i);
        void *data = mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(data != MAP_FAILED);
        ep->uccl_regmr(conn_id, data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

        conn_ids.push_back(conn_id);
        datas.push_back(data);
    }

    server_tpt(conn_ids, mhandles, datas);

    while (!quit) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int i = 0; i < FLAGS_nflow; i++) {
        ep->uccl_deregmr(conn_ids[i], mhandles[i]);
        munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
        delete mhandles[i];
    }
}

static void client_worker(void)
{
    std::vector<ConnID> conn_ids;
    std::vector<void *> datas;
    std::vector<struct Mhandle *> mhandles;

    mhandles.resize(FLAGS_nflow);

    for (int i = 0; i < FLAGS_nflow; i++) {
        auto conn_id = ep->uccl_connect(0, i % NUM_ENGINES, FLAGS_serverip);
        printf("Client connected to %s (flow#%d)\n", FLAGS_serverip.c_str(), i);
        void *data = mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(data != MAP_FAILED);
        ep->uccl_regmr(conn_id, data, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

        conn_ids.push_back(conn_id);
        datas.push_back(data);
    }

    client_tpt(conn_ids, mhandles, datas);

    for (int i = 0; i < FLAGS_nflow; i++) {
        ep->uccl_deregmr(conn_ids[i], mhandles[i]);
        munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
        delete mhandles[i];
    }
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    ep.emplace(GID_INDEX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);
    
    std::thread server_thread(server_worker);
    std::thread client_thread(client_worker);

    server_thread.join();
    client_thread.join();

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    return 0;
}
