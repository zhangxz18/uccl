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

using namespace uccl;

static volatile bool quit = false;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_uint32(iterations, 100000, "Number of iterations to run.");

static void server_worker(void)
{
    std::string remote_ip;
    auto ep = RDMAEndpoint(GID_INDEX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_accept(0, remote_ip);

    printf("Server accepted connection from %s\n", remote_ip.c_str());

    void *data = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(data != MAP_FAILED);

    ep.uccl_regmr(conn_id, data, 65536, 0);

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


    while (!quit) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ep.uccl_deregmr(conn_id);
}

static void client_worker(void)
{
    auto ep = RDMAEndpoint(GID_INDEX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_connect(0, FLAGS_serverip);

    printf("Client connected to %s\n", FLAGS_serverip.c_str());
    
    void *data = mmap(nullptr, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(data != MAP_FAILED);

    ep.uccl_regmr(conn_id, data, 65536, 0);

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

    ep.uccl_deregmr(conn_id);
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
