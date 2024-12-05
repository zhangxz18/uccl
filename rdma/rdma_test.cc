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

static void server_worker(void)
{
    std::string remote_ip;
    auto ep = RdmaEndpoint(DEV_RDMA_DEFAULT, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_accept(remote_ip);

    printf("Server accepted connection from %s\n", remote_ip.c_str());
}

static void client_worker(void)
{
    auto ep = RdmaEndpoint(DEV_RDMA_DEFAULT, NUM_ENGINES, ENGINE_CPU_START);

    auto conn_id = ep.uccl_connect(FLAGS_serverip);

    printf("Client connected to %s\n", FLAGS_serverip.c_str());
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
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
