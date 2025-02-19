#include "util_efa.h"

#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "transport_config.h"
#include "util.h"

using namespace uccl;

#define TCP_PORT 12345  // Port for exchanging QPNs & GIDs

// Exchange QPNs and GIDs via TCP
void exchange_qpns(const char *peer_ip, ConnMeta *local_metadata,
                   ConnMeta *remote_metadata) {
    int sock;
    struct sockaddr_in addr;
    char mode = peer_ip ? 'c' : 's';

    sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt,
               sizeof(opt));  // Avoid port conflicts

    addr.sin_family = AF_INET;
    addr.sin_port = htons(TCP_PORT);
    addr.sin_addr.s_addr = peer_ip ? inet_addr(peer_ip) : INADDR_ANY;

    if (mode == 's') {
        printf("Server waiting for connection...\n");
        bind(sock, (struct sockaddr *)&addr, sizeof(addr));
        listen(sock, 128);
        sock = accept(sock, NULL, NULL);  // Blocks if no client
        printf("Server accepted connection\n");
    } else {
        printf("Client attempting connection...\n");
        while (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("Connect failed, retrying...");
            sleep(1);
        }
        printf("Client connected\n");
    }

    // Set receive timeout to avoid blocking
    struct timeval timeout = {5, 0};  // 5 seconds timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // Send local QPN and GID
    if (send(sock, local_metadata, sizeof(ConnMeta), 0) <= 0)
        perror("send() failed");

    // Receive remote QPN and GID
    if (recv(sock, remote_metadata, sizeof(ConnMeta), 0) <= 0)
        perror("recv() timeout");

    close(sock);
    printf("QPNs and GIDs exchanged\n");
}

void run_server() {
    auto *socket = EFAFactory::CreateSocket(0, 0);

    auto local_meta = new ConnMeta();
    socket->get_conn_metadata(local_meta);
    auto remote_meta = new ConnMeta();
    exchange_qpns(nullptr, local_meta, remote_meta);

    std::vector<FrameDesc *> frames;
    do {
        frames = socket->poll_recv_cq(1);
    } while (frames.size() == 0);

    CHECK(frames.size() == 1);
    auto *frame = frames[0];

    LOG(INFO) << "Received packet: " << (char *)frame->get_pkt_hdr_addr();

    socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
    socket->push_pkt_data(frame->get_pkt_data_addr());
    socket->push_frame_desc((uint64_t)frame);
}

void run_client(const char *server_ip) {
    auto *socket = EFAFactory::CreateSocket(0, 0);

    auto local_meta = new ConnMeta();
    socket->get_conn_metadata(local_meta);
    auto remote_meta = new ConnMeta();
    exchange_qpns(server_ip, local_meta, remote_meta);

    auto *dev = EFAFactory::GetEFADevice(0);
    auto *dest_ah = dev->create_ah(remote_meta->gid);

    // Send a packet
    auto frame_desc = socket->pop_frame_desc();
    auto pkt_hdr = socket->pop_pkt_hdr();
    auto pkt_data = socket->pop_pkt_data();
    auto *frame = FrameDesc::Create(frame_desc, pkt_hdr, pkt_data,
                                    kUcclPktHdrLen, kUcclPktdataLen, 0);
    frame->set_dest_ah(dest_ah);
    frame->set_dest_qpn(remote_meta->qpn_list[0]);

    strcpy((char *)pkt_hdr, "Hello World");
    socket->send_packet(frame);

    // Poll CQ
    std::vector<FrameDesc *> frames;
    do {
        frames = socket->poll_send_cq(1);
    } while (frames.size() == 0);

    CHECK(frames.size() == 1);
    CHECK(frames[0] == frame);

    socket->push_frame_desc(frame_desc);
    socket->push_pkt_hdr(pkt_hdr);
    socket->push_pkt_data(pkt_data);
}

// TO RUN THE TEST:
// On server: ./util_efa_test
// On client: ./util_efa_test server_ip
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::once_flag init_flag;
    std::call_once(init_flag, []() { EFAFactory::Init(); });

    if (argc == 1) {
        // Server
        run_server();
    } else {
        // Client
        run_client(argv[1]);
    }

    return 0;
}
