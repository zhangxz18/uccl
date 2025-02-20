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
#define ITERATIONS 10240

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

    auto *dev = EFAFactory::GetEFADevice(0);
    auto *dest_ah = dev->create_ah(remote_meta->gid);

    std::vector<FrameDesc *> frames;
    FrameDesc *frame;

    TimePoint start_time, mid_time, end_time;
    std::chrono::duration<double> duration1, duration2;
    for (int i = 0; i < ITERATIONS; i++) {
        // Receiving a data packet.
        do {
            frames = socket->poll_recv_cq(1);
        } while (frames.size() == 0);

        start_time = std::chrono::high_resolution_clock::now();

        CHECK(frames.size() == 1);
        frame = frames[0];
        CHECK(strcmp((char *)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
                     "Hello World") == 0);
        socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
        socket->push_pkt_data(frame->get_pkt_data_addr());
        socket->push_frame_desc((uint64_t)frame);
        CHECK(socket->recv_queue_wrs() >=
              kMaxSendWr * kMaxPath - kMaxRecvDeficitCnt * kMaxPath);

        // Send it back.
        frame = FrameDesc::Create(socket->pop_frame_desc(),
                                  socket->pop_pkt_hdr(), kUcclPktHdrLen,
                                  socket->pop_pkt_data(), kUcclPktdataLen, 0);
        strcpy((char *)frame->get_pkt_hdr_addr(), "Hello World");
        frame->set_dest_ah(dest_ah);
        frame->set_dest_qpn(remote_meta->qpn_list[0]);
        socket->post_send_wr(frame);

        mid_time = std::chrono::high_resolution_clock::now();
        duration1 += mid_time - start_time;

        do {
            frames = socket->poll_send_cq(1);
        } while (frames.size() == 0);
        CHECK(frames.size() == 1);
        socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
        socket->push_pkt_data(frame->get_pkt_data_addr());
        socket->push_frame_desc((uint64_t)frame);
        CHECK_EQ(socket->send_queue_free_space(), kMaxRecvWr * kMaxPath);

        end_time = std::chrono::high_resolution_clock::now();
        duration2 += end_time - mid_time;
    }

    LOG(INFO)
        << "post_send_wr duration "
        << std::chrono::duration_cast<std::chrono::microseconds>(duration1)
                   .count() *
               1.0 / ITERATIONS
        << " us" << " poll_send_cq duration "
        << std::chrono::duration_cast<std::chrono::microseconds>(duration2)
                   .count() *
               1.0 / ITERATIONS
        << " us";

    // Receiving an ack ctrl packet.
    uint32_t finished_sends = 0;
    do {
        frames = socket->poll_ctrl_cq(1, finished_sends);
    } while (frames.size() == 0);
    CHECK(frames.size() == 1 && finished_sends == 0);
    frame = frames[0];
    CHECK(strcmp((char *)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
                 "Ctrl Packet") == 0);
    socket->push_pkt_hdr(frame->get_pkt_hdr_addr());
    socket->push_frame_desc((uint64_t)frame);
    CHECK(socket->recv_queue_wrs() >=
          kMaxSendWr * kMaxPath - kMaxRecvDeficitCnt * kMaxPath);
}

void run_client(const char *server_ip) {
    auto *socket = EFAFactory::CreateSocket(0, 0);

    auto local_meta = new ConnMeta();
    socket->get_conn_metadata(local_meta);
    auto remote_meta = new ConnMeta();
    exchange_qpns(server_ip, local_meta, remote_meta);

    auto *dev = EFAFactory::GetEFADevice(0);
    auto *dest_ah = dev->create_ah(remote_meta->gid);

    uint64_t frame_desc, pkt_hdr, pkt_data;
    FrameDesc *frame;
    std::vector<FrameDesc *> frames;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; i++) {
        // Send a data packet
        frame_desc = socket->pop_frame_desc();
        pkt_hdr = socket->pop_pkt_hdr();
        pkt_data = socket->pop_pkt_data();
        frame = FrameDesc::Create(frame_desc, pkt_hdr, kUcclPktHdrLen, pkt_data,
                                  kUcclPktdataLen, 0);
        frame->set_dest_ah(dest_ah);
        frame->set_dest_qpn(remote_meta->qpn_list[0]);
        strcpy((char *)pkt_hdr, "Hello World");
        socket->post_send_wr(frame);
        do {
            frames = socket->poll_send_cq(1);
        } while (frames.size() == 0);
        CHECK(frames.size() == 1);
        CHECK(frames[0] == frame);
        socket->push_frame_desc(frame_desc);
        socket->push_pkt_hdr(pkt_hdr);
        socket->push_pkt_data(pkt_data);
        CHECK_EQ(socket->send_queue_free_space(), kMaxSendWr * kMaxPath);

        // Receiving the packet back
        do {
            frames = socket->poll_recv_cq(1);
        } while (frames.size() == 0);
        CHECK(frames.size() == 1);
        frame = frames[0];
        CHECK(strcmp((char *)(frame->get_pkt_hdr_addr() + EFA_UD_ADDITION),
                     "Hello World") == 0);
        socket->push_frame_desc(frame_desc);
        socket->push_pkt_hdr(pkt_hdr);
        socket->push_pkt_data(pkt_data);
        CHECK(socket->recv_queue_wrs() >=
              kMaxSendWr * kMaxPath - kMaxRecvDeficitCnt * kMaxPath);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    LOG(INFO) << "Round trip time: " << duration_us.count() * 1.0 / ITERATIONS
              << " us";

    // Sending a ctrl packet
    frame_desc = socket->pop_frame_desc();
    pkt_hdr = socket->pop_pkt_hdr();
    frame = FrameDesc::Create(frame_desc, pkt_hdr,
                              kUcclPktHdrLen + kUcclSackHdrLen, 0, 0, 0);
    frame->set_dest_ah(dest_ah);
    frame->set_dest_qpn(remote_meta->qpn_list[kMaxPath]);  // ctrl qp
    strcpy((char *)pkt_hdr, "Ctrl Packet");
    socket->post_send_wr(frame);
    uint32_t finished_sends = 0;
    do {
        frames = socket->poll_ctrl_cq(1, finished_sends);
    } while (finished_sends == 0);
    CHECK(finished_sends == 1 && frames.size() == 0);
    // No need to push pkt_hdr and frame_desc, as poll_ctrl_cq() frees them.
    CHECK_EQ(socket->send_queue_free_space(), kMaxSendWr * kMaxPath);
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
