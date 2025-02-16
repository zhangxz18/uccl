#include "util_efa.h"

#include "transport_config.h"

namespace uccl {

EFAFactory efa_ctl;

void EFAFactory::Init() {}

EFASocket *EFAFactory::CreateSocket(int socket_id) {
    auto socket = new EFASocket(socket_id);
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    efa_ctl.socket_q_.push_back(socket);
    return socket;
}

void EFAFactory::Shutdown() {
    // eBPF program detaching is done by the afxdp daemon

    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    for (auto socket : efa_ctl.socket_q_) {
        delete socket;
    }
    efa_ctl.socket_q_.clear();
}

EFASocket::EFASocket(int socket_id)
    : unpulled_tx_pkts_(0), fill_queue_entries_(0) {
    // TODO(yang): negotiate with afxdp daemon for socket_id and num_frames.

    socket_id_ = socket_id;

    // initialize queues, or misterious queue sync problems will happen
    memset(&recv_queue_, 0, sizeof(recv_queue_));
    memset(&send_queue_, 0, sizeof(send_queue_));
    memset(&complete_queue_, 0, sizeof(complete_queue_));
    memset(&fill_queue_, 0, sizeof(fill_queue_));

    // Step1: retrieve the file descriptors for AF_XDP socket
    xsk_fd_ = efa_ctl.xsk_fds_[socket_id_];

    // Step2: map UMEM and build four rings for the AF_XDP socket
    int ret = create_efa_socket();
    CHECK_EQ(ret, 0) << "xsk_socket__create_shared failed, " << ret;

    LOG(INFO) << "[AF_XDP] socket " << socket_id << " successfully shared";

    // apply_setsockopt(xsk_fd_);

    populate_fill_queue(FILL_RING_SIZE);
}

void EFASocket::destroy_efa_socket() {}

/**
 * @brief: Manually map UMEM and build four rings for a AF_XDP socket
 * @note: (RX/TX/FILL/COMP_RING_SIZE, NUM_FRAMES, FRAME_SIZE) need negotiating
 * with privileged processes
 */
int EFASocket::create_efa_socket() {
    struct xsk_ring_cons *rx = &recv_queue_;
    struct xsk_ring_prod *tx = &send_queue_;
    struct xsk_ring_prod *fill = &fill_queue_;
    struct xsk_ring_cons *comp = &complete_queue_;
    struct xdp_mmap_offsets off;

    /* initialize frame allocator */
    uint64_t frame_pool_size = efa_ctl.num_frames_ / NUM_QUEUES;
    frame_pool_ = new SharedPool<uint64_t, /*Sync=*/true>(frame_pool_size);
    uint64_t frame_pool_offset = FRAME_SIZE * frame_pool_size * socket_id_;
    LOG(INFO) << "[AF_XDP] frame pool " << socket_id_
              << " initialized: frame_pool_size = " << frame_pool_size
              << " frame_pool_offset = " << std::hex << "0x"
              << frame_pool_offset;
    for (uint64_t i = 0; i < frame_pool_size; i++) {
        uint64_t frame_offset = frame_pool_offset + XDP_PACKET_HEADROOM;
        push_frame(frame_offset);
        frame_pool_offset += FRAME_SIZE;
    }
    // Flushing the cache to prevent one socket pool's entries from being pushed
    // to another socket pool, in case socket creatations are done by a single
    // main thread.
    frame_pool_->flush_th_cache();

    return 0;
}

uint32_t EFASocket::pull_complete_queue() {
    uint32_t idx_cq, completed;
    completed = xsk_ring_cons__peek(&complete_queue_,
                                    XSK_RING_CONS__DEFAULT_NUM_DESCS, &idx_cq);
    if (completed > 0) {
        for (int i = 0; i < completed; i++) {
            uint64_t frame_offset =
                *xsk_ring_cons__comp_addr(&complete_queue_, idx_cq++);

            DCHECK((frame_offset & XDP_PACKET_HEADROOM_MASK) ==
                   XDP_PACKET_HEADROOM)
                << std::hex << frame_offset;

            // TODO(yang): why collecting stats here is smaller than at tx time?
            // out_bytes_ +=
            //     FrameDesc::get_frame_len(frame_offset, umem_buffer_);
            // TODO(yang): why will this trigger SEGV? Seems kernel bug.
            // out_bytes_ +=
            //     FrameDesc::get_uccl_frame_len(frame_offset, umem_buffer_);

            if (FrameDesc::is_txpulltime_free(frame_offset, umem_buffer_)) {
                push_frame(frame_offset);
                /**
                 * Yang: I susspect this is a bug that AWS ENA driver will pull
                 * the same frame multiple times. A temp fix is we mark it as
                 * not txpulltime free, so that the next time polling will not
                 * double free it.
                 */
                FrameDesc::mark_not_txpulltime_free(frame_offset, umem_buffer_);
            }
            // In other cases, the transport layer should handle frame freeing.
        }
        // out_packets_ += completed;

        xsk_ring_cons__release(&complete_queue_, completed);
        unpulled_tx_pkts_ -= completed;
    }
    VLOG(2) << "tx complete_queue completed = " << completed
            << " unpulled_tx_pkts = " << unpulled_tx_pkts_;
    return completed;
}

uint32_t EFASocket::send_packet(frame_desc frame) {
    // reserving a slot in the send queue.
    uint32_t send_index;
    VLOG(2) << "tx send_packets num_frames = " << 1;
    while (xsk_ring_prod__reserve(&send_queue_, 1, &send_index) != 1) {
        LOG_EVERY_N(WARNING, 1000000)
            << "send_queue is full. Busy waiting... unpulled_tx_pkts "
            << unpulled_tx_pkts_ << " send_queue_free_entries "
            << send_queue_free_entries();
        kick_tx();
    }
    struct xdp_desc *desc = xsk_ring_prod__tx_desc(&send_queue_, send_index);
    desc->addr = frame.frame_offset;
    desc->len = frame.frame_len;
    xsk_ring_prod__submit(&send_queue_, 1);
    unpulled_tx_pkts_++;

    out_bytes_ += frame.frame_len;
    out_packets_++;

    return kick_tx_and_pull();
}

uint32_t EFASocket::send_packets(std::vector<frame_desc> &frames) {
    // reserving slots in the send queue.
    uint32_t send_index;
    auto num_frames = frames.size();
    VLOG(2) << "tx send_packets num_frames = " << num_frames;
    while (xsk_ring_prod__reserve(&send_queue_, num_frames, &send_index) !=
           num_frames) {
        LOG_EVERY_N(WARNING, 1000000)
            << "send_queue is full. Busy waiting... unpulled_tx_pkts "
            << unpulled_tx_pkts_ << " send_queue_free_entries "
            << send_queue_free_entries() << " num_frames " << num_frames;
        kick_tx();
    }
    for (int i = 0; i < num_frames; i++) {
        struct xdp_desc *desc =
            xsk_ring_prod__tx_desc(&send_queue_, send_index++);
        desc->addr = frames[i].frame_offset;
        desc->len = frames[i].frame_len;

        out_bytes_ += frames[i].frame_len;
    }
    out_packets_ += num_frames;
    xsk_ring_prod__submit(&send_queue_, num_frames);
    unpulled_tx_pkts_ += num_frames;

    return kick_tx_and_pull();
}

void EFASocket::populate_fill_queue(uint32_t budget) {
    // TODO(yang): figure out why cloudlab needs xsk_prod_nb_free().
#if defined(AWS_C5) || defined(AWS_G4) || defined(AWS_G4METAL)
    auto stock_frames = budget;
#else
    auto stock_frames = xsk_prod_nb_free(&fill_queue_, budget);
#endif
    if (stock_frames <= 0) return;

    uint32_t idx_fq;
    int ret = xsk_ring_prod__reserve(&fill_queue_, stock_frames, &idx_fq);
    if (ret <= 0) return;

    for (int i = 0; i < ret; i++)
        *xsk_ring_prod__fill_addr(&fill_queue_, idx_fq++) = pop_frame();

    fill_queue_entries_ += ret;
    VLOG(2) << "afxdp reserved fill_queue slots = " << ret
            << " fill_queue_entries_ = " << fill_queue_entries_;

    xsk_ring_prod__submit(&fill_queue_, ret);
}

std::vector<EFASocket::frame_desc> EFASocket::recv_packets(uint32_t budget) {
    std::vector<EFASocket::frame_desc> frames;
    frames.reserve(budget);
    uint32_t idx_rx, rcvd;
    rcvd = xsk_ring_cons__peek(&recv_queue_, budget, &idx_rx);
    if (!rcvd) {
        kick_rx();
        return frames;
    }
    fill_queue_entries_ -= rcvd;
    VLOG(2) << "rx recv_packets num_frames = " << rcvd
            << " fill_queue_entries_ = " << fill_queue_entries_;

    for (int i = 0; i < rcvd; i++) {
        const struct xdp_desc *desc =
            xsk_ring_cons__rx_desc(&recv_queue_, idx_rx++);

        /**
         * Yang: Under AFXDP zerocopy mode, XDP_TX'ed packets by the XDP hook
         * will trigger spurious packet receiving behavior. This should be
         * caused by some subtle kernel bugs. We temporarily work around this by
         * filtering out these packets who normally have a wrong offset.
         */
        if (desc->addr & XDP_PACKET_HEADROOM_MASK != XDP_PACKET_HEADROOM) {
            LOG_EVERY_N(WARNING, 1000000)
                << "Received a frame with wrong offset: 0x" << std::hex
                << desc->addr;
            // auto frame_offset = desc->addr;
            // frame_offset &= ~XDP_PACKET_HEADROOM_MASK;
            // frame_offset |= XDP_PACKET_HEADROOM;
            // push_frame(frame_offset);
            continue;
        }

        frames.push_back({desc->addr, desc->len});
        in_bytes_ += desc->len;
    }
    in_packets_ += rcvd;

    xsk_ring_cons__release(&recv_queue_, rcvd);

    // Do filling even it is a small batch to control tail latency.
    populate_fill_queue(rcvd);

    return frames;
}

std::string EFASocket::to_string() {
    std::string s;
    s += Format("free frames: %u, unpulled tx pkts: %u, fill queue entries: %u",
                frame_pool_->size(), unpulled_tx_pkts_, fill_queue_entries_);
    if (socket_id_ == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           now - last_stat_)
                           .count();
        last_stat_ = now;

        auto out_packets_rate = (double)out_packets_.load() / elapsed;
        auto out_bytes_rate = (double)out_bytes_.load() / elapsed / 1000 * 8;
        auto in_packets_rate = (double)in_packets_.load() / elapsed;
        auto in_bytes_rate = (double)in_bytes_.load() / elapsed / 1000 * 8;
        out_packets_ = 0;
        out_bytes_ = 0;
        in_packets_ = 0;
        in_bytes_ = 0;

        s += Format(
            "\n\t\t\t        total in: %lf Mpps, %lf Gbps; total out: %lf "
            "Mpps, %lf Gbps",
            in_packets_rate, in_bytes_rate, out_packets_rate, out_bytes_rate);
    }
    return s;
}

void EFASocket::shutdown() {
    // pull_complete_queue to make sure all frames are tx successfully.
    while (unpulled_tx_pkts_) {
        kick_tx();
        pull_complete_queue();
    }
}

EFASocket::~EFASocket() {
    destroy_efa_socket();
    delete frame_pool_;
}
}  // namespace uccl
