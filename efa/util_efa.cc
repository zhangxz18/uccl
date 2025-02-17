#include "util_efa.h"

#include "transport_config.h"

namespace uccl {

EFAFactory efa_ctl;

void EFAFactory::Init() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        EFAFactory::InitDev(GID_INDEX_LIST[i]);
    }
}

void EFAFactory::InitDev(int gid_idx) {
    struct EFADevice dev;
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    int i, nb_devices;

    // Check if the device is already initialized.
    DCHECK(efa_ctl->gid_2_dev_map.find(gid_idx) ==
           efa_ctl->gid_2_dev_map.end());

    // Get Infiniband name from GID index.
    DCHECK(util_rdma_get_ib_name_from_gididx(gid_idx, dev.ib_name) == 0);

    // Get IP address from GID index.
    DCHECK(util_rdma_get_ip_from_gididx(gid_idx, &dev.local_ip_str) == 0);

    // Get the list of RDMA devices.
    device_list = ibv_get_device_list(&nb_devices);
    if (device_list == nullptr || nb_devices == 0) {
        perror("ibv_get_device_list");
        goto error;
    }

    // Find the device by name.
    for (i = 0; i < nb_devices; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), dev.ib_name) == 0) {
            break;
        }
    }
    if (i == nb_devices) {
        fprintf(stderr, "No device found for %s\n", dev.ib_name);
        goto free_devices;
    }

    // Open the device.
    memset(&dev_attr, 0, sizeof(dev_attr));
    if ((context = ibv_open_device(device_list[i])) == nullptr) {
        perror("ibv_open_device");
        goto free_devices;
    }

    if (ibv_query_device(context, &dev_attr)) {
        perror("ibv_query_device");
        goto close_device;
    }

    // Currently, we only use one port.
    if (dev_attr.phys_port_cnt != EFA_PORT_NUM /* 1 */) {
        fprintf(stderr, "Only one port is supported\n");
        goto close_device;
    }

    // Port number starts from 1.
    if (ibv_query_port(context, 1, &port_attr)) {
        perror("ibv_query_port");
        goto close_device;
    }

    if (port_attr.state != IBV_PORT_ACTIVE) {
        fprintf(stderr, "Port is not active\n");
        goto close_device;
    }

    if (port_attr.link_layer != IBV_LINK_LAYER_UNSPECIFIED) {
        fprintf(stderr, "EFA link layer is not supported\n");
        goto close_device;
    }

    dev.dev_attr = dev_attr;
    dev.port_attr = port_attr;
    dev.EFA_PORT_NUM = EFA_PORT_NUM;
    dev.gid_idx = gid_idx;
    dev.context = context;

    if (ibv_query_gid(context, EFA_PORT_NUM, gid_idx, &dev.gid)) {
        perror("ibv_query_gid");
        goto close_device;
    }

    // Allocate a PD for this device.
    dev.pd = ibv_alloc_pd(context);
    if (dev.pd == nullptr) {
        perror("ibv_alloc_pd");
        goto close_device;
    }

    // Detect DMA-BUF support.
    {
        struct ibv_pd *pd;
        pd = ibv_alloc_pd(context);
        if (pd == nullptr) {
            perror("ibv_alloc_pd");
            goto close_device;
        }
        // Test kernel DMA-BUF support with a dummy call (fd=-1)
        (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
        dev.dma_buf_support =
            !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);

        VLOG(5) << "DMA-BUF support: " << dev.dma_buf_support;
    }

    efa_ctl->gid_2_dev_map.insert({gid_idx, efa_ctl->devices_.size()});
    efa_ctl->devices_.push_back(dev);
    return;

close_device:
    ibv_close_device(context);
free_devices:
    ibv_free_device_list(device_list);
error:
    throw std::runtime_error("Failed to initialize EFAFactory");
}

EFASocket *EFAFactory::CreateSocket(int socket_id) {
    auto socket = new EFASocket(socket_id);
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    efa_ctl.socket_q_.push_back(socket);
    return socket;
}

struct EFADevice *EFAFactory::GetEFADevice(int gid_idx) {
    auto dev_idx_iter = efa_ctl.gid_2_dev_map.find(gid_idx);
    DCHECK(dev_idx_iter != efa_ctl.gid_2_dev_map.end());
    auto dev_idx = *dev_idx_iter;
    DCHECK(dev_idx >= 0 && dev_idx < efa_ctl.devices_.size());
    return &efa_ctl.devices_[dev_idx];
}

void EFAFactory::Shutdown() {
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    for (auto socket : efa_ctl.socket_q_) {
        delete socket;
    }
    efa_ctl.socket_q_.clear();

    devices_.clear();
    gid_2_dev_map.clear();
}

EFASocket::EFASocket(int gid_idx, int socket_id) : next_qp_for_send_(0);
gid_idx_(gid_idx), socket_id_(socket_id), unpolled_send_wrs_(0),
    recv_queue_wrs_(0) {
    LOG(INFO) << "[AF_XDP] creating gid_idx " << gid_idx_ << " socket_id "
              << socket_id;

    auto *factory_dev = EFAFactory::GetEFADevice(gid_idx);
    context_ = factory_dev->context;
    pd_ = factory_dev->pd;

    // Allocate memory for packet headers.
    void *pkt_hdr_buf_ =
        mmap(nullptr, PktHdrBuffPool::kNumPktHdr * PktHdrBuffPool::kPktHdrSize,
             PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(pkt_hdr_buf_ != nullptr) << "aligned_alloc failed";
    auto *pkt_hdr_mr_ =
        ibv_reg_mr(pd_, pkt_hdr_buf_,
                   PktHdrBuffPool::kNumPktHdr * PktHdrBuffPool::kPktHdrSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(pkt_hdr_mr_ != nullptr) << "ibv_reg_mr failed";
    pkt_hdr_pool_ = new PktHdrBuffPool(pkt_hdr_mr_);

    // Allocate memory for packet data.
    void *pkt_data_buf_ = nullptr;
    auto cuda_ret =
        cudaMalloc(&pkt_data_buf_, PktDataBuffPool::kNumPktData *
                                       PktDataBuffPool::kPktDataSize);
    DCHECK(cuda_ret == cudaSuccess) << "cudaMalloc failed";
    auto pkt_data_mr_ =
        ibv_reg_mr(pd_, pkt_data_buf_,
                   PktDataBuffPool::kNumPktData * PktDataBuffPool::kPktDataSize,
                   IBV_ACCESS_LOCAL_WRITE);
    DCHECK(pkt_data_mr_ != nullptr) << "ibv_reg_mr failed";
    pkt_data_pool_ = new PktDataBuffPool(pkt_data_mr_);

    // Allocate memory for frame desc.
    void *frame_desc_buf_ = mmap(
        nullptr,
        FrameDescBuffPool::kNumFrameDesc * FrameDescBuffPool::kFrameDescSize,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    DCHECK(frame_desc_buf_ != nullptr) << "aligned_alloc failed";
    frame_desc_pool_ = new FrameDescBuffPool();

    // Create completion queue.
    send_cq_ = ibv_create_cq(context_, kMaxCqeTotal, NULL, NULL, 0);
    recv_cq_ = ibv_create_cq(context_, kMaxCqeTotal, NULL, NULL, 0);
    if (!send_cq_) {
        perror("Failed to allocate send/recv_cq_");
        exit(1);
    }
    // Create send/recv QPs.
    for (int i = 0; i < kMaxPath; i++) {
        qp_list_[i] = create_qp(send_cq_, recv_cq_, kMaxSendWr, kMaxRecvWr);
        refill_recv_queue_data(kMaxRecvWr, i);
    }

    // Create QP for ACK packets.
    ctrl_cq_ = ibv_create_cq(context_, kMaxCqeTotal, NULL, NULL, 0);
    if (!ctrl_cq_) {
        perror("Failed to allocate ctrl CQ");
        exit(1);
    }
    ctrl_qp_ =
        create_qp(ctrl_cq_, ctrl_cq_, kMaxSendRecvWrCtrl, kMaxSendRecvWrCtrl);
    refill_recv_queue_ctrl(kMaxRecvWr);
}

// Create and configure a UD QP
struct ibv_qp *EFASocket::create_qp(struct ibv_cq *send_cq,
                                    struct ibv_cq *recv_cq,
                                    uint32_t send_cq_size,
                                    uint32_t recv_cq_size) {
    struct ibv_qp_init_attr qp_attr = {};
    qp_attr.send_cq = send_cq;
    qp_attr.recv_cq = recv_cq;
    qp_attr.cap.max_send_wr = send_cq_size;
    qp_attr.cap.max_recv_wr = recv_cq_size;
    qp_attr.cap.max_send_sge = 2;
    qp_attr.cap.max_recv_sge = 2;

    qp_attr.qp_type = IBV_QPT_DRIVER;
    struct ibv_qp *qp = qp =
        efadv_create_driver_qp(rdma->pd, &qp_attr, EFADV_QP_DRIVER_TYPE_SRD);

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = EFA_PORT_NUM;
    attr.qkey = QKEY;
    if (ibv_modify_qp(
            qp, &attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        perror("Failed to modify QP to INIT");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
        perror("Failed to modify QP to RTR");
        exit(1);
    }

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = SQ_PSN;  // Set initial Send Queue PSN
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        perror("Failed to modify QP to RTS");
        exit(1);
    }

    return qp;
}

void EFASocket::create_ah(uint8_t remote_gid_idx, union ibv_gid remote_gid) {
    struct ibv_ah_attr ah_attr = {};

    ah_attr.is_global = 1;  // Enable Global Routing Header (GRH)
    ah_attr.port_num = EFA_PORT_NUM;
    ah_attr.grh.sgid_index = gid_idx_;  // Local GID index
    ah_attr.grh.dgid = remote_gid;      // Destination GID
    ah_attr.grh.flow_label = 0;
    ah_attr.grh.hop_limit = 255;
    ah_attr.grh.traffic_class = 0;

    struct ibv_ah *ah = ibv_create_ah(pd_, &ah_attr);
    if (!ah) {
        perror("Failed to create AH");
        exit(1);
    }

    DCHECK(remote_gid_idx <= 255) << "remote_gid_idx too large";
    ah_list_[remote_gid_idx] = ah;
}

uint32_t EFASocket::send_packet(FrameDesc *frame) {
    struct ibv_sge sge[2] = {{0}, {0}};
    struct ibv_send_wr send_wr = {0}, *bad_send_wr;
    struct ibv_qp *src_qp;

    auto dest_gid_idx = frame->get_dest_gid_idx();
    auto dest_qpn = frame->get_dest_qpn();

    // This is a ack packet.
    if (frame->get_pkt_data_len() == 0) {
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        send_wr.num_sge = 1;
        src_qp = ctrl_qp_;
    } else {  // This is a data packet
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                  get_pkt_data_lkey()};
        send_wr.num_sge = 2;
        src_qp = qp_list_[get_next_qp_idx_for_send()];
    }

    send_wr.wr_id = (uint64_t)frame;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.sg_list = sge;
    send_wr.wr.ud.ah = ah_list_[dest_gid_idx];
    send_wr.wr.ud.remote_qpn = dest_qpn;
    send_wr.wr.ud.remote_qkey = QKEY;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(src_qp, &send_wr, &bad_send_wr)) {
        perror("Server: Failed to post send");
        exit(1);
    }

    unpolled_send_wrs_++;
    return 1;
}

uint32_t EFASocket::send_packets(std::vector<FrameDesc *> &frames) {
    struct ibv_sge sge[2] = {{0}, {0}};
    struct ibv_send_wr send_wr = {0}, *bad_send_wr;
    struct ibv_qp *src_qp;

    for (auto *frame : frames) {
        auto dest_gid_idx = frame->get_dest_gid_idx();
        auto dest_qpn = frame->get_dest_qpn();
        auto src_qp_idx = get_next_qp_idx_for_send();

        // This is a ack packet.
        if (frame->get_pkt_data_len() == 0) {
            sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                      get_pkt_hdr_lkey()};
            send_wr.num_sge = 1;
            src_qp = ctrl_qp_;
        } else {  // This is a data packet
            sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                      get_pkt_hdr_lkey()};
            sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                      get_pkt_data_lkey()};
            send_wr.num_sge = 2;
            src_qp = qp_list_[get_next_qp_idx_for_send()];
        }

        send_wr.wr_id = (uint64_t)frame;
        send_wr.opcode = IBV_WR_SEND;
        send_wr.sg_list = sge;
        send_wr.wr.ud.ah = ah_list_[dest_gid_idx];
        send_wr.wr.ud.remote_qpn = dest_qpn;
        send_wr.wr.ud.remote_qkey = QKEY;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        // TODO(yang): Using chained post list

        if (ibv_post_send(src_qp, &send_wr, &bad_send_wr)) {
            perror("Server: Failed to post send");
            exit(1);
        }
    }

    unpolled_send_wrs_ += frames.size();
    return frames.size();
}

std::vector<FrameDesc *> EFASocket::poll_send_cq(uint32_t bugget) {
    std::vector<FrameDesc *> frames;
    frames.reserve(budget);

    while (frames.size() < budget) {
        int ne = ibv_poll_cq(send_cq_, kMaxBatchCQ, wc_);
        DCHECK(ne >= 0) << "poll_send_cq ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "poll_send_cq: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            DCHECK(wc_[i].opcode == IBV_WC_SEND);

            auto *frame = (FrameDesc *)wc_[i].wr_id;
            frames.push_back(frame);
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxBatchCQ) break;
    }

    return frames;
}

std::vector<FrameDesc *> EFASocket::recv_packets(uint32_t budget) {
    std::vector<FrameDesc *> frames;
    frames.reserve(budget);

    while (frames.size() < budget) {
        int ne = ibv_poll_cq(recv_cq_, kMaxBatchCQ, wc_);
        DCHECK(ne >= 0) << "recv_packets ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "recv_packets: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            DCHECK(wc_[i].opcode == IBV_WC_RECV);

            auto *frame = (FrameDesc *)wc_[i].wr_id;
            frames.push_back(frame);
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxBatchCQ) break;
    }

    recv_queue_wrs_ -= frames.size();
    return frames;
}

std::vector<FrameDesc *> EFASocket::recv_acks_and_poll_ctrl_cq(
    uint32_t budget) {
    std::vector<FrameDesc *> frames;
    frames.reserve(budget);

    while (frames.size() < budget) {
        int ne = ibv_poll_cq(ctrl_cq_, kMaxBatchCQ, wc_);
        DCHECK(ne >= 0) << "recv_acks ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "recv_acks: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            FrameDesc *frame = (FrameDesc *)wc_[i].wr_id;

            if (wc_[i].opcode == IBV_WC_RECV) {
                frames.push_back(frame);
            } else if (wc_[i].opcode == IBV_WC_SEND) {
                auto pkt_hdr_addr = frame->get_pkt_hdr_addr();
                pkt_hdr_pool_->free_buff(pkt_hdr_addr);
                frame_desc_pool_->free_buff((uint64_t)frame);
            } else {
                DCHECK(false) << "Wrong wc_[i].opcode: " << wc_[i].opcode;
            }
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxBatchCQ) break;
    }

    recv_queue_wrs_ -= frames.size();
    return frames;
}

void EFASocket::refill_recv_queue_data(uint32_t budget, uint32_t qp_idx) {
    int ret;
    uint64_t pkt_hdr_buf, pkt_data_buf, frame_desc_buf;
    auto *qp = qp_list_[qp_idx];

    for (int i = 0; i < budget; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= pkt_data_pool_->alloc_buff(&pkt_data_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0);

        auto *frame_desc = FrameDesc::Create(
            frame_desc_buf, pkt_hdr_buf, pkt_data_buf,
            PktHdrBuffPool::kPktHdrSize, PktDataBuffPool::kPktDataSize, 0);

        // recv size does not need to exactly match send size. But we need limit
        // the hdr sge to exactly match hdr size, so that we can split hdr and
        // data between GPU and CPU.
        struct ibv_sge sge[2] = {
            {(uintptr_t)pkt_hdr_buf, EFA_GRH_SIZE + kUcclPktHdrLen,
             get_pkt_hdr_lkey()},
            {(uintptr_t)pkt_data_buf, kUcclPktdataLen, get_pkt_data_lkey()}};

        struct ibv_recv_wr wr = {}, *bad_wr;
        wr.wr_id = (uint64_t)frame_desc;
        wr.num_sge = 2;
        wr.sg_list = sge;

        // Post receive buffer
        if (ibv_post_recv(qp, &wr, &bad_wr)) {
            perror("Failed to post recv");
            exit(1);
        }
    }
}
void EFASocket::refill_recv_queue_ctrl(uint32_t budget) {
    int ret;
    uint64_t pkt_hdr_buf, frame_desc_buf;
    auto *qp = ctrl_qp_;

    for (int i = 0; i < budget; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0);

        auto *frame_desc = FrameDesc::Create(frame_desc_buf, pkt_hdr_buf, 0,
                                             PktHdrBuffPool::kPktHdrSize, 0, 0);

        // recv size does not need to exactly match send size.
        struct ibv_sge sge[1] = {{(uintptr_t)pkt_hdr_buf,
                                  PktHdrBuffPool::kPktHdrSize,
                                  get_pkt_hdr_lkey()}};

        struct ibv_recv_wr wr = {}, *bad_wr;
        wr.wr_id = (uint64_t)frame_desc;
        wr.num_sge = 1;
        wr.sg_list = sge;

        // Post receive buffer
        if (ibv_post_recv(qp, &wr, &bad_wr)) {
            perror("Failed to post recv");
            exit(1);
        }
    }
}

std::string EFASocket::to_string() {
    std::string s;
    s += Format("free frames: %u, unpulled tx pkts: %u, fill queue entries: %u",
                frame_pool_->size(), unpolled_send_wrs_, recv_queue_wrs_);
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
    // pull_completion_queue to make sure all frames are tx successfully.
    while (unpolled_send_wrs_) {
        pull_completion_queue();
    }
}

EFASocket::~EFASocket() {
    delete pkt_hdr_pool_;
    delete pkt_data_pool_;
    delete frame_desc_pool_;
}
}  // namespace uccl
