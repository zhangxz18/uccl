#include "util_efa.h"

#include "transport_config.h"

namespace uccl {

EFAFactory efa_ctl;

void EFAFactory::Init() {
    for (int i = 0; i < NUM_DEVICES; i++) {
        EFAFactory::InitDev(i);
    }
}

void EFAFactory::InitDev(int dev_idx) {
    struct EFADevice *dev = new struct EFADevice();
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    int i, nb_devices;

    // Check if the device is already initialized.
    DCHECK(efa_ctl.dev_map.find(dev_idx) == efa_ctl.dev_map.end());

    // Get Infiniband name from dev_idx.
    DCHECK(util_efa_get_ib_name_from_dev_idx(dev_idx, dev->ib_name) == 0);

    // Get IP address from dev_idx.
    DCHECK(util_efa_get_ip_from_dev_idx(dev_idx, &dev->local_ip_str) == 0);

    // Get the list of RDMA devices.
    device_list = ibv_get_device_list(&nb_devices);
    if (device_list == nullptr || nb_devices == 0) {
        perror("ibv_get_device_list");
        goto error;
    }

    // Find the device by name.
    for (i = 0; i < nb_devices; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), dev->ib_name) == 0) {
            break;
        }
    }
    if (i == nb_devices) {
        fprintf(stderr, "No device found for %s\n", dev->ib_name);
        goto free_devices;
    }
    DCHECK(i == dev_idx);
    LOG(INFO) << "Found device: " << dev->ib_name << " at dev_idx " << i
              << " with gid_idx " << (uint32_t)EFA_GID_IDX;

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
    if (dev_attr.phys_port_cnt != IB_PORT_NUM /* 1 */) {
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

    dev->dev_attr = dev_attr;
    dev->port_attr = port_attr;
    dev->dev_idx = dev_idx;
    dev->ib_port_num = IB_PORT_NUM;
    dev->context = context;

    if (ibv_query_gid(context, IB_PORT_NUM, EFA_GID_IDX, &dev->gid)) {
        perror("ibv_query_gid");
        goto close_device;
    }

    // Allocate a PD for this device.
    dev->pd = ibv_alloc_pd(context);
    if (dev->pd == nullptr) {
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
        dev->dma_buf_support =
            !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);

        VLOG(5) << "DMA-BUF support: " << dev->dma_buf_support;
    }

    efa_ctl.dev_map.insert({dev_idx, dev});
    return;

close_device:
    ibv_close_device(context);
free_devices:
    ibv_free_device_list(device_list);
error:
    throw std::runtime_error("Failed to initialize EFAFactory");
}

EFASocket *EFAFactory::CreateSocket(int dev_idx, int socket_id) {
    auto socket = new EFASocket(dev_idx, socket_id);
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    efa_ctl.socket_q_.push_back(socket);
    return socket;
}

struct EFADevice *EFAFactory::GetEFADevice(int dev_idx) {
    auto dev_iter = efa_ctl.dev_map.find(dev_idx);
    DCHECK(dev_iter != efa_ctl.dev_map.end());
    auto *dev = dev_iter->second;
    return dev;
}

void EFAFactory::Shutdown() {
    std::lock_guard<std::mutex> lock(efa_ctl.socket_q_lock_);
    for (auto socket : efa_ctl.socket_q_) {
        delete socket;
    }
    efa_ctl.socket_q_.clear();

    efa_ctl.dev_map.clear();
}

EFASocket::EFASocket(int dev_idx, int socket_id)
    : next_qp_idx_for_send_(0),
      dev_idx_(dev_idx),
      socket_id_(socket_id),
      send_queue_wrs_(0),
      recv_queue_wrs_(0) {
    LOG(INFO) << "[AF_XDP] creating dev_idx_ " << dev_idx_ << " socket_id "
              << socket_id;

    memset(deficit_cnt_recv_wrs_, 0, sizeof(deficit_cnt_recv_wrs_));

    auto *factory_dev = EFAFactory::GetEFADevice(dev_idx);
    context_ = factory_dev->context;
    pd_ = factory_dev->pd;
    gid_ = factory_dev->gid;

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
    DCHECK(send_cq_ && recv_cq_) << "Failed to allocate send/recv_cq_";

    // Create send/recv QPs.
    for (int i = 0; i < kMaxPath; i++) {
        qp_list_[i] = create_qp(send_cq_, recv_cq_, kMaxSendWr, kMaxRecvWr);
        post_recv_wrs(kMaxRecvWr, i);
    }

    // Create QP for ACK packets.
    ctrl_cq_ = ibv_create_cq(context_, kMaxCqeTotal, NULL, NULL, 0);
    DCHECK(ctrl_cq_) << "Failed to allocate ctrl CQ";

    ctrl_qp_ = create_qp(ctrl_cq_, ctrl_cq_, kMaxSendRecvWrForCtrl,
                         kMaxSendRecvWrForCtrl);
    post_recv_wrs_for_ctrl(kMaxSendRecvWrForCtrl);
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

#ifdef USE_SRD
    qp_attr.qp_type = IBV_QPT_DRIVER;
    struct ibv_qp *qp = qp =
        efadv_create_driver_qp(pd_, &qp_attr, EFADV_QP_DRIVER_TYPE_SRD);
#else
    qp_attr.qp_type = IBV_QPT_UD;
    struct ibv_qp *qp = ibv_create_qp(pd_, &qp_attr);
#endif

    if (!qp) {
        perror("Failed to create QP");
        exit(1);
    }

    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = IB_PORT_NUM;
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

uint32_t EFASocket::post_send_wr(FrameDesc *frame) {
    struct ibv_sge sge[2] = {{0}, {0}};
    struct ibv_send_wr send_wr = {0}, *bad_send_wr;
    struct ibv_qp *src_qp;

    auto dest_ah = frame->get_dest_ah();
    auto dest_qpn = frame->get_dest_qpn();

    if (frame->get_pkt_data_len() == 0) {
        // This is a ack packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        send_wr.num_sge = 1;
        src_qp = ctrl_qp_;

        frame->set_src_qp_idx(kMaxPath);  // indicating ctrl qp
    } else {
        // This is a data packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                  get_pkt_data_lkey()};
        send_wr.num_sge = 2;
        auto src_qp_idx = get_next_qp_idx_for_send();
        src_qp = qp_list_[src_qp_idx];

        frame->set_src_qp_idx(src_qp_idx);
        send_queue_wrs_++;
    }

    send_wr.wr_id = (uint64_t)frame;
    send_wr.opcode = IBV_WR_SEND;
    send_wr.sg_list = sge;
    send_wr.wr.ud.ah = dest_ah;
    send_wr.wr.ud.remote_qpn = dest_qpn;
    send_wr.wr.ud.remote_qkey = QKEY;
    send_wr.send_flags = IBV_SEND_SIGNALED;

    if (ibv_post_send(src_qp, &send_wr, &bad_send_wr)) {
        perror("Server: Failed to post send");
        exit(1);
    }

    return 1;
}

uint32_t EFASocket::post_send_wrs(std::vector<FrameDesc *> &frames) {
    int i = 0;
    auto *wr_head = &send_wr_vec_[0];
    auto src_qp_idx = get_next_qp_idx_for_send();

    for (auto *frame : frames) {
        auto dest_ah = frame->get_dest_ah();
        auto dest_qpn = frame->get_dest_qpn();
        DCHECK(frame->get_pkt_data_len() != 0);
        frame->set_src_qp_idx(src_qp_idx);
        VLOG(3) << "post_send_wrs i " << i << " src_qp_idx " << src_qp_idx;

        auto *sge = send_sge_vec_[i % kMaxChainedWr];
        auto *wr = &send_wr_vec_[i % kMaxChainedWr];

        // This is a data packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};
        sge[1] = {frame->get_pkt_data_addr(), frame->get_pkt_data_len(),
                  get_pkt_data_lkey()};

        wr->num_sge = 2;
        wr->wr_id = (uint64_t)frame;
        wr->opcode = IBV_WR_SEND;
        wr->sg_list = sge;
        wr->wr.ud.ah = dest_ah;
        wr->wr.ud.remote_qpn = dest_qpn;
        wr->wr.ud.remote_qkey = QKEY;
        wr->send_flags = IBV_SEND_SIGNALED;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == frames.size();
        wr->next = is_last ? nullptr : &send_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_send_wr *bad_send_wr;
            if (ibv_post_send(qp_list_[src_qp_idx], wr_head, &bad_send_wr)) {
                perror("[util_efa]: Failed to post send wrs");
                exit(1);
            }
            if (i + 1 != frames.size()) {
                wr_head = &send_wr_vec_[(i + 1) % kMaxChainedWr];
                src_qp_idx = get_next_qp_idx_for_send();
            }
        }

        send_queue_wrs_++;
        i++;
    }

    return frames.size();
}

uint32_t EFASocket::post_send_wrs_for_ctrl(std::vector<FrameDesc *> &frames) {
    int i = 0;
    auto *wr_head = &send_wr_vec_[0];

    for (auto *frame : frames) {
        auto dest_ah = frame->get_dest_ah();
        auto dest_qpn = frame->get_dest_qpn();
        DCHECK(frame->get_pkt_data_len() == 0);
        frame->set_src_qp_idx(kMaxPath);  // indicating ctrl qp

        auto *sge = send_sge_vec_[i % kMaxChainedWr];
        auto *wr = &send_wr_vec_[i % kMaxChainedWr];

        // This is a ack packet.
        sge[0] = {frame->get_pkt_hdr_addr(), frame->get_pkt_hdr_len(),
                  get_pkt_hdr_lkey()};

        wr->num_sge = 1;
        wr->wr_id = (uint64_t)frame;
        wr->opcode = IBV_WR_SEND;
        wr->sg_list = sge;
        wr->wr.ud.ah = dest_ah;
        wr->wr.ud.remote_qpn = dest_qpn;
        wr->wr.ud.remote_qkey = QKEY;
        wr->send_flags = IBV_SEND_SIGNALED;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == frames.size();
        wr->next = is_last ? nullptr : &send_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_send_wr *bad_send_wr;
            if (ibv_post_send(ctrl_qp_, wr_head, &bad_send_wr)) {
                perror("[util_efa]: Failed to post send wrs for ctrl");
                exit(1);
            }
            if (i + 1 != frames.size())
                wr_head = &send_wr_vec_[(i + 1) % kMaxChainedWr];
        }

        i++;
    }

    return frames.size();
}

void EFASocket::post_recv_wrs(uint32_t budget, uint16_t qp_idx) {
    DCHECK(qp_idx < kMaxPath);
    auto &deficit_cnt = deficit_cnt_recv_wrs_[qp_idx];
    deficit_cnt += budget;
    if (deficit_cnt < kMaxRecvWrDeficit) return;

    int ret;
    uint64_t pkt_hdr_buf, pkt_data_buf, frame_desc_buf;
    auto *qp = qp_list_[qp_idx];

    auto *wr_head = &recv_wr_vec_[0];
    for (int i = 0; i < deficit_cnt; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= pkt_data_pool_->alloc_buff(&pkt_data_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0);

        auto *frame_desc = FrameDesc::Create(frame_desc_buf, pkt_hdr_buf,
                                             EFA_UD_ADDITION + kUcclPktHdrLen,
                                             pkt_data_buf, kUcclPktdataLen, 0);
        frame_desc->set_src_qp_idx(qp_idx);

        auto *sge = recv_sge_vec_[i % kMaxChainedWr];
        auto *wr = &recv_wr_vec_[i % kMaxChainedWr];

        // recv size does not need to exactly match send size. But we need limit
        // the hdr sge to exactly match hdr size, so that we can split hdr and
        // data between GPU and CPU.
        sge[0] = {(uintptr_t)pkt_hdr_buf, EFA_UD_ADDITION + kUcclPktHdrLen,
                  get_pkt_hdr_lkey()};
        sge[1] = {(uintptr_t)pkt_data_buf, kUcclPktdataLen,
                  get_pkt_data_lkey()};

        wr->wr_id = (uint64_t)frame_desc;
        wr->num_sge = 2;
        wr->sg_list = sge;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == deficit_cnt;
        wr->next = is_last ? nullptr : &recv_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_recv_wr *bad_wr;
            // Post receive buffer
            if (ibv_post_recv(qp, wr_head, &bad_wr)) {
                perror("Failed to post recv");
                exit(1);
            }
            if (i + 1 != deficit_cnt)
                wr_head = &recv_wr_vec_[(i + 1) % kMaxChainedWr];
        }
    }

    recv_queue_wrs_ += deficit_cnt;
    deficit_cnt = 0;
}

void EFASocket::post_recv_wrs_for_ctrl(uint32_t budget) {
    auto &deficit_cnt = deficit_cnt_recv_wrs_[kMaxPath];
    deficit_cnt += budget;
    VLOG(3) << "post_recv_wrs_for_ctrl deficit_cnt " << deficit_cnt;
    if (deficit_cnt < kMaxRecvWrDeficit) return;

    int ret;
    uint64_t pkt_hdr_buf, frame_desc_buf;
    auto *qp = ctrl_qp_;

    auto *wr_head = &recv_wr_vec_[0];
    for (int i = 0; i < deficit_cnt; i++) {
        ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_buf);
        ret |= frame_desc_pool_->alloc_buff(&frame_desc_buf);
        DCHECK(ret == 0);

        auto *frame_desc = FrameDesc::Create(
            frame_desc_buf, pkt_hdr_buf,
            EFA_UD_ADDITION + kUcclPktHdrLen + kUcclSackHdrLen, 0, 0, 0);
        frame_desc->set_src_qp_idx(kMaxPath);

        auto *sge = recv_sge_vec_[i % kMaxChainedWr];
        auto *wr = &recv_wr_vec_[i % kMaxChainedWr];

        // recv size does not need to exactly match send size.
        sge[0] = {(uintptr_t)pkt_hdr_buf,
                  EFA_UD_ADDITION + kUcclPktHdrLen + kUcclSackHdrLen,
                  get_pkt_hdr_lkey()};

        wr->wr_id = (uint64_t)frame_desc;
        wr->num_sge = 1;
        wr->sg_list = sge;

        bool is_last = (i + 1) % kMaxChainedWr == 0 || (i + 1) == deficit_cnt;
        wr->next = is_last ? nullptr : &recv_wr_vec_[(i + 1) % kMaxChainedWr];

        if (is_last) {
            struct ibv_recv_wr *bad_wr;
            // Post receive buffer
            if (ibv_post_recv(qp, wr_head, &bad_wr)) {
                perror("Failed to post recv");
                exit(1);
            }
            if (i + 1 != deficit_cnt)
                wr_head = &recv_wr_vec_[(i + 1) % kMaxChainedWr];
        }
    }

    deficit_cnt = 0;
}

std::vector<FrameDesc *> EFASocket::poll_send_cq(uint32_t budget) {
    std::vector<FrameDesc *> frames;

    while (frames.size() < budget) {
        int ne = ibv_poll_cq(send_cq_, kMaxPollBatch, wc_);
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
        if (ne < kMaxPollBatch) break;
    }

    send_queue_wrs_ -= frames.size();
    return frames;
}

std::vector<FrameDesc *> EFASocket::poll_recv_cq(uint32_t budget) {
    std::vector<FrameDesc *> frames;

    while (frames.size() < budget) {
        int ne = ibv_poll_cq(recv_cq_, kMaxPollBatch, wc_);
        DCHECK(ne >= 0) << "poll_recv_cq ibv_poll_cq error";

        for (int i = 0; i < ne; i++) {
            // Check the completion status.
            DCHECK(wc_[i].status == IBV_WC_SUCCESS)
                << "poll_recv_cq: completion error: "
                << ibv_wc_status_str(wc_[i].status);

            DCHECK(wc_[i].opcode == IBV_WC_RECV);

            auto *frame = (FrameDesc *)wc_[i].wr_id;
            frames.push_back(frame);

            auto src_qp_idx = frame->get_src_qp_idx();
            post_recv_wrs(1, src_qp_idx);
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxPollBatch) break;
    }

    recv_queue_wrs_ -= frames.size();
    return frames;
}

std::vector<FrameDesc *> EFASocket::poll_ctrl_cq(uint32_t budget,
                                                 uint32_t *polled_send_acks) {
    std::vector<FrameDesc *> frames;

    while (frames.size() + *polled_send_acks < budget) {
        int ne = ibv_poll_cq(ctrl_cq_, kMaxPollBatch, wc_);
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
                (*polled_send_acks)++;
            } else {
                DCHECK(false) << "Wrong wc_[i].opcode: " << wc_[i].opcode;
            }
        }

        // TODO(yang): do we need to do anything smarter?
        if (ne < kMaxPollBatch) break;
    }

    if (frames.size()) {
        post_recv_wrs_for_ctrl(frames.size());
    }
    return frames;
}

std::string EFASocket::to_string() {
    std::string s;
    s += Format(
        "free pkt hdr: %u, free pkt data: %u, free frame desc: %u, unpulled tx "
        "pkts: %u, fill queue entries: %u",
        pkt_hdr_pool_->avail_slots(), pkt_data_pool_->avail_slots(),
        frame_desc_pool_->avail_slots(), send_queue_wrs_, recv_queue_wrs_);
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
    while (send_queue_wrs_) {
        std::ignore = poll_send_cq(kMaxPollBatch);
    }
}

EFASocket::~EFASocket() {
    delete pkt_hdr_pool_;
    delete pkt_data_pool_;
    delete frame_desc_pool_;
}
}  // namespace uccl
