#include "util_rdma.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h> 

#include <infiniband/verbs.h>

#include <glog/logging.h>

#include "transport_config.h"

namespace uccl {

RDMAFactory rdma_ctl;

void RDMAFactory::init_dev(int gid_idx)
{
    struct FactoryDevice dev;
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    int i, nb_devices;

    // Check if the device is already initialized.
    DCHECK(rdma_ctl.gid_2_dev_map.find(gid_idx) == rdma_ctl.gid_2_dev_map.end());
    
    // Get Infiniband name from GID index.
    DCHECK(util_rdma_get_ib_name_from_gididx(gid_idx, dev.ib_name) == 0);

    // Get IP address from Infiniband name.
    if (!SINGLE_IP.empty())
        dev.local_ip_str = SINGLE_IP;
    else
        DCHECK(util_rdma_get_ip_from_ib_name(dev.ib_name, &dev.local_ip_str) == 0);

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

    if (USE_ROCE && port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        fprintf(stderr, "RoCE is not supported\n");
        goto close_device;
    } else if (!USE_ROCE && port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
        fprintf(stderr, "IB is not supported\n");
        goto close_device;
    }

    dev.dev_attr = dev_attr;
    dev.port_attr = port_attr;
    dev.ib_port_num = IB_PORT_NUM;
    dev.gid_idx = gid_idx;
    dev.context = context;
    
    if (ibv_query_gid(context, IB_PORT_NUM, gid_idx, &dev.gid)) {
        perror("ibv_query_gid");
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
        (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
        dev.dma_buf_support = !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);

        LOG(INFO) << "DMA-BUF support: " << dev.dma_buf_support;
    }

    rdma_ctl.gid_2_dev_map.insert({gid_idx, rdma_ctl.devices_.size()});
    
    rdma_ctl.devices_.push_back(dev);

    return;

close_device:
    ibv_close_device(context);

free_devices:
    ibv_free_device_list(device_list);
error:
    throw std::runtime_error("Failed to initialize RDMAFactory");
}

/**
 * @brief This function frees all resources allocated by the RDMAFactory, including:
 *  - All RDMA contexts
 *  - Resources allocated in init_dev().
 */
void RDMAFactory::shutdown(void) 
{
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    for (auto ctx : rdma_ctl.context_q_) {
        delete ctx;
    }
    rdma_ctl.context_q_.clear();
    
    rdma_ctl.devices_.clear();
    rdma_ctl.gid_2_dev_map.clear();
}

struct FactoryDevice *RDMAFactory::get_factory_dev(int dev)
{
    return &rdma_ctl.devices_[dev];
}

/**
 * @brief Create a new RDMA context for a given device running on a specific engine.
 * 
 * @param dev 
 * @param meta 
 * @return RDMAContext* 
 */
RDMAContext *RDMAFactory::CreateContext(int dev, struct XchgMeta meta)
{
    RDMAContext *ctx = new RDMAContext(dev, meta);
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    rdma_ctl.context_q_.push_back(ctx);
    return ctx;
}

RDMAContext::RDMAContext(int dev, struct XchgMeta meta):
    dev_(dev), ready_entropy_cnt_(0), wheel_({freq_ghz})
{
    auto *factory_dev = RDMAFactory::get_factory_dev(dev);

    is_send_ = meta.ToEngine.is_send;

    memset(&send_comm_, 0, sizeof(send_comm_));
    memset(&recv_comm_, 0, sizeof(recv_comm_));

    auto comm_base = is_send_ ? &send_comm_.base : &recv_comm_.base;

    // Copy fields from FactoryDevice.
    context_ = factory_dev->context;
    local_gid_ = factory_dev->gid;
    ib_port_num_ = factory_dev->ib_port_num;
    sgid_index_ = factory_dev->gid_idx;
    port_attr_ = factory_dev->port_attr;

    // Copy fields from from Endpoint.
    // fifo_key and fifo_addr are set later.
    comm_base->remote_ctx.remote_gid = meta.ToEngine.remote_gid;
    comm_base->remote_ctx.remote_port_attr = meta.ToEngine.remote_port_attr;
    
    mtu_ = meta.ToEngine.mtu;
    mtu_bytes_ = util_rdma_get_mtu_from_ibv_mtu(mtu_);

    // Crate PD.
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");

    // Create a dedicated CQ for UC QPs.
    struct ibv_cq_init_attr_ex cq_ex_attr;
    cq_ex_attr.cqe = kCQSize;
    cq_ex_attr.cq_context = nullptr;
    cq_ex_attr.channel = nullptr;
    cq_ex_attr.comp_vector = 0;
    cq_ex_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM | IBV_WC_EX_WITH_SRC_QP | 
        IBV_WC_EX_WITH_COMPLETION_TIMESTAMP; // Timestamp support.
    
    if (kTestNoHWTimestamp)
        cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;

    cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
    cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED | IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
    cq_ex_ = ibv_create_cq_ex(context_, &cq_ex_attr);
    if (cq_ex_ == nullptr)
        throw std::runtime_error("ibv_create_cq_ex failed");
    
    // Configure CQ moderation.
    struct ibv_modify_cq_attr cq_attr;
    cq_attr.attr_mask = IBV_CQ_ATTR_MODERATE;
    cq_attr.moderate.cq_count = kCQMODCount;
    cq_attr.moderate.cq_period = kCQMODPeriod;

    int ret = ibv_modify_cq(ibv_cq_ex_to_cq(cq_ex_), &cq_attr);
    if (ret) {
        throw std::runtime_error("ibv_modify_cq failed");
    }

    // Create up to kPortEntropy UC QPs.
    for (int i = 0; i < kPortEntropy; i++) {
        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));

        qp_init_attr.qp_context = this;
        qp_init_attr.send_cq = ibv_cq_ex_to_cq(cq_ex_);
        qp_init_attr.recv_cq = ibv_cq_ex_to_cq(cq_ex_);
        qp_init_attr.qp_type = kTestRC ? IBV_QPT_RC : IBV_QPT_UC;

        /// FIXME: max_send_wr and max_recv_wr
        qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv + kMaxRetr;
        qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv + kMaxRetr;
        qp_init_attr.cap.max_send_sge = kMaxSge;
        qp_init_attr.cap.max_recv_sge = kMaxSge;
        qp_init_attr.cap.max_inline_data = 0;

        struct ibv_qp *qp = ibv_create_qp(pd_, &qp_init_attr);
        if (qp == nullptr)
            throw std::runtime_error("ibv_create_qp failed");
        
        uc_qps_[i].local_psn = 0xDEADBEEF + i;
        
        // Modify QP state to INIT.
        struct ibv_qp_attr qpAttr;
        memset(&qpAttr, 0, sizeof(qpAttr));
        qpAttr.qp_state = IBV_QPS_INIT;
        qpAttr.pkey_index = 0;
        qpAttr.port_num = ib_port_num_;
        qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            throw std::runtime_error("ibv_modify_qp failed");
        }

        uc_qps_[i].qp = qp;

        uc_qps_[i].txtracking.set_rdma_ctx(this);

        INIT_LIST_HEAD(&uc_qps_[i].ack.ack_link); 
        uc_qps_[i].ack.qpidx = i;
    }

    // Initialize work request extension buffer pool.
    wr_ex_pool_.emplace();

    // Create Ctrl QP, CQ, and MR.
    ctrl_local_psn_ = 0xDEADBEEF + kPortEntropy;
    util_rdma_create_qp(this, context_, &ctrl_qp_, IBV_QPT_UC, true, true,
        (struct ibv_cq **)&ctrl_cq_ex_, false, kCQSize, pd_, &ctrl_mr_, nullptr, kCtrlMRSize, 
            CtrlPktBuffPool::kNumPkt, CtrlPktBuffPool::kNumPkt, 1, 1);
    
    // Initialize Control packet buffer pool.
    ctrl_pkt_pool_.emplace(ctrl_mr_);

    // Create FIFO QP, CQ, and MR.
    fifo_local_psn_ = 0xDEADBEEF + kPortEntropy + 1;
    util_rdma_create_qp(this, context_, &fifo_qp_, IBV_QPT_RC, false, false,
        &fifo_cq_, false, kCQSize, pd_, &fifo_mr_, nullptr, kFifoMRSize, 
            kMaxReq * kMaxRecv, kMaxReq * kMaxRecv, 1, 1);

    comm_base->fifo = reinterpret_cast<struct RemFifo *>(fifo_mr_->addr);

    // Create Retr QP, CQ and MR.
    retr_local_psn_ = 0xDEADBEEF + kPortEntropy + 2;
    util_rdma_create_qp(this, context_, &retr_qp_, IBV_QPT_UC, true, false,
        (struct ibv_cq **)&retr_cq_ex_, false, kCQSize, pd_, &retr_mr_, nullptr, kRetrMRSize, 
            RetrChunkBuffPool::kNumChunk, RetrChunkBuffPool::kNumChunk, 2, 1);
    void *retr_hdr = mmap(nullptr, RetrHdrBuffPool::kNumHdr * RetrHdrBuffPool::kHdrSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (retr_hdr == MAP_FAILED)
        throw std::runtime_error("mmap failed");
    retr_hdr_mr_ = ibv_reg_mr(pd_, retr_hdr, RetrHdrBuffPool::kNumHdr * RetrHdrBuffPool::kHdrSize, 
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (retr_hdr_mr_ == nullptr)
        throw std::runtime_error("ibv_reg_mr failed");
    
    // Initialize retransmission chunk and header buffer pool.
    {
        retr_chunk_pool_.emplace(retr_mr_);
        retr_hdr_pool_.emplace(retr_hdr_mr_);
    }

    if (!is_send_) {
        // Create QP, MR for GPU flush, share the same CQ with UC QPs.
        util_rdma_create_qp(this, context_, &gpu_flush_qp_, IBV_QPT_RC, false, false,
            (struct ibv_cq **)&cq_ex_, true, 0, pd_, &gpu_flush_mr_, &gpu_flush_, sizeof(int), kMaxReq * kMaxRecv, kMaxReq * kMaxRecv, kMaxSge, kMaxSge);

        gpu_flush_sge_.addr = (uint64_t)&gpu_flush_;
        gpu_flush_sge_.length = 1;
        gpu_flush_sge_.lkey = gpu_flush_mr_->lkey;
    }

    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));

    // Populate recv work requests on all UC QPs for consuming immediate data.
    if (!kTestRC) {
        for (int i = 0; i < kPortEntropy; i++) {
            auto qp = uc_qps_[i].qp;
            /// FIXME: 
            for (int i = 0; i < kMaxReq * kMaxRecv + kMaxRetr; i++) {
                struct ibv_recv_wr *bad_wr;
                DCHECK(ibv_post_recv(qp, &wr, &bad_wr) == 0);
            }
        }
    }

    // Populate recv work requests on Ctrl QP for consuming control packets if we are sender.
    if (is_send_)
    {
        struct ibv_sge sge;
        for (int i = 0; i < CtrlPktBuffPool::kNumPkt - 1; i++) {
            uint64_t pkt_addr;
            if (ctrl_pkt_pool_->alloc_buff(&pkt_addr))
                throw std::runtime_error("Failed to allocate buffer for control packet");
            sge.addr = pkt_addr;
            sge.length = CtrlPktBuffPool::kPktSize;
            sge.lkey = ctrl_pkt_pool_->get_lkey();
            wr.wr_id = pkt_addr;
            wr.next = nullptr;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            struct ibv_recv_wr *bad_wr;
            if (ibv_post_recv(ctrl_qp_, &wr, &bad_wr))
                throw std::runtime_error("ibv_post_recv failed");
        }
    }

    // Populate recv work requestrs on Retr QP for consuming retransmission chunks if we are receiver.
    if (!is_send_) 
    {
        struct ibv_sge sge;
        for (int i = 0; i < kMaxRetr; i++) {
            uint64_t chunk_addr;
            if (retr_chunk_pool_->alloc_buff(&chunk_addr))
                throw std::runtime_error("Failed to allocate buffer for retransmission chunk");
            sge.addr = chunk_addr;
            sge.length = RetrChunkBuffPool::kRetrChunkSize;
            sge.lkey = retr_chunk_pool_->get_lkey();
            wr.wr_id = chunk_addr;
            wr.next = nullptr;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            struct ibv_recv_wr *bad_wr;
            if (ibv_post_recv(retr_qp_, &wr, &bad_wr))
                throw std::runtime_error("ibv_post_recv failed");
        }
    }

    // Timing wheel.
    wheel_.catchup();
}

RDMAContext::~RDMAContext()
{
    if (gpu_flush_mr_ != nullptr) {
        munmap(gpu_flush_mr_->addr, gpu_flush_mr_->length);
        ibv_dereg_mr(gpu_flush_mr_);
    }

    if (gpu_flush_qp_ != nullptr) {
        ibv_destroy_qp(gpu_flush_qp_);
    }

    if (ctrl_mr_ != nullptr) {
        munmap(ctrl_mr_->addr, ctrl_mr_->length);
        ibv_dereg_mr(ctrl_mr_);
    }
    if (ibv_cq_ex_to_cq(ctrl_cq_ex_) != nullptr) {
        ibv_destroy_cq(ibv_cq_ex_to_cq(ctrl_cq_ex_));
    }
    if (ctrl_qp_ != nullptr) {
        ibv_destroy_qp(ctrl_qp_);
    }

    if (retr_mr_ != nullptr) {
        munmap(retr_mr_->addr, retr_mr_->length);
        ibv_dereg_mr(retr_mr_);
    }

    if (retr_hdr_mr_ != nullptr) {
        munmap(retr_hdr_mr_->addr, retr_hdr_mr_->length);
        ibv_dereg_mr(retr_hdr_mr_);
    }

    if (fifo_mr_ != nullptr) {
        munmap(fifo_mr_->addr, fifo_mr_->length);
        ibv_dereg_mr(fifo_mr_);
    }

    if (fifo_cq_ != nullptr) {
        ibv_destroy_cq(fifo_cq_);
    }

    if (fifo_qp_ != nullptr) {
        ibv_destroy_qp(fifo_qp_);
    }

    for (int i = 0; i < kPortEntropy; i++) {
        ibv_destroy_qp(uc_qps_[i].qp);
    }

    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
    }
    if (ibv_cq_ex_to_cq(cq_ex_) != nullptr) {
        ibv_destroy_cq(ibv_cq_ex_to_cq(cq_ex_));
    }

    LOG(INFO) << "RDMAContext destroyed";
}

uint64_t TXTracking::ack_chunks(uint32_t num_acked_chunks)
{
    DCHECK(num_acked_chunks <= unacked_chunks_.size());

    uint64_t timestamp = 0;
    
    while (num_acked_chunks) {
        auto &chunk = unacked_chunks_.front();
        if (--chunk.req->events == 0) {
            auto poll_ctx = chunk.req->poll_ctx;
            // Wakeup app thread waiting one endpoint
            {
                std::lock_guard<std::mutex> lock(poll_ctx->mu);
                poll_ctx->done = true;
                poll_ctx->cv.notify_one();
            }
            LOG(INFO) << "Message complete";
            // Free the request.
            rdma_ctx_->free_request(chunk.req);
        }

        // Free wr_ex here.
        rdma_ctx_->wr_ex_pool_->free_buff(reinterpret_cast<uint64_t>(chunk.wr_ex));

        if (timestamp == 0)
            timestamp = chunk.timestamp;
        unacked_chunks_.erase(unacked_chunks_.begin());
        num_acked_chunks--;
    }

    return timestamp;
}

} // namespace uccl