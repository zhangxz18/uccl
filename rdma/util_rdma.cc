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

    // Currently, we only support RoCEv2.
    if (port_attr.state != IBV_PORT_ACTIVE ||
        (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)) {
        fprintf(stderr, "Port is not active or not Ethernet\n");
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
RDMAContext *RDMAFactory::CreateContext(int dev, struct RDMAExchangeFormatLocal meta)
{
    RDMAContext *ctx = new RDMAContext(dev, meta);
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    rdma_ctl.context_q_.push_back(ctx);
    return ctx;
}

RDMAContext::RDMAContext(int dev, struct RDMAExchangeFormatLocal meta):
    dev_(dev), sync_cnt_(0),ctrl_pkt_pool_()
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

    // Copy fields from from Endpoint.
    // fifo_key and fifo_addr are set later.
    comm_base->remote_ctx.remote_gid = meta.ToEngine.remote_gid;
    
    mtu_ = meta.ToEngine.mtu;

    qp_vec_.resize(kPortEntropy);
    local_psn_.resize(kPortEntropy);
    remote_psn_.resize(kPortEntropy);

    // Crate PD.
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");

    // Create a dedicated CQ for UC QPs.
    cq_ = ibv_create_cq(context_, kCQSize, nullptr, nullptr, 0);
    if (cq_ == nullptr)
        throw std::runtime_error("ibv_create_cq failed");
    
    // Configure CQ moderation.
    struct ibv_modify_cq_attr cq_attr;
    cq_attr.attr_mask = IBV_CQ_ATTR_MODERATE;
    cq_attr.moderate.cq_count = kCQMODCount;
    cq_attr.moderate.cq_period = kCQMODPeriod;

    int ret = ibv_modify_cq(cq_, &cq_attr);
    if (ret) {
        throw std::runtime_error("ibv_modify_cq failed");
    }

    // Create up to kPortEntropy UC QPs.
    for (int i = 0; i < kPortEntropy; i++) {
        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));

        qp_init_attr.qp_context = this;
        qp_init_attr.send_cq = cq_;
        qp_init_attr.recv_cq = cq_;
        qp_init_attr.qp_type = IBV_QPT_UC;

        qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv;
        qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv;
        qp_init_attr.cap.max_send_sge = kMaxSge;
        qp_init_attr.cap.max_recv_sge = kMaxSge;
        qp_init_attr.cap.max_inline_data = 0;

        struct ibv_qp *qp = ibv_create_qp(pd_, &qp_init_attr);
        if (qp == nullptr)
            throw std::runtime_error("ibv_create_qp failed");
        
        local_psn_[i] = 0xDEADBEEF + i;
        
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

        qp_vec_[i] = qp;
    }

    ctrl_local_psn_ = 0xDEADBEEF + kPortEntropy;
    util_rdma_create_qp(this, context_, &ctrl_qp_, IBV_QPT_UC, 
        &ctrl_cq_, kCQSize, pd_, &ctrl_mr_, kCtrlMRSize, 
            kMaxReq * kMaxRecv, kMaxReq * kMaxRecv);

    retr_local_psn_ = 0xDEADBEEF + kPortEntropy + 1;
    util_rdma_create_qp(this, context_, &retr_qp_, IBV_QPT_RC, 
        &retr_cq_, kCQSize, pd_, &retr_mr_, kRetrMRSize, 
            kMaxReq * kMaxRecv, kMaxReq * kMaxRecv);

    fifo_local_psn_ = 0xDEADBEEF + kPortEntropy + 2;
    util_rdma_create_qp(this, context_, &fifo_qp_, IBV_QPT_RC, 
        &fifo_cq_, kCQSize, pd_, &fifo_mr_, kFifoMRSize, 
            kMaxReq * kMaxRecv, kMaxReq * kMaxRecv);

    comm_base->fifo = reinterpret_cast<struct RemFifo *>(fifo_mr_->addr);

    // Populate recv work requests on all UC QPs for consuming immediate data.
    struct ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    for (int i = 0; i < kPortEntropy; i++) {
        auto qp = qp_vec_[i];
        for (int i = 0; i < kMaxReq * kMaxRecv; i++) {
            struct ibv_recv_wr *bad_wr;
            DCHECK(ibv_post_recv(qp, &wr, &bad_wr) == 0);
        }
    }

    // Set buffer pool address for control packets.
    ctrl_pkt_pool_.set_pool_addr(ctrl_mr_);

    // Populate recv work requests on Ctrl QP for consuming control packets.
    struct ibv_sge sge;
    for (int i = 0; i < CtrlPktBuffPool::kNumPkt >> 1; i++) {
        uint64_t pkt_addr;
        if (ctrl_pkt_pool_.alloc_buff(&pkt_addr))
            throw std::runtime_error("Failed to allocate buffer for control packet");
        sge.addr = pkt_addr;
        sge.length = CtrlPktBuffPool::kPktSize;
        sge.lkey = ctrl_pkt_pool_.get_lkey();
        wr.wr_id = pkt_addr;
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        struct ibv_recv_wr *bad_wr;
        if (ibv_post_recv(ctrl_qp_, &wr, &bad_wr)) {
            throw std::runtime_error("ibv_post_recv failed");
        }
    }
}

RDMAContext::~RDMAContext()
{
    if (ctrl_mr_ != nullptr) {
        munmap(ctrl_mr_->addr, ctrl_mr_->length);
        ibv_dereg_mr(ctrl_mr_);
    }
    if (ctrl_cq_ != nullptr) {
        ibv_destroy_cq(ctrl_cq_);
    }
    if (ctrl_qp_ != nullptr) {
        ibv_destroy_qp(ctrl_qp_);
    }

    if (retr_mr_ != nullptr) {
        munmap(retr_mr_->addr, retr_mr_->length);
        ibv_dereg_mr(retr_mr_);
    }
    if (retr_cq_ != nullptr) {
        ibv_destroy_cq(retr_cq_);
    }
    if (retr_qp_ != nullptr) {
        ibv_destroy_qp(retr_qp_);
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

    for (auto qp : qp_vec_) {
        ibv_destroy_qp(qp);
    }

    if (pd_ != nullptr) {
        ibv_dealloc_pd(pd_);
    }
    if (cq_ != nullptr) {
        ibv_destroy_cq(cq_);
    }

    LOG(INFO) << "RDMAContext destroyed";
}

} // namespace uccl