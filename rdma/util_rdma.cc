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

std::string RDMAFactory::to_string(void) const
{
    char buf[512];
    snprintf(buf, sizeof(buf), 
        "RDMAFactory: infiniband_name=%s, guid=%llx, portNum=%d, link=%d, speed=%d, maxQp=%d",
             interface_name_, guid_, ib_port_num_, link_, speed_, max_qp_);
    return std::string(buf);
}

/**
 * @brief This function initializes the RDMA NIC, including:
 *  - Open the RDMA device
 *  - Query the device attributes
 *  - Create memory region cache
 *  - and more.
 * 
 * @param infiniband_name 
 */
void RDMAFactory::init(const char *infiniband_name) 
{
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    int i, nb_devices;
    
    strcpy(rdma_ctl.interface_name_, infiniband_name);

    // Get the list of RDMA devices
    device_list = ibv_get_device_list(&nb_devices);
    if (device_list == nullptr || nb_devices == 0) {
        perror("ibv_get_device_list");
        exit(EXIT_FAILURE);
    }

    // Currently, we only use one device that is specified by the interface name
    for (i = 0; i < nb_devices; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), infiniband_name) == 0) {
            break;
        }
    }
    if (i == nb_devices) {
        fprintf(stderr, "No device found for %s\n", infiniband_name);
        goto free_devices;
    }

    // Open device
    memset(&dev_attr, 0, sizeof(dev_attr));
    if ((context = ibv_open_device(device_list[i])) == nullptr) {
        perror("ibv_open_device");
        goto free_devices;
    }

    if (ibv_query_device(context, &dev_attr)) {
        perror("ibv_query_device");
        goto close_device;
    }

    // Currently, we only use one port
    if (dev_attr.phys_port_cnt != 1) {
        fprintf(stderr, "Only one port is supported\n");
        goto close_device;
    }

    // Port number starts from 1
    if (ibv_query_port(context, 1, &port_attr)) {
        perror("ibv_query_port");
        goto close_device;
    }

    // Currently, we only support RoCEv2
    if (port_attr.state != IBV_PORT_ACTIVE ||
        (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)) {
        fprintf(stderr, "Port is not active or not Ethernet\n");
        goto close_device;
    }

    rdma_ctl.guid_ = dev_attr.sys_image_guid;
    rdma_ctl.port_attr_ = port_attr;
    rdma_ctl.ib_port_num_ = IB_PORT_NUM;
    rdma_ctl.sgid_idx_ = SGID_INDEX;
    rdma_ctl.link_ = port_attr.link_layer;
    rdma_ctl.speed_ = port_attr.active_speed;
    rdma_ctl.context_ = context;
    rdma_ctl.pd_refs_ = 0;
    rdma_ctl.pd_ = nullptr;
    rdma_ctl.max_qp_ = dev_attr.max_qp;
    rdma_ctl.mr_cache_.capacity = 0;
    rdma_ctl.mr_cache_.population = 0;
    rdma_ctl.mr_cache_.slots = 0;

    __atomic_store_n(&rdma_ctl.stats_.fatalErrorCount, 0, __ATOMIC_RELAXED);

    /// TODO: check RELAXED_ORDERING.


    /// TODO: create memory region cache for fast MR registration.

    LOG(INFO) << rdma_ctl.to_string();

    ibv_free_device_list(device_list);

    return;

close_device:
    ibv_close_device(context);

free_devices:
    ibv_free_device_list(device_list);

    throw std::runtime_error("Failed to initialize RDMAFactory");
}

/**
 * @brief This function frees all resources allocated by the RDMAFactory, including:
 *  - All RDMA contexts
 *  - Resources allocated in init().
 */
void RDMAFactory::shutdown(void) 
{
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    for (auto ctx : rdma_ctl.context_q_) {
        delete ctx;
    }
    rdma_ctl.context_q_.clear();
}

struct ibv_context *RDMAFactory::get_ib_context(void)
{
    return rdma_ctl.context_;
}

uint8_t RDMAFactory::get_ib_port_num(void)
{
    return rdma_ctl.ib_port_num_;
}

uint8_t RDMAFactory::get_sgid_index(void)
{
    return rdma_ctl.sgid_idx_;
}

RDMAContext *RDMAFactory::CreateContext(int engine_idx, struct RDMAExchangeFormatLocal meta)
{
    RDMAContext *ctx = new RDMAContext(engine_idx, rdma_ctl.context_, meta);
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    rdma_ctl.context_q_.push_back(ctx);
    return ctx;
}

RDMAContext::RDMAContext(int engine_idx, struct ibv_context *context, struct RDMAExchangeFormatLocal meta):
    engine_idx_(engine_idx), context_(context), sync_cnt_(0)
{
    local_gid_ = meta.local_gid;
    mtu_ = meta.mtu;
    ib_port_num_ = meta.ib_port_num;
    sgid_index_ = meta.sgid_index;

    qp_vec_.resize(kPortEntropy);
    local_psn_.resize(kPortEntropy);
    remote_psn_.resize(kPortEntropy);

    // Create a dedicated CQ for UC QPs
    cq_ = ibv_create_cq(context_, kCQSize, nullptr, nullptr, 0);
    if (cq_ == nullptr)
        throw std::runtime_error("ibv_create_cq failed");

    // Crate PD for UC QPs
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");

    // Create up to kPortEntropy UC QPs
    for (int i = 0; i < kPortEntropy; i++) {
        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));

        qp_init_attr.qp_context = this;
        qp_init_attr.send_cq = cq_;
        qp_init_attr.recv_cq = cq_;
        qp_init_attr.qp_type = IBV_QPT_UC;

        qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv;
        qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = 0;

        struct ibv_qp *qp = ibv_create_qp(pd_, &qp_init_attr);
        if (qp == nullptr)
            throw std::runtime_error("ibv_create_qp failed");
        
        local_psn_[i] = meta.local_psn + i;
        
        // Modify QP state to INIT
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

    // Create a dedicated CQ for control messages
    ctrl_cq_ = ibv_create_cq(context_, kCQSize, nullptr, nullptr, 0);
    if (ctrl_cq_ == nullptr)
        throw std::runtime_error("ibv_create_cq failed");
    
    // Create PD for control messages
    ctrl_pd_ = ibv_alloc_pd(context_);
    if (ctrl_pd_ == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");
    
    // Create memory region for control messages
    ctrl_mr_addr_ = mmap(nullptr, kCtrlMRSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ctrl_mr_addr_ == MAP_FAILED)
        throw std::runtime_error("mmap failed");

    ctrl_mr_ = ibv_reg_mr(ctrl_pd_, ctrl_mr_addr_, kCtrlMRSize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (ctrl_mr_ == nullptr)
        throw std::runtime_error("ibv_reg_mr failed");

    // Create a dedicated QP for control messages
    struct ibv_qp_init_attr ctrl_qp_init_attr;
    memset(&ctrl_qp_init_attr, 0, sizeof(ctrl_qp_init_attr));

    ctrl_qp_init_attr.qp_context = this;
    ctrl_qp_init_attr.send_cq = ctrl_cq_;
    ctrl_qp_init_attr.recv_cq = ctrl_cq_;
    ctrl_qp_init_attr.qp_type = IBV_QPT_UC;

    ctrl_qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv;
    ctrl_qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv;
    ctrl_qp_init_attr.cap.max_send_sge = 1;
    ctrl_qp_init_attr.cap.max_recv_sge = 1;
    ctrl_qp_init_attr.cap.max_inline_data = 0;

    ctrl_qp_ = ibv_create_qp(ctrl_pd_, &ctrl_qp_init_attr);
    if (ctrl_qp_ == nullptr)
        throw std::runtime_error("ibv_create_qp failed");
    ctrl_local_psn_ = meta.local_psn + kPortEntropy;

    // Modify QP state to INIT
    struct ibv_qp_attr ctrl_qp_attr;
    memset(&ctrl_qp_attr, 0, sizeof(ctrl_qp_attr));
    ctrl_qp_attr.qp_state = IBV_QPS_INIT;
    ctrl_qp_attr.pkey_index = 0;
    ctrl_qp_attr.port_num = ib_port_num_;
    ctrl_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(ctrl_qp_, &ctrl_qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        throw std::runtime_error("ibv_modify_qp failed");
    }

    // Create a dedicated CQ for retransmission
    retr_cq_ = ibv_create_cq(context_, kCQSize, nullptr, nullptr, 0);
    if (retr_cq_ == nullptr)
        throw std::runtime_error("ibv_create_cq failed");

    // Create PD for retransmission
    retr_pd_ = ibv_alloc_pd(context_);
    if (retr_pd_ == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");

    // Create memory region for retransmission
    retr_mr_addr_ = mmap(nullptr, kRetrMRSize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (retr_mr_addr_ == MAP_FAILED)
        throw std::runtime_error("mmap failed");

    retr_mr_ = ibv_reg_mr(retr_pd_, retr_mr_addr_, kRetrMRSize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (retr_mr_ == nullptr)
        throw std::runtime_error("ibv_reg_mr failed");

    // Create a dedicated QP for retransmission
    struct ibv_qp_init_attr retr_qp_init_attr;
    memset(&retr_qp_init_attr, 0, sizeof(retr_qp_init_attr));

    retr_qp_init_attr.qp_context = this;
    retr_qp_init_attr.send_cq = retr_cq_;
    retr_qp_init_attr.recv_cq = retr_cq_;
    retr_qp_init_attr.qp_type = IBV_QPT_UC;
    
    retr_qp_init_attr.cap.max_send_wr = kMaxReq * kMaxRecv;
    retr_qp_init_attr.cap.max_recv_wr = kMaxReq * kMaxRecv;
    retr_qp_init_attr.cap.max_send_sge = 1;
    retr_qp_init_attr.cap.max_recv_sge = 1;
    retr_qp_init_attr.cap.max_inline_data = 0;

    retr_qp_ = ibv_create_qp(retr_pd_, &retr_qp_init_attr);
    if (retr_qp_ == nullptr)
        throw std::runtime_error("ibv_create_qp failed");
    retr_local_psn_ = meta.local_psn + kPortEntropy + 1;

    // Modify QP state to INIT
    struct ibv_qp_attr retr_qp_attr;
    memset(&retr_qp_attr, 0, sizeof(retr_qp_attr));
    retr_qp_attr.qp_state = IBV_QPS_INIT;
    retr_qp_attr.pkey_index = 0;
    retr_qp_attr.port_num = ib_port_num_;
    retr_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(retr_qp_, &retr_qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        throw std::runtime_error("ibv_modify_qp failed");
    }
}

RDMAContext::~RDMAContext()
{
    if (ctrl_mr_ != nullptr) {
        ibv_dereg_mr(ctrl_mr_);
    }
    if (ctrl_mr_addr_ != nullptr) {
        munmap(ctrl_mr_addr_, kCtrlMRSize);
    }
    if (ctrl_pd_ != nullptr) {
        ibv_dealloc_pd(ctrl_pd_);
    }
    if (ctrl_cq_ != nullptr) {
        ibv_destroy_cq(ctrl_cq_);
    }
    if (ctrl_qp_ != nullptr) {
        ibv_destroy_qp(ctrl_qp_);
    }

    if (retr_mr_ != nullptr) {
        ibv_dereg_mr(retr_mr_);
    }
    if (retr_mr_addr_ != nullptr) {
        munmap(retr_mr_addr_, kCtrlMRSize);
    }
    if (retr_pd_ != nullptr) {
        ibv_dealloc_pd(retr_pd_);
    }
    if (retr_cq_ != nullptr) {
        ibv_destroy_cq(retr_cq_);
    }
    if (retr_qp_ != nullptr) {
        ibv_destroy_qp(retr_qp_);
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

}