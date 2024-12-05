#include "util_rdma.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
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
             interface_name_, guid_, port_num_, link_, speed_, max_qp_);
    return std::string(buf);
}

void RDMAFactory::init(const char *infiniband_name, uint64_t num_queues) 
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
    rdma_ctl.port_num_ = 1;
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

    /// TODO: check RELAXED_ORDERING

    LOG(INFO) << rdma_ctl.to_string();

    ibv_free_device_list(device_list);

    return;

close_device:
    ibv_close_device(context);

free_devices:
    ibv_free_device_list(device_list);

    exit(EXIT_FAILURE);
}

RDMAContext *RDMAFactory::CreateContext(int engine_idx) 
{
    auto context = new RDMAContext(engine_idx, rdma_ctl.context_);
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    rdma_ctl.context_q_.push_back(context);
    return context;
}

void RDMAFactory::shutdown(void) 
{
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);

    for (auto context : rdma_ctl.context_q_) {
        
        for (int i = 0; i < kPortEntropy; i++) {
            if (context->qps_[i]) ibv_destroy_qp(context->qps_[i]);
            if (context->cq_) ibv_destroy_cq(context->cq_);
            if (context->pd_) ibv_dealloc_pd(context->pd_);
        }
        LOG(INFO) << "RDMAContext destroyed.";
        delete context;
    }

    rdma_ctl.context_q_.clear();
}

RDMAContext::RDMAContext(int engine_idx, struct ibv_context *context): 
    engine_idx_(engine_idx), context_(context) 
{
    // Create PD
    pd_ = ibv_alloc_pd(context_);
    if (pd_ == nullptr) {
        perror("ibv_alloc_pd");
        goto fail;
    }

    // Create CQ
    cq_ = ibv_create_cq(context_, kCqSize, nullptr /* TODO: void *cq_context */, nullptr, 0);
    if (cq_ == nullptr) {
        perror("ibv_create_cq");
        goto free_pd;
    }

    // Pack load GID info
    struct ibv_port_attr port_attr;
    if (util_rdma_get_gid_index(context_, 1, &port_attr, &gid_index_))
        goto free_cq;
    if (util_rdma_query_gid(context_, 1, gid_index_, &gid_))
        goto free_cq;

    // Create UC Qps
    for (int i = 0; i < kPortEntropy; i++) {
        struct ibv_qp_init_attr qpInitAttr;
        memset(&qpInitAttr, 0, sizeof(qpInitAttr));
        
        qpInitAttr.qp_context = this;
        qpInitAttr.send_cq = cq_;
        qpInitAttr.recv_cq = cq_;
        qpInitAttr.qp_type = IBV_QPT_UC;

        qpInitAttr.cap.max_send_wr = kMaxReq * kMaxRecv;
        qpInitAttr.cap.max_recv_wr = kMaxReq * kMaxRecv;
        qpInitAttr.cap.max_send_sge = 1;
        qpInitAttr.cap.max_recv_sge = 1;
        qpInitAttr.cap.max_inline_data = 0;

        qps_[i] = ibv_create_qp(pd_, &qpInitAttr);
        if (qps_[i] == nullptr) {
            perror("ibv_create_qp");
            goto free_qps;
        }

        // Modify QP state to INIT
        struct ibv_qp_attr qpAttr;
        memset(&qpAttr, 0, sizeof(qpAttr));
        qpAttr.qp_state = IBV_QPS_INIT;
        qpAttr.pkey_index = 0;
        qpAttr.port_num = 1;
        qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        if (ibv_modify_qp(qps_[i], &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            perror("ibv_modify_qp");
            goto free_qps;
        }

        // Modify QP state to RTR
        // There is no need for UC to set max_dest_rd_atomic and min_rnr_timer
        memset(&qpAttr.ah_attr, 0, sizeof(qpAttr.ah_attr));
        qpAttr.qp_state = IBV_QPS_RTR;
        qpAttr.path_mtu = RDMA_MTU == 1024 ? IBV_MTU_1024 : IBV_MTU_4096;
        qpAttr.dest_qp_num = 0; // remote
        qpAttr.rq_psn = 0;
        
        qpAttr.ah_attr.is_global = 1;
        qpAttr.ah_attr.grh.dgid.global.subnet_prefix = 0; // remote
        qpAttr.ah_attr.grh.dgid.global.interface_id = 0; // remote
        qpAttr.ah_attr.grh.flow_label = 0; // ???
        qpAttr.ah_attr.grh.sgid_index = gid_index_;
        qpAttr.ah_attr.grh.hop_limit = 255;
        qpAttr.ah_attr.grh.traffic_class = 0; // ???

        qpAttr.ah_attr.sl = 0; // ???
        qpAttr.ah_attr.src_path_bits = 0;
        qpAttr.ah_attr.port_num = 1;

        if (ibv_modify_qp(qps_[i], &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)) {
            perror("ibv_modify_qp to RTR");
            goto free_qps;
        }

        // Modify QP state to RTS
        // There is no need for UC to set timeout, retry_cnt, and rnr_retry and max_rd_atomic
        memset(&qpAttr, 0, sizeof(qpAttr));
        qpAttr.qp_state = IBV_QPS_RTS;
        qpAttr.sq_psn = 0;
        if (ibv_modify_qp(qps_[i], &qpAttr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
            perror("ibv_modify_qp to RTS");
            goto free_qps;
        }
    }

    std::cout << "RDMAContext created." << std::endl;

    return;

free_qps:
    for (int i = 0; i < kPortEntropy; i++) {
        if (qps_[i]) ibv_destroy_qp(qps_[i]);
    }
free_cq:
    ibv_destroy_cq(cq_);
free_pd:
    ibv_dealloc_pd(pd_);
fail:
    exit(EXIT_FAILURE);
}

}