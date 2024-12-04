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
        "RDMAFactory: interface_name=%s, guid=%llx, portNum=%d, link=%d, speed=%d, maxQp=%d",
             interface_name_, guid_, portNum_, link_, speed_, maxQp_);
    return std::string(buf);
}

void RDMAFactory::init(const char *interface_name, uint64_t num_queues) 
{
    struct ibv_device **device_list;
    struct ibv_context *context;
    struct ibv_device_attr devAttr;
    struct ibv_port_attr portAttr;
    int i, nb_devices;
    
    strcpy(rdma_ctl.interface_name_, interface_name);

    device_list = ibv_get_device_list(&nb_devices);
    if (device_list == nullptr || nb_devices == 0) {
        perror("ibv_get_device_list");
        exit(EXIT_FAILURE);
    }

    // Currently, we only use the device specified by the interface name
    for (i = 0; i < nb_devices; i++) {
        if (strcmp(ibv_get_device_name(device_list[i]), interface_name) == 0) {
            break;
        }
    }

    if (i == nb_devices) {
        fprintf(stderr, "No device found for %s\n", interface_name);
        goto free_devices;
    }

    memset(&devAttr, 0, sizeof(devAttr));

    if ((context = ibv_open_device(device_list[i])) == nullptr) {
        perror("ibv_open_device");
        goto free_devices;
    }

    if (ibv_query_device(context, &devAttr)) {
        perror("ibv_query_device");
        goto close_device;
    }

    // Currently, we only use one port
    if (devAttr.phys_port_cnt != 1) {
        fprintf(stderr, "Only one port is supported\n");
        goto close_device;
    }

    // Port number starts from 1
    if (ibv_query_port(context, 1, &portAttr)) {
        perror("ibv_query_port");
        goto close_device;
    }

    if (portAttr.state != IBV_PORT_ACTIVE ||
        (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)) {
        fprintf(stderr, "Port is not active or not Ethernet\n");
        goto close_device;
    }

    rdma_ctl.guid_ = devAttr.sys_image_guid;
    rdma_ctl.portAttr_ = portAttr;
    rdma_ctl.portNum_ = 1;
    rdma_ctl.link_ = portAttr.link_layer;
    rdma_ctl.speed_ = portAttr.active_speed;
    rdma_ctl.context_ = context;
    rdma_ctl.pdRefs_ = 0;
    rdma_ctl.pd_ = nullptr;
    rdma_ctl.maxQp_ = devAttr.max_qp;
    rdma_ctl.mrCache_.capacity = 0;
    rdma_ctl.mrCache_.population = 0;
    rdma_ctl.mrCache_.slots = 0;

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

RDMAContext *RDMAFactory::CreateContext(int queue_id) 
{
    auto context = new RDMAContext(queue_id);
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);
    rdma_ctl.context_q_.push_back(context);
    return context;
}

void RDMAFactory::shutdown(void) 
{
    std::lock_guard<std::mutex> lock(rdma_ctl.context_q_lock_);

    for (auto context : rdma_ctl.context_q_) {
        delete context;
    }

    rdma_ctl.context_q_.clear();
}

}