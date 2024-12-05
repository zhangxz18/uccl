#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>

#include <deque>
#include <mutex>

#include <infiniband/verbs.h>

#include "transport_config.h"

class RDMAContext;
class RDMAFactory;
extern RDMAFactory rdma_ctl;

namespace uccl {

struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};

struct ncclIbStats {
  int fatalErrorCount;
};

/**
 * @brief This helper function converts an Infiniband name (e.g., mlx5_0) to an Ethernet name (e.g., eth0)
 * @return int -1 on error, 0 on success
 */
static inline int util_rdma_ib2eth_name(const char *ib_name, char *ethernet_name)
{
    char command[512];
    snprintf(command, sizeof(command), 
        "ls -l /sys/class/infiniband/%s/device/net | sed -n '2p' | sed 's/.* //'", ib_name);
    FILE *fp = popen(command, "r");
    if (fp == nullptr) {
        perror("popen");
        return -1;
    }
    if (fgets(ethernet_name, 64, fp) == NULL) {
        pclose(fp);
        return -1;
    }
    pclose(fp);
    // Remove newline character if present
    ethernet_name[strcspn(ethernet_name, "\n")] = '\0';
    return 0;
}

/**
 * @brief This helper function gets the GUID of an RoCE device
 * @return int -1 on error, 0 on success
 */
static inline int util_rdma_get_gid_index(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr, int *gid_index)
{
    *gid_index = GID_INDEX;
    return 0;
}

static inline int util_rdma_query_gid(struct ibv_context *context, uint8_t port_num, int gid_index ,union ibv_gid *gid)
{
    return ibv_query_gid(context, port_num, gid_index, gid);
}

/**
 * @brief Per engine context for RDMA
 * 
 */
class RDMAContext {
    public:

        static const int kCqSize = 4096;
        static const int kMaxReq = 32;
        static const int kMaxRecv = 8;

        int engine_idx_;

        struct ibv_context *context_ = nullptr;

        int gid_index_ = -1;

        union ibv_gid gid_;

        // Protection domain (PD)
        struct ibv_pd *pd_ = nullptr;

        // Completion queue (CQ)
        struct ibv_cq *cq_ = nullptr;

        // Unreliable Connection (UC) Queue Pairs (QPs)
        struct ibv_qp *qps_[kPortEntropy] = {nullptr};

        RDMAContext(int engine_idx, struct ibv_context *context);
    
    friend class RDMAFactory;
};

/**
 * @brief Global RDMA factory
 * 
 */
class RDMAFactory {

    char interface_name_[256];

    struct ibv_context *context_;
    __be64 guid_;
    struct ibv_port_attr port_attr_;
    int port_num_;
    uint8_t link_;
    int speed_;
    int pd_refs_;
    struct ibv_pd *pd_;

    int max_qp_;
    struct ncclIbMrCache mr_cache_;
    struct ncclIbStats stats_;

    std::deque<RDMAContext *> context_q_;
    std::mutex context_q_lock_;

    public:
        static void init(const char *infiniband_name, uint64_t num_queues);
        static RDMAContext *CreateContext(int queue_id);
        static void shutdown(void);

        std::string to_string(void) const;
};

}

#endif