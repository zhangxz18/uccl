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

static inline int convert_ib_name_to_ethernet_name(const char *ib_name, char *ethernet_name)
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
 * @brief Per engine context for RDMA
 * 
 */
class RDMAContext {
    public:
        // aka engine index
        int queue_id_;

        // UC QPs
        struct ibv_qp *qps_[kPortEntropy];
        // UC CQ
        struct ibv_cq *cq_;

        RDMAContext(int queue_id): queue_id_(queue_id) {}
    
    friend class RDMAFactory;
};

/**
 * @brief Global RDMA factory
 * 
 */
class RDMAFactory {

    char interface_name_[256];

    __be64 guid_;
    struct ibv_port_attr portAttr_;
    int portNum_;
    uint8_t link_;
    int speed_;
    struct ibv_context *context_;
    int pdRefs_;
    struct ibv_pd *pd_;

    int maxQp_;
    struct ncclIbMrCache mrCache_;
    struct ncclIbStats stats_;

    std::deque<RDMAContext *> context_q_;
    std::mutex context_q_lock_;

    public:
        static void init(const char *interface_name, uint64_t num_queues);
        static RDMAContext *CreateContext(int queue_id);
        static void shutdown(void);

        std::string to_string(void) const;
};

}

#endif