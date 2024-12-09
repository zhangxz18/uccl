#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>
#include <vector>

#include <deque>
#include <mutex>

#include <glog/logging.h>

#include <infiniband/verbs.h>

#include "transport_config.h"

namespace uccl {

class RDMAContext;
class RDMAFactory;
extern RDMAFactory rdma_ctl;

/**
 * @brief Exchange format between EndPoint and Engine.
 */
struct RDMAExchangeFormatLocal {
    union ibv_gid local_gid;
    union ibv_gid remote_gid;
    
    ibv_mtu mtu;
    int ib_port_num;
    int sgid_index;
    
    uint32_t local_qpn;
    uint32_t remote_qpn;
    
    uint32_t local_psn;
    uint32_t remote_psn;
};

/**
 * @brief Exchange format with remote peer.
 * 
 */
struct RDMAExchangeFormatRemote {
    uint32_t qpn;
    uint32_t psn;
};

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
 * @brief  RDMA context for each UcclFlow, which is produced by RDMAFactory. It contains:
 *   - Multiple Unreliable Connection (UC) QPs and a shared CQ.
 *   - A high-priority QP for control messages and a dedicated CQ, PD, and MR.
 *   - A QP for retransmission and a dedicated CQ, PD, and MR.
 */
class RDMAContext {
    constexpr static int kCtrlMRSize = 128 * 1024;
    constexpr static int kRetrMRSize = 4096 * 1024;
    constexpr static int kCQSize = 4096;
    constexpr static int kMaxReq = 8;
    constexpr static int kMaxRecv = 32;
    public:

        // Memory region for data transfer.
        struct ibv_mr * data_mr_ = nullptr;
        // Protection domain for data transfer.
        struct ibv_pd * data_pd_ = nullptr;

        // Unreliable Connection QPs.
        std::vector<struct ibv_qp *> qp_vec_;
        // Local PSN for UC QPs.
        std::vector<uint32_t> local_psn_;
        // Remote PSN for UC QPs.
        std::vector<uint32_t> remote_psn_;
        // Shared CQ for all UC QPs.
        struct ibv_cq *cq_;
        // Protection domain for UC QPs.
        struct ibv_pd *pd_;
        
        // High-priority QP for control messages(e.g., ACK).
        struct ibv_qp *ctrl_qp_;
        // Local PSN for control messages.
        uint32_t ctrl_local_psn_;
        // Remote PSN for control messages.
        uint32_t ctrl_remote_psn_;
        // Dedicated CQ for control messages.
        struct ibv_cq *ctrl_cq_;
        // Protection domain for control messages.
        struct ibv_pd *ctrl_pd_;
        // Memory region for control messages.
        struct ibv_mr *ctrl_mr_;
        // Memory address for control messages.
        void *ctrl_mr_addr_;

        // Retransmission QP.
        struct ibv_qp *retr_qp_;
        // Local PSN for retransmission.
        uint32_t retr_local_psn_;
        // Remote PSN for retransmission.
        uint32_t retr_remote_psn_;
        // Dedicated CQ for retransmission.
        struct ibv_cq *retr_cq_;
        // Prorection domain for retransmission.
        struct ibv_pd *retr_pd_;
        // Memory region for retransmission.
        struct ibv_mr *retr_mr_;
        // Memory address for retransmission.
        void *retr_mr_addr_;

        // The engine index that this context belongs to.
        int engine_idx_;

        struct ibv_context *context_;

        union ibv_gid local_gid_;
        union ibv_gid remote_gid_;

        ibv_mtu mtu_;
        uint8_t ib_port_num_;
        uint8_t sgid_index_;

        // When sync_cnt_ equals to kPortEntropy + 2, the flow is ready.
        uint32_t sync_cnt_;

        RDMAContext(int engine_idx, struct ibv_context *context, struct RDMAExchangeFormatLocal meta);

        ~RDMAContext(void);
        
        friend class RDMAFactory;
};

/**
 * @brief Global RDMA factory, which is responsible for
 *  - Initializing the RDMA NIC.
 *  - Creating RDMA contexts for one UcclFlow
 */
class RDMAFactory {

    char interface_name_[256];

    struct ibv_context *context_;
    __be64 guid_;
    struct ibv_port_attr port_attr_;
    uint8_t ib_port_num_;
    uint8_t sgid_idx_;
    uint8_t link_;
    int speed_;
    int pd_refs_;
    struct ibv_pd *pd_;

    int max_qp_;
    struct ncclIbMrCache mr_cache_;
    struct ncclIbStats stats_;

    // Track all RDMA contexts created by this factory.
    std::deque<RDMAContext *> context_q_;
    std::mutex context_q_lock_;
    
    public:
        static void init(const char *infiniband_name);
        static RDMAContext *CreateContext(int engine_idx, struct RDMAExchangeFormatLocal meta);
        static void shutdown(void);

        static struct ibv_context *get_ib_context(void);
        static uint8_t get_ib_port_num(void);
        static uint8_t get_sgid_index(void); 
        
        std::string to_string(void) const;
};

static inline int modify_qp_rtr(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t remote_qpn, uint32_t remote_psn)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_ctx->mtu_;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
    attr.ah_attr.grh.dgid = rdma_ctx->remote_gid_;
    attr.ah_attr.grh.sgid_index = rdma_ctx->sgid_index_;
    attr.ah_attr.grh.hop_limit = 0xff;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTR(mtu, port_num, sgidx_idx, dest_qp_num, rq_psn):" << (uint32_t)attr.path_mtu << "," << (uint32_t)attr.ah_attr.port_num << ","
        << (uint32_t)attr.ah_attr.grh.sgid_index << "," << attr.dest_qp_num << "," << attr.rq_psn;
        VLOG(1) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN);
}

static inline int modify_qp_rts(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t local_psn)
{
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = local_psn;

    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTS(sq_psn):" << attr.sq_psn;
        VLOG(1) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
}

} // namespace uccl

#endif