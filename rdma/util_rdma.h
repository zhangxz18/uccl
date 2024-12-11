#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>

#include <deque>
#include <mutex>

#include <sys/mman.h>

#include <glog/logging.h>

#include <infiniband/verbs.h>

#include "transport_config.h"

#include "util.h"

namespace uccl {

class RDMAContext;
class RDMAFactory;
extern RDMAFactory rdma_ctl;

/**
 * @brief Exchange format between EndPoint and Engine.
 */
struct RDMAExchangeFormatLocal {
    union {
        // Endpoint --> Engine
        struct {
            int dev;
            union ibv_gid remote_gid;
            uint32_t remote_qpn;
            uint32_t remote_psn;
            ibv_mtu mtu;
            bool fifo;
            uint32_t fifo_key;
            uint64_t fifo_addr;
            // Memory Region
            void *addr;
            size_t len;
            int type;
        } ToEngine;
        // Engine --> Endpoint
        struct {
            uint32_t local_qpn;
            uint32_t local_psn;
            bool fifo;
            uint32_t fifo_key;
            uint64_t fifo_addr;;
        } ToEndPoint;
    };
};

/**
 * @brief Exchange format with remote peer.
 * @ref net_ib.cc:812:struct ncclIbConnectionMetadata
 */
struct RDMAExchangeFormatRemote {
    // QP information (Only one QP is used for now).
    uint32_t qpn;
    uint32_t psn;
    uint32_t fifo_key;
    uint64_t fifo_addr;
};

struct SendFifo {
    uint64_t addr;
    int size;
    uint32_t rkeys;
    uint32_t nreqs;
    uint32_t tag;
    uint64_t idx;
    char padding[28];
};

struct RemFifo {
    struct SendFifo elems[256][8];
    uint64_t fifo_tail;
    uint64_t addr;
    uint32_t flags;
};

struct RemoteRDMAContext {
    union ibv_gid remote_gid;
    uint32_t fifo_key;
    uint64_t fifo_addr;
};

/**
 * @brief  RDMA context for each UcclFlow, which is produced by RDMAFactory. It contains:
 *   - Multiple Unreliable Connection (UC) QPs and a shared CQ.
 *   - A high-priority QP for control messages and a dedicated CQ, PD, and MR.
 *   - A QP for retransmission and a dedicated CQ, PD, and MR.
 */
class RDMAContext {
    public:
        constexpr static int kTotalQP = kPortEntropy + 3;
        constexpr static int kFifoMRSize = sizeof(struct RemFifo);
        constexpr static int kCtrlMRSize = 128 * 1024;
        constexpr static int kRetrMRSize = 4096 * 1024;
        constexpr static int kCQSize = 4096;
        constexpr static int kMaxReq = 8;
        constexpr static int kMaxRecv = 32;

        // Memory region for data transfer.
        struct ibv_mr *data_mr_ = nullptr;
        // Protection domain for data transfer.
        struct ibv_pd *data_pd_ = nullptr;

        // QPs for data transfer based on Unreliable Connection (UC).
        std::vector<struct ibv_qp *> qp_vec_;
        // Local PSN for UC QPs.
        std::vector<uint32_t> local_psn_;
        // Remote PSN for UC QPs.
        std::vector<uint32_t> remote_psn_;
        // Shared CQ for all UC QPs.
        struct ibv_cq *cq_;
        // Protection domain for UC QPs.
        struct ibv_pd *pd_;

        // Fifo QP based on Reliable Connection (RC).
        struct ibv_qp *fifo_qp_;
        // Local PSN for Fifo.
        uint32_t fifo_local_psn_;
        // Remote PSN for Fifo.
        uint32_t fifo_remote_psn_;
        // Dedicated CQ for Fifo.
        struct ibv_cq *fifo_cq_;
        // Protection domain for Fifo.
        struct ibv_pd *fifo_pd_;
        // Memory region for Fifo.
        struct ibv_mr *fifo_mr_;
        
        // (high-priority) QP for control messages (e.g., ACK) based on Unreliable Datagram (UD).
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

        // QP for retransmission based on Reliable Connection (RC).
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

        // The device index that this context belongs to.
        int dev_;
        // The engine index that this context belongs to.
        int engine_idx_;

        struct ibv_context *context_;
        union ibv_gid local_gid_;
        ibv_mtu mtu_;
        uint8_t ib_port_num_;
        uint8_t sgid_index_;
        
        struct RemoteRDMAContext remote_ctx_;

        // When sync_cnt_ equals to kTotalQP, the flow is ready.
        uint32_t sync_cnt_;

        RDMAContext(int dev, int engine_idx, struct RDMAExchangeFormatLocal meta);

        ~RDMAContext(void);
        
        friend class RDMAFactory;
};

// Technically, it is equivalent to RDMADevice, but has more detailed device-specific information.
struct FactoryDevice {
    char ib_name[64];
    std::string local_ip_str;
    
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    
    uint8_t ib_port_num;
    uint8_t gid_idx;
    union ibv_gid gid;
};

/**
 * @brief Global RDMA factory, which is responsible for
 *  - Initializing the RDMA NIC.
 *  - Creating RDMA contexts for one UcclFlow
 */
class RDMAFactory {

    std::vector<struct FactoryDevice> devices_;

    // int gid_idx --> int dev
    std::unordered_map<int, int> gid_2_dev_map;

    // Track all RDMA contexts created by this factory.
    std::deque<RDMAContext *> context_q_;
    std::mutex context_q_lock_;
    
    public:
        static void init_dev(int gid_idx);
        static RDMAContext *CreateContext(int dev, int engine_idx, struct RDMAExchangeFormatLocal meta);
        static struct FactoryDevice *get_factory_dev(int dev);
        static void shutdown(void);
        
        std::string to_string(void) const;
};

static inline int modify_qp_rtr(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t remote_qpn, uint32_t remote_psn, bool rc)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_ctx->mtu_;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
    attr.ah_attr.grh.dgid = rdma_ctx->remote_ctx_.remote_gid;
    attr.ah_attr.grh.sgid_index = rdma_ctx->sgid_index_;
    attr.ah_attr.grh.hop_limit = 0xff;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;

    if (rc) {
        attr.min_rnr_timer = 12;
        attr.max_dest_rd_atomic = 1;
        attr_mask |= IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;
    }
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTR(mtu, port_num, sgidx_idx, dest_qp_num, rq_psn):" << (uint32_t)attr.path_mtu << "," << (uint32_t)attr.ah_attr.port_num << ","
        << (uint32_t)attr.ah_attr.grh.sgid_index << "," << attr.dest_qp_num << "," << attr.rq_psn;
        VLOG(1) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline int modify_qp_rts(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t local_psn, bool rc)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = local_psn;

    if (rc) {
        attr.timeout = 14;
        attr.retry_cnt = 7;
        attr.rnr_retry = 7;
        attr.max_rd_atomic = 1;
        attr_mask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    }

    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "QP#";
        oss << qp->qp_num;
        oss << " RTS(sq_psn):" << attr.sq_psn;
        VLOG(1) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline void util_rdma_create_qp(RDMAContext *rdma_ctx, struct ibv_context *context, struct ibv_qp **qp, enum ibv_qp_type qp_type,
    struct ibv_cq **cq, uint32_t cqsize, struct ibv_pd **pd, struct ibv_mr **mr, size_t mr_size, uint32_t max_send_wr, uint32_t max_recv_wr)
{
    // Create a dedicated CQ for control messages
    *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
    if (*cq == nullptr)
        throw std::runtime_error("ibv_create_cq failed");
    
    // Create PD for control messages
    *pd = ibv_alloc_pd(context);
    if (*pd == nullptr)
        throw std::runtime_error("ibv_alloc_pd failed");
    
    // Create memory region for control messages
    void *addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED)
        throw std::runtime_error("mmap failed");
    memset(addr, 0, mr_size);

    *mr = ibv_reg_mr(*pd, addr, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0));
    if (*mr == nullptr)
        throw std::runtime_error("ibv_reg_mr failed");

    // Create a dedicated QP for control messages
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.qp_context = rdma_ctx;
    qp_init_attr.send_cq = *cq;
    qp_init_attr.recv_cq = *cq;
    qp_init_attr.qp_type = qp_type;

    qp_init_attr.cap.max_send_wr = max_send_wr;
    qp_init_attr.cap.max_recv_wr = max_recv_wr;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    qp_init_attr.cap.max_inline_data = 0;

    *qp = ibv_create_qp(*pd, &qp_init_attr);
    if (*qp == nullptr)
        throw std::runtime_error("ibv_create_qp failed");

    // Modify QP state to INIT
    struct ibv_qp_attr qp_attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = rdma_ctx->ib_port_num_;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

    if (ibv_modify_qp(*qp, &qp_attr, attr_mask)) {
        throw std::runtime_error("ibv_modify_qp failed");
    }
}

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
 * @brief This helper function gets the Infiniband name from the GID index.
 * 
 * @param gid_idx 
 * @param ib_name 
 * @return int 
 */
static inline int util_rdma_get_ib_name_from_gididx(int gid_idx, char *ib_name)
{
    sprintf(ib_name, "%s%d", IB_DEVICE_NAME_PREFIX, gid_idx);
    return 0;
}

/**
 * @brief This helper function gets the IP address of the device from Infiniband name.
 * 
 * @param ib_name 
 * @param ip 
 * @return int 
 */
static inline int util_rdma_get_ip_from_ib_name(const char *ib_name, std::string *ip)
{
    char ethernet_name[64];
    if (util_rdma_ib2eth_name(ib_name, ethernet_name)) {
        return -1;
    }

    *ip = get_dev_ip(ethernet_name);

    return *ip == "" ? -1 : 0;
}


} // namespace uccl

#endif