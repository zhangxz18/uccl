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

class IMMData{
    public:
        // High                              Low
        //  |  HINT  |  CSN  |  RID  |  MID  |
        //     1bit    20bit    8bit    3bit
        constexpr static int kHint = 31;
        constexpr static int kCSN = 11;
        constexpr static int kRID = 3;
        constexpr static int kMID = 0;

        IMMData(uint32_t imm_data):imm_data_(imm_data) {}
        ~IMMData() = default;

        inline uint32_t GetHint(void) {
            return (imm_data_ >> kHint) & 0x1;
        }

        inline uint32_t GetCSN(void) {
            return (imm_data_ >> kCSN) & 0xFFFFF;
        }

        inline uint32_t GetRID(void) {
            return (imm_data_ >> kRID) & 0xFF;
        }

        inline uint32_t GetMID(void) {
            return (imm_data_ >> kMID) & 0x7;
        }

        inline void SetHint(uint32_t hint) {
            imm_data_ |= (hint & 0x1) << kHint;
        }

        inline void SetCSN(uint32_t psn) {
            imm_data_ |= (psn & 0xFFFFF) << kCSN;
        }

        inline void SetRID(uint32_t mid) {
            imm_data_ |= (mid & 0xFF) << kRID;
        }

        inline void SetMID(uint32_t rid) {
            imm_data_ |= (rid & 0x7) << kMID;
        }

        inline uint32_t GetImmData(void) {
            return imm_data_;
        }

    private:
        uint32_t imm_data_;    
};

/// @brief Exchange format between EndPoint and Engine.
struct RDMAExchangeFormatLocal {
    union {
        // Endpoint --> Engine
        struct {
            bool is_send;
            int dev;
            union ibv_gid remote_gid;
            uint32_t remote_qpn;
            uint32_t remote_psn;
            ibv_mtu mtu;
            bool fifo;
            uint32_t fifo_key; // Only valid when fifo is true
            uint64_t fifo_addr; // Only valid when fifo is true
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
            uint32_t fifo_key; // Only valid when fifo is true
            uint64_t fifo_addr; // Only valid when fifo is true
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

struct FifoItem {
    uint64_t addr;
    uint32_t size;
    uint32_t rkey;
    uint32_t nmsgs;
    uint64_t idx;
    uint32_t msg_id;
    char padding[28];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

struct RemFifo {
    // FIFO elements prepared for sending to remote peer.
    struct FifoItem elems[kMaxReq][kMaxRecv];
    uint32_t received_bytes[kMaxReq][kMaxRecv];
    // Tail pointer of the FIFO.
    uint64_t fifo_tail;
};

struct RemoteRDMAContext {
    union ibv_gid remote_gid;
    uint32_t fifo_key;
    uint64_t fifo_addr;
};

/// @ref ncclIbRequest
struct FlowRequest {
    enum type {
        UNUSED = 0,
        SEND,
        RECV,
        FLUSH,
    };

    enum type type;
    int nmsgs;
    PollCtx *poll_ctx;

    union {
        struct {
            int size;
            void *data;
            uint32_t lkey;
            int sent_offset;
        } send;

        struct {
            uint32_t *received_bytes;
            struct FifoItem *elems;
            uint32_t fin_msg;
        } recv;
    };
};

struct RemSizesFifo {
    int elems[kMaxReq][kMaxRecv];
    uint64_t fifo_tail;
    uint64_t addr;
    uint32_t rkey;
    uint32_t flags;
    struct ibv_mr *mr;
    struct ibv_sge sge;
};

/// @ref ncclIbNetCommBase
struct alignas(32) NetCommBase {
    // Is this a send or receive flow?
    bool is_send;
    // Tack outstanding requests.
    struct FlowRequest reqs[kMaxReq];
    // Remote RDMA context.
    struct RemoteRDMAContext remote_ctx;
    // Pointing to rdma_ctx_->fifo_mr_->addr.
    struct RemFifo *fifo;
};

/// @ref ncclIbSendComm
struct SendComm {
    struct NetCommBase base;
    
    struct ibv_send_wr wrs[kMaxRecv + 1];
    struct ibv_sge sges[kMaxRecv];

    // Track outstanding requests.
    struct FlowRequest *fifo_reqs[kMaxReq][kMaxRecv];
    uint64_t fifo_head;
    
    struct RemSizesFifo rem_sizes_fifo;

};

/// @ref ncclIbRecvComm
struct RecvComm {
    struct NetCommBase base;
};

/**
 * @brief  RDMA context for each UcclFlow, which is produced by RDMAFactory. It contains:
 *   - Multiple Unreliable Connection (UC) QPs and a shared CQ.
 *   - A high-priority QP for control messages and a dedicated CQ, PD, and MR.
 *   - A QP for retransmission and a dedicated CQ, PD, and MR.
 *   - A Reliable Connection (RC) QP for FIFO and a dedicated CQ, PD, and MR.
 */
class RDMAContext {
    public:
        constexpr static int kTotalQP = kPortEntropy + 3;
        constexpr static int kFifoMRSize = sizeof(struct RemFifo);
        constexpr static int kCtrlMRSize = 128 * 1024;
        constexpr static int kRetrMRSize = 4096 * 1024;
        constexpr static int kCQSize = 4096;

        // Protection domain for all RDMA resources.
        struct ibv_pd *pd_ = nullptr;
        
        // Memory region for data transfer.
        struct ibv_mr *data_mr_ = nullptr;

        // QPs for data transfer based on Unreliable Connection (UC).
        std::vector<struct ibv_qp *> qp_vec_;
        // QPN to index mapping.
        std::unordered_map<uint32_t, int> qpn2idx_;
        uint32_t fill_cnt_[kPortEntropy] = {0};
        // Local PSN for UC QPs.
        std::vector<uint32_t> local_psn_;
        // Remote PSN for UC QPs.
        std::vector<uint32_t> remote_psn_;
        // Shared CQ for all UC QPs.
        struct ibv_cq *cq_;

        // Fifo QP based on Reliable Connection (RC).
        struct ibv_qp *fifo_qp_;
        // Local PSN for Fifo.
        uint32_t fifo_local_psn_;
        // Remote PSN for Fifo.
        uint32_t fifo_remote_psn_;
        // Dedicated CQ for Fifo.
        struct ibv_cq *fifo_cq_;
        // Memory region for Fifo.
        struct ibv_mr *fifo_mr_;

        // Whether its FIFO CQ is under polling.
        bool fifo_cq_polling_;
        
        // (high-priority) QP for control messages (e.g., ACK) based on Unreliable Datagram (UD).
        struct ibv_qp *ctrl_qp_;
        // Local PSN for control messages.
        uint32_t ctrl_local_psn_;
        // Remote PSN for control messages.
        uint32_t ctrl_remote_psn_;
        // Dedicated CQ for control messages.
        struct ibv_cq *ctrl_cq_;
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
        // Memory region for retransmission.
        struct ibv_mr *retr_mr_;

        // The device index that this context belongs to.
        int dev_;

        // RDMA device context per device.
        struct ibv_context *context_;
        // Local GID of this device.
        union ibv_gid local_gid_;
        // MTU of this device.
        ibv_mtu mtu_;
        // IB port number of this device.
        uint8_t ib_port_num_;
        // GID index of this device.
        uint8_t sgid_index_;
        
        /**
         * @brief Communication abstraction for sending and receiving.
         * Since data flow is unidirectional in NCCL, we use two different structures.
         */
        union {
            // For connection setup by connect().
            struct SendComm send_comm_;
            // For connection setup by accept().
            struct RecvComm recv_comm_;
        };

        // Whether this context is for sending or receiving.
        bool is_send_;

        /**
         * @brief Figure out the request id.
         * @param req 
         * @param comm_base 
         * @return int 
         */
        inline int get_request_id(struct FlowRequest *req, struct NetCommBase *comm_base) {
            return req - comm_base->reqs;
        }

        /**
         * @brief Get the request by id.
         * @param id 
         * @param comm_base 
         * @return struct FlowRequest* 
         */
        inline struct FlowRequest *get_request_by_id(int id, struct NetCommBase *comm_base) {
            return &comm_base->reqs[id];
        }

        inline void free_request(struct FlowRequest *req) {
            req->type = FlowRequest::UNUSED;
        }

        /**
         * @brief Get an unused request, if no request is available, return nullptr.
         * @param comm_base 
         * @return struct FlowRequest* 
         */
        inline struct FlowRequest *get_request(struct NetCommBase *comm_base) {
            for (int i = 0; i < kMaxReq; i++) {
                auto *req = &comm_base->reqs[i];
                if (req->type == FlowRequest::UNUSED) {
                    req->nmsgs = 0;
                    req->poll_ctx = nullptr;
                    return req;
                }
            }
            return nullptr;
        }

        inline struct ibv_qp *select_qp(void) {
            return qp_vec_[std::rand() % qp_vec_.size()];
        }

        // When sync_cnt_ equals to kTotalQP, the flow is ready.
        uint32_t sync_cnt_;

        RDMAContext(int dev, struct RDMAExchangeFormatLocal meta);

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
        static RDMAContext *CreateContext(int dev, struct RDMAExchangeFormatLocal meta);
        static struct FactoryDevice *get_factory_dev(int dev);
        static void shutdown(void);
        
        std::string to_string(void) const;
};

static inline int modify_qp_rtr(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t remote_qpn, uint32_t remote_psn, bool rc)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    auto comm_base = rdma_ctx->is_send_ ? &rdma_ctx->send_comm_.base : &rdma_ctx->recv_comm_.base;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_ctx->mtu_;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
    attr.ah_attr.grh.dgid = comm_base->remote_ctx.remote_gid;
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
    struct ibv_cq **cq, uint32_t cqsize, struct ibv_pd *pd, struct ibv_mr **mr, size_t mr_size, uint32_t max_send_wr, uint32_t max_recv_wr)
{
    // Creating CQ
    *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
    if (*cq == nullptr)
        throw std::runtime_error("ibv_create_cq failed");
    
    // Creating MR
    void *addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED)
        throw std::runtime_error("mmap failed");
    memset(addr, 0, mr_size);

    *mr = ibv_reg_mr(pd, addr, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0));
    if (*mr == nullptr)
        throw std::runtime_error("ibv_reg_mr failed");

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

    // Creating QP
    *qp = ibv_create_qp(pd, &qp_init_attr);
    if (*qp == nullptr)
        throw std::runtime_error("ibv_create_qp failed");

    // Modifying QP state to INIT
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

static inline int util_rdma_get_mtu_from_ibv_mtu(ibv_mtu mtu)
{
    switch (mtu) {
        case IBV_MTU_256: return 256;
        case IBV_MTU_512: return 512;
        case IBV_MTU_1024: return 1024;
        case IBV_MTU_2048: return 2048;
        case IBV_MTU_4096: return 4096;
        default: return 0;
    }
} 

}// namespace uccl

#endif