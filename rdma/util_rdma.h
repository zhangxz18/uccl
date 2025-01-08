#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <set>

#include <sys/mman.h>
#include <infiniband/verbs.h>

#include <glog/logging.h>

#include "transport_config.h"
#include "transport_cc.h"

#include "util.h"
#include "util_list.h"

namespace uccl {

class RDMAContext;
class RDMAFactory;
extern RDMAFactory rdma_ctl;

// LRH (Local Routing Header) + GRH (Global Routing Header) + BTH (Base Transport Header)
static constexpr uint32_t IB_HDR_OVERHEAD = (8 + 40 + 12);
// Ethernet + IPv4 + UDP + BTH
static constexpr uint32_t ROCE_IPV4_HDR_OVERHEAD = (14 + 20 + 8 + 12);
// Ethernet + IPv6 + UDP + BTH
static constexpr uint32_t ROCE_IPV6_HDR_OVERHEAD = (14 + 40 + 8 + 12);

// For quick computation at MTU 4096
static constexpr uint32_t MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD = ((kChunkSize + 4096) / 4096) * ROCE_IPV4_HDR_OVERHEAD;
static constexpr uint32_t MAX_CHUNK_ROCE_IPV6_4096_HDR_OVERHEAD = ((kChunkSize + 4096) / 4096) * ROCE_IPV6_HDR_OVERHEAD;
static constexpr uint32_t MAX_CHUNK_IB_4096_HDR_OVERHEAD = ((kChunkSize + 4096) / 4096) * IB_HDR_OVERHEAD;

/**
 * @brief A buffer pool with the following properties:
 * - Constructed with a memory region provided by the caller or mmap by itself.
 * - Not thread-safe, single producer, single consumer.
 * - Fixed size elements.
 * - Size must be power of 2.
 */
class BuffPool {
    public:

        BuffPool(uint32_t num_elements, size_t element_size, struct ibv_mr *mr = nullptr, void (*init_cb)(uint64_t buff) = nullptr)
            : num_elements_(num_elements), element_size_(element_size), mr_(mr) {
            if (mr_) {
                base_addr_ = mr->addr;
            } else {
                base_addr_ = mmap(nullptr, num_elements_ * element_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (base_addr_ == MAP_FAILED)
                    throw std::runtime_error("Failed to allocate memory for BuffPool.");
            }
            buffer_pool_ = new uint64_t[num_elements_];
            head_ = tail_ = 0;
            for (uint32_t i = 0; i < num_elements_ - 1; i++) {
                if (init_cb) init_cb((uint64_t)base_addr_ + i * element_size_);
                free_buff((uint64_t)base_addr_ + i * element_size_);
            }
        }

        ~BuffPool() {
            if (!mr_) {
                munmap(base_addr_, num_elements_ * element_size_);
            }
            delete[] buffer_pool_;
        }

        bool full(void) {
            return ((tail_ + 1) & (num_elements_ - 1)) == head_;
        }

        bool empty(void) {
            return head_ == tail_;
        }

        uint32_t get_lkey(void) {
            if (!mr_) return 0;
            return mr_->lkey;
        }

        int alloc_buff(uint64_t *buff_addr) {
            if (empty()) return -1;

            *buff_addr = (uint64_t)base_addr_ + buffer_pool_[head_];
            head_ = (head_ + 1) & (num_elements_ - 1);
            return 0;
        }

        void free_buff(uint64_t buff_addr) {
            if (full()) return;
            buff_addr -= (uint64_t)base_addr_;
            buffer_pool_[tail_] = buff_addr;
            tail_ = (tail_ + 1) & (num_elements_ - 1);
        }

    protected:
        void *base_addr_;
        uint32_t head_;
        uint32_t tail_;
        uint32_t num_elements_;
        size_t element_size_;
        struct ibv_mr *mr_;
        uint64_t *buffer_pool_;
};

/**
 * @brief Buffer pool for work request extension.
 * - Single producer, single consumer.
 */
class WrExBuffPool : public BuffPool {
    static constexpr uint32_t kWrSize = sizeof(struct wr_ex);
    static constexpr uint32_t kNumWr = 4096;
    static_assert((kNumWr & (kNumWr - 1)) == 0, "kNumWr must be power of 2");
    public:

        WrExBuffPool() : BuffPool(kNumWr, kWrSize, nullptr, [] (uint64_t buff) {
            struct wr_ex *wr_ex = reinterpret_cast<struct wr_ex *>(buff);
            auto wr = &wr_ex->wr;
            wr->sg_list = &wr_ex->sge;
            wr->num_sge = 1;
            wr->next = nullptr;
            wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        }) {}

        ~WrExBuffPool() = default;
};

struct retr_chunk_hdr {
    uint32_t qidx;
    uint32_t pad;
    uint64_t remote_addr;
};

class RetrChunkBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kRetrChunkSize = kChunkSize + sizeof(retr_chunk_hdr);
        static constexpr uint32_t kNumChunk = 64;
        static_assert((kNumChunk & (kNumChunk - 1)) == 0, "kNumChunk must be power of 2");

        RetrChunkBuffPool(struct ibv_mr *mr) : BuffPool(kNumChunk, kRetrChunkSize, mr) {}

        ~RetrChunkBuffPool() = default;
};

class RetrHdrBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kHdrSize = sizeof(struct retr_chunk_hdr);
        static constexpr uint32_t kNumHdr = 256;
        static_assert((kNumHdr & (kNumHdr - 1)) == 0, "kNumHdr must be power of 2");

        RetrHdrBuffPool(struct ibv_mr *mr) : BuffPool(kNumHdr, kHdrSize, mr) {}

        ~RetrHdrBuffPool() = default;
};

class CtrlPktBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kPktSize = 192;
        static constexpr uint32_t kNumPkt = 2048;
        static_assert((kNumPkt & (kNumPkt - 1)) == 0, "kNumPkt must be power of 2");

        CtrlPktBuffPool(struct ibv_mr *mr) : BuffPool(kNumPkt, kPktSize, mr) {}

        ~CtrlPktBuffPool() = default;
};

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

        inline void SetRID(uint32_t rid) {
            imm_data_ |= (rid & 0xFF) << kRID;
        }

        inline void SetMID(uint32_t mid) {
            imm_data_ |= (mid & 0x7) << kMID;
        }

        inline uint32_t GetImmData(void) {
            return imm_data_;
        }

    private:
        uint32_t imm_data_;    
};
struct XchgMeta {
    union {
        // Endpoint --> Engine
        struct {
            union {
                // kInstallFlowRDMA
                struct {
                    union ibv_gid remote_gid;
                    struct ibv_port_attr remote_port_attr;
                    ibv_mtu mtu;
                    int dev;
                    bool is_send;
                };

                // kRegMR/kRegMRDMABUF
                struct {
                    void *addr;
                    size_t len;
                    int type;
                    int offset;
                    int fd;
                };

                // kDeregMR
                struct {
                    struct ibv_mr *mr;
                };

                // kSyncFlowRDMA
                struct {
                    uint32_t remote_qpn;
                    uint32_t remote_psn;
                    bool fifo;
                    uint32_t fifo_key; // Only valid when fifo is true
                    uint64_t fifo_addr; // Only valid when fifo is true
                };
            };
        } ToEngine;
        // Engine --> Endpoint
        struct {
            union {
                // kCompleteFlowRDMA
                struct {
                    uint32_t local_qpn;
                    uint32_t local_psn;
                    bool fifo;
                    uint32_t fifo_key; // Only valid when fifo is true
                    uint64_t fifo_addr; // Only valid when fifo is true
                };
                // kCompleteRegMR
                struct {
                    struct ibv_mr *mr;
                };
            };
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
    uint32_t rid;
    char padding[28];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

struct RemFifo {
    // FIFO elements prepared for sending to remote peer.
    struct FifoItem elems[kMaxReq][kMaxRecv];
    uint32_t received_bytes[kMaxReq][kMaxRecv];
    // Tail pointer of the FIFO.
    uint64_t fifo_tail;
    // Only used for testing RC.
    uint32_t sizes[kMaxReq][kMaxRecv];
};

struct RemoteRDMAContext {
    union ibv_gid remote_gid;
    struct ibv_port_attr remote_port_attr;
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

    // Only used for testing RC.
    int events;

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

/// @ref ncclIbNetCommBase
struct alignas(32) NetCommBase {
    // Is this a send or receive flow?
    bool is_send;
    // Track outstanding SEND/RECV requests.
    struct FlowRequest reqs[kMaxReq];
    // Remote RDMA context.
    struct RemoteRDMAContext remote_ctx;
    // Pointing to rdma_ctx_->fifo_mr_->addr.
    struct RemFifo *fifo;
};

/// @ref ncclIbSendComm
struct SendComm {
    struct NetCommBase base;

    // Track outstanding FIFO requests.
    struct FlowRequest *fifo_reqs[kMaxReq][kMaxRecv];
    uint64_t fifo_head;
};

/// @ref ncclIbRecvComm
struct RecvComm {
    struct NetCommBase base;
};

class RXTracking {
        static constexpr uint32_t kMAXWQE = 4;
        static constexpr uint32_t kMAXBytes = kMAXWQE * kChunkSize;
    public:

        std::set<int> ready_csn_;
        
        RXTracking() = default;
        ~RXTracking() = default;

        // Immediate Acknowledgement.
        inline void cumulate_wqe(void) { cumulative_wqe_++;}
        inline void cumulate_bytes(uint32_t bytes) { cumulative_bytes_ += bytes;}
        inline void encounter_ooo(void) { ooo_ = true;}
        
        inline bool need_imm_ack(void) { return ooo_ || cumulative_wqe_ == kMAXWQE || cumulative_bytes_ >= kMAXBytes;}
        inline void clear_imm_ack(void) {
            ooo_ = false;
            cumulative_wqe_ = 0;
            cumulative_bytes_ = 0;
        }
        
        // Post recv WQEs.
        inline uint32_t need_fill(void) { return fill_cnt_++;}
        inline uint32_t fill_count(void) { return fill_cnt_;}
        inline void clear_fill(void) { fill_cnt_ = 0;}
    
    private:
        bool ooo_ = false;
        uint32_t cumulative_wqe_ = 0;
        uint32_t cumulative_bytes_ = 0;
        uint32_t fill_cnt_ = 0;
};

class TXTracking {
    public:
        struct ChunkTrack {
            struct FlowRequest *req;
            uint32_t csn;
            struct wr_ex *wr_ex;
            bool last_chunk;
            uint64_t timestamp;
        };

        TXTracking() = default;
        ~TXTracking() = default;

        inline bool empty(void) {
            return unacked_chunks_.empty();
        }

        inline TXTracking::ChunkTrack get_unacked_chunk_from_idx(uint32_t idx) {
            return unacked_chunks_[idx];
        }

        inline TXTracking::ChunkTrack get_oldest_unacked_chunk(void) {
            return unacked_chunks_.front();
        }

        uint64_t ack_chunks(uint32_t num_acked_chunks);

        inline void track_chunk(struct FlowRequest *req, uint32_t csn, struct wr_ex * wr_ex, bool last_chunk, uint64_t timestamp) {
            unacked_chunks_.push_back({req, csn, wr_ex, last_chunk, timestamp});
        }

        inline size_t track_size(void) {
            return unacked_chunks_.size();
        }

        inline uint64_t track_lookup_ts(uint32_t track_idx) {
            return unacked_chunks_[track_idx].timestamp;
        }

        inline void set_rdma_ctx(struct RDMAContext *rdma_ctx) {
            rdma_ctx_ = rdma_ctx;
        }

    private:
        struct RDMAContext *rdma_ctx_ = nullptr;
        std::vector<TXTracking::ChunkTrack> unacked_chunks_;
};

struct ack_item {
    uint32_t qpidx;
    struct list_head ack_link;
};

struct UCQPWrapper {
    struct ibv_qp *qp;
    uint32_t local_psn;
    uint32_t remote_psn;
    uint32_t in_wheel_cnt_ = 0;
    uint32_t signal_cnt_ = 0;
    swift::Pcb pcb;
    TXTracking txtracking;
    RXTracking rxtracking;
    struct ack_item ack;
};

/**
 * @brief  RDMA context for each UcclFlow, which is produced by RDMAFactory. It contains:
 *   - Multiple Unreliable Connection (UC) QPs and a shared CQ.
 *   - A high-priority QP for control messages and a dedicated CQ, PD, and MR.
 *   - A QP for retransmission and a dedicated CQ, PD, and MR.
 *   - A Reliable Connection (RC) QP for FIFO and a dedicated CQ, PD, and MR.
 *   - A MR for retransmission.
 */
class RDMAContext {
    public:
        constexpr static int kTotalQP = kPortEntropy + 3;
        constexpr static int kFifoIndex = kPortEntropy + 1;
        constexpr static int kFifoMRSize = sizeof(struct RemFifo);
        constexpr static int kCtrlMRSize = CtrlPktBuffPool::kPktSize * CtrlPktBuffPool::kNumPkt;
        /// TODO: How to determine the size of retransmission MR?
        constexpr static int kRetrMRSize = RetrChunkBuffPool::kRetrChunkSize * RetrChunkBuffPool::kNumChunk;
        constexpr static int kCQSize = 4096;

        // Protection domain for all RDMA resources.
        struct ibv_pd *pd_ = nullptr;

        // QPs for data transfer based on Unreliable Connection (UC).
        struct UCQPWrapper uc_qps_[kPortEntropy];
        // UC QPN to index mapping.
        std::unordered_map<uint32_t, int> qpn2idx_;
        // Shared CQ for all UC QPs.
        struct ibv_cq_ex *cq_ex_;

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

        // QP for GPU flush.
        struct ibv_qp *gpu_flush_qp_;
        // Memory region for GPU flush.
        struct ibv_mr *gpu_flush_mr_;
        struct ibv_sge gpu_flush_sge_;
        // GPU flush buffer
        int gpu_flush_;
        
        // (high-priority) QP for control messages (e.g., ACK).
        struct ibv_qp *ctrl_qp_;
        // Local PSN for control messages.
        uint32_t ctrl_local_psn_;
        // Remote PSN for control messages.
        uint32_t ctrl_remote_psn_;
        // Dedicated CQ for control messages.
        struct ibv_cq_ex *ctrl_cq_ex_;
        // Memory region for control messages.
        struct ibv_mr *ctrl_mr_;

        // Retransmission QP.
        struct ibv_qp *retr_qp_;
        // Local PSN for retransmission.
        uint32_t retr_local_psn_;
        // Remote PSN for retransmission.
        uint32_t retr_remote_psn_;
        // Dedicated CQ for retransmission.
        struct ibv_cq_ex *retr_cq_ex_;
        // Memory region for retransmission.
        struct ibv_mr *retr_mr_;
        struct ibv_mr *retr_hdr_mr_;
        uint32_t inflight_retr_chunks_;

        // Global timing wheel for all UC QPs.
        TimingWheel wheel_;

        // The device index that this context belongs to.
        int dev_;

        // RDMA device context per device.
        struct ibv_context *context_;
        // Local GID of this device.
        union ibv_gid local_gid_;
        // Port attributes of this device.
        struct ibv_port_attr port_attr_;
        // MTU of this device.
        ibv_mtu mtu_;
        uint32_t mtu_bytes_;
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

        // Buffer pool for control packets.
        std::optional<CtrlPktBuffPool> ctrl_pkt_pool_;

        // Buffer pool for retransmission headers.
        std::optional<RetrHdrBuffPool> retr_hdr_pool_;

        // Buffer pool for retransmission chunks.
        std::optional<RetrChunkBuffPool> retr_chunk_pool_;

        // Buffer pool for work request extension items.
        std::optional<WrExBuffPool> wr_ex_pool_;

        /**
         * @brief Figure out the request id.
         * @param req 
         * @param comm_base 
         * @return uint64_t 
         */
        inline uint64_t get_request_id(struct FlowRequest *req, struct NetCommBase *comm_base) {
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
            LOG(INFO) << "free_request: " << req;
            req->type = FlowRequest::UNUSED;
            req->nmsgs = 0;
            req->poll_ctx = nullptr;
            req->events = 0;
            if (!is_send_) {
                req->recv.received_bytes = 0;
                req->recv.fin_msg = 0;
                req->recv.elems = nullptr;
            }
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
                    LOG(INFO) << "get_request: " << req;
                    return req;
                }
            }
            LOG(INFO) << "get_request: nullptr";
            return nullptr;
        }
        
        // Select a QP index in a round-robin manner.
        inline uint32_t select_qpidx_rr(void) {
            static uint32_t next_qp_idx = 0;
            return next_qp_idx++ % kPortEntropy;
        }

        // Select a QP index randomly.
        inline uint32_t select_qpidx_rand(void) {
            return std::rand() % kPortEntropy;
        }

        // When ready_entropy_cnt_ equals to kTotalQP, the flow is ready.
        uint32_t ready_entropy_cnt_;

        RDMAContext(int dev, struct XchgMeta meta);

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

    // DMA-BUF support
    bool dma_buf_support;
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
        static RDMAContext *CreateContext(int dev, struct XchgMeta meta);
        static struct FactoryDevice *get_factory_dev(int dev);
        static bool need_sync_clock(int dev);
        static void shutdown(void);
        
        std::string to_string(void) const;
};

static inline uint16_t util_rdma_extract_local_subnet_prefix(uint64_t subnet_prefix)
{
    return (be64toh(subnet_prefix) & 0xffff);
}

static inline int modify_qp_rtr_gpuflush(struct ibv_qp *qp, RDMAContext *rdma_ctx)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN;

    auto comm_base = &rdma_ctx->recv_comm_.base;

    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_ctx->mtu_;
    if (USE_ROCE) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = rdma_ctx->local_gid_;
        attr.ah_attr.grh.sgid_index = rdma_ctx->sgid_index_;
        attr.ah_attr.grh.hop_limit = 0xff;
    } else {
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = rdma_ctx->port_attr_.lid;
    }

    attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
    attr.dest_qp_num = qp->qp_num;
    attr.rq_psn = 0;

    attr.min_rnr_timer = 12;
    attr.max_dest_rd_atomic = 1;
    attr_mask |= IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

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

static inline int modify_qp_rtr(struct ibv_qp *qp, RDMAContext *rdma_ctx, uint32_t remote_qpn, uint32_t remote_psn)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    auto comm_base = rdma_ctx->is_send_ ? &rdma_ctx->send_comm_.base : &rdma_ctx->recv_comm_.base;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = rdma_ctx->mtu_;
    if (USE_ROCE) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
        attr.ah_attr.grh.dgid = comm_base->remote_ctx.remote_gid;
        attr.ah_attr.grh.sgid_index = rdma_ctx->sgid_index_;
        attr.ah_attr.grh.hop_limit = 0xff;
    } else {
        if (util_rdma_extract_local_subnet_prefix(rdma_ctx->local_gid_.global.subnet_prefix) != 
            util_rdma_extract_local_subnet_prefix(comm_base->remote_ctx.remote_gid.global.subnet_prefix)) {
                LOG(ERROR) << "Only support same subnet communication for now.";
        }
        attr.ah_attr.is_global = 0;
        attr.ah_attr.port_num = rdma_ctx->ib_port_num_;
        attr.ah_attr.dlid = comm_base->remote_ctx.remote_port_attr.lid;
    }
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = remote_psn;

    if (qp->qp_type == IBV_QPT_RC) {
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

static inline void util_rdma_create_qp(RDMAContext *rdma_ctx, struct ibv_context *context, struct ibv_qp **qp, enum ibv_qp_type qp_type, bool cq_ex, bool ts,
    struct ibv_cq **cq, bool share_cq, uint32_t cqsize, struct ibv_pd *pd, struct ibv_mr **mr, void *addr, 
        size_t mr_size, uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge)
{
    // Creating CQ
    if (!share_cq) {
        if (cq_ex) {
            struct ibv_cq_init_attr_ex cq_ex_attr;
            cq_ex_attr.cqe = cqsize;
            cq_ex_attr.cq_context = nullptr;
            cq_ex_attr.channel = nullptr;
            cq_ex_attr.comp_vector = 0;
            cq_ex_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM | IBV_WC_EX_WITH_SRC_QP | 
                IBV_WC_EX_WITH_COMPLETION_TIMESTAMP; // Timestamp support.
            if (kTestNoHWTimestamp)
                cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
            cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
            cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED | IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
            auto cq_ex = (struct ibv_cq_ex **)cq;
            *cq_ex= ibv_create_cq_ex(context, &cq_ex_attr);
            if (*cq_ex == nullptr)
                throw std::runtime_error("ibv_create_cq_ex failed");
        } else {
            *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
            if (*cq == nullptr)
                throw std::runtime_error("ibv_create_cq failed");
        }
    }
    
    // Creating MR
    if (addr == nullptr) {
        addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (addr == MAP_FAILED)
            throw std::runtime_error("mmap failed");
    }
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
    qp_init_attr.cap.max_send_sge = max_send_sge;
    qp_init_attr.cap.max_recv_sge = max_recv_sge;
    qp_init_attr.cap.max_inline_data = kMaxRecv * sizeof(struct FifoItem);

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
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

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