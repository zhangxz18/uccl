#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <set>

#include <sys/mman.h>
#include <infiniband/verbs.h>

#include <glog/logging.h>

#include "transport_config.h"
#include "transport_cc.h"

#include "util.h"
#include "util_endian.h"
#include "util_list.h"

namespace uccl {

typedef uint64_t FlowID;
typedef uint64_t PeerID;

class RDMAContext;
class RDMAFactory;
extern std::shared_ptr<RDMAFactory> rdma_ctl;

// LRH (Local Routing Header) + GRH (Global Routing Header) + BTH (Base Transport Header)
static constexpr uint32_t IB_HDR_OVERHEAD = (8 + 40 + 12);
// Ethernet + IPv4 + UDP + BTH
static constexpr uint32_t ROCE_IPV4_HDR_OVERHEAD = (14 + 20 + 8 + 12);
// Ethernet + IPv6 + UDP + BTH
static constexpr uint32_t ROCE_IPV6_HDR_OVERHEAD = (14 + 40 + 8 + 12);

static constexpr uint32_t BASE_PSN = 0;

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
 * - Actual size is num_elements - 1.
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
            // Reserve one element for distinguished empty/full state.
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

        inline bool full(void) {
            return ((tail_ + 1) & (num_elements_ - 1)) == head_;
        }

        inline bool empty(void) {
            return head_ == tail_;
        }

        inline uint32_t get_lkey(void) {
            if (!mr_) return 0;
            return mr_->lkey;
        }

        inline int alloc_buff(uint64_t *buff_addr) {
            if (empty()) return -1;

            *buff_addr = (uint64_t)base_addr_ + buffer_pool_[head_];
            head_ = (head_ + 1) & (num_elements_ - 1);
            return 0;
        }

        inline void free_buff(uint64_t buff_addr) {
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
    // Lossy QP index for the lost chunk.
    uint32_t qidx;
    uint32_t pad;
    // Target address for the lost chunk.
    uint64_t remote_addr;
};

/**
 * @brief Buffer pool for retransmission chunks (original chunk + retransmission header).
 * Original chunk and retransmission header are transmitted through scatter-gather list.
 */
class RetrChunkBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kRetrChunkSize = kChunkSize + sizeof(retr_chunk_hdr);
        static constexpr uint32_t kNumChunk = 128;
        static_assert((kNumChunk & (kNumChunk - 1)) == 0, "kNumChunk must be power of 2");

        RetrChunkBuffPool(struct ibv_mr *mr) : BuffPool(kNumChunk, kRetrChunkSize, mr) {}

        ~RetrChunkBuffPool() = default;
};

/**
 * @brief Buffer pool for retransmission headers.
 */
class RetrHdrBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kHdrSize = sizeof(struct retr_chunk_hdr);
        static constexpr uint32_t kNumHdr = 256;
        static_assert((kNumHdr & (kNumHdr - 1)) == 0, "kNumHdr must be power of 2");

        RetrHdrBuffPool(struct ibv_mr *mr) : BuffPool(kNumHdr, kHdrSize, mr) {}

        ~RetrHdrBuffPool() = default;
};

/**
 * @brief Buffer pool for control packets.
 */
class CtrlChunkBuffPool : public BuffPool {
    public:
        static constexpr uint32_t kPktSize = 32;
        static constexpr uint32_t kChunkSize = kPktSize * kMaxBatchCQ;
        static constexpr uint32_t kNumChunk = kMaxBatchCQ << 6;
        static_assert((kNumChunk & (kNumChunk - 1)) == 0, "kNumChunk must be power of 2");

        CtrlChunkBuffPool(struct ibv_mr *mr) : BuffPool(kNumChunk, kChunkSize, mr) {}

        ~CtrlChunkBuffPool() = default;
};

class IMMData {
    public:
        // High--------------------32bit----------------------Low
        //  |***Reserved***|  NCHUNK  |  CSN  |  RID  |  MID  |
        //       3bit          10bit     8bit    8bit    3bit
        constexpr static int kMID = 0;
        constexpr static int kRID = 3;
        constexpr static int kCSN = 11;
        constexpr static int kNCHUNK = kCSN + UINT_CSN_BIT;

        IMMData(uint32_t imm_data):imm_data_(imm_data) {}
        ~IMMData() = default;

        inline uint32_t GetNCHUNK(void) {
            return (imm_data_ >> kNCHUNK) & 0x3FF;
        }

        inline uint32_t GetCSN(void) {
            return (imm_data_ >> kCSN) & UINT_CSN_MASK;
        }

        inline uint32_t GetRID(void) {
            return (imm_data_ >> kRID) & 0xFF;
        }

        inline uint32_t GetMID(void) {
            return (imm_data_ >> kMID) & 0x7;
        }

        inline void SetNCHUNK(uint32_t nchunk) {
            imm_data_ |= (nchunk & 0x3FF) << kNCHUNK;
        }

        inline void SetCSN(uint32_t psn) {
            imm_data_ |= (psn & UINT_CSN_MASK) << kCSN;
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
/**
 * @brief Metadata for control messages.
 */
union CtrlMeta {
    // kInstallCtx
    struct {
        union ibv_gid remote_gid;
        struct ibv_port_attr remote_port_attr;
        bool is_send;
        PeerID peer_id;
        int bootstrap_fd;
    } install_ctx;
};

struct FifoItem {
    uint64_t addr;
    uint32_t size;
    uint32_t rkey;
    uint32_t nmsgs;
    uint32_t rid;
    uint64_t idx;
    uint32_t engine_offset;
    char padding[28];
};
static_assert(sizeof(struct FifoItem) == 64, "FifoItem size is not 64 bytes");

/**
 * @brief A FIFO queue for flow control.
 * Receiver posts a buffer to the FIFO queue for the sender to use RDMA WRITE.
 */
struct RemFifo {
    // FIFO elements prepared for sending to remote peer.
    struct FifoItem elems[kMaxReq][kMaxRecv];
    // Tail pointer of the FIFO.
    uint64_t fifo_tail;
    // Only used for testing RC.
    uint32_t sizes[kMaxReq][kMaxRecv];
};

struct RemoteRDMAContext {
    union ibv_gid remote_gid;
    struct ibv_port_attr remote_port_attr;
};

enum ReqType {
    ReqTx,
    ReqRx,
    ReqFlush,
    ReqTxRC,
    ReqRxRC,
};

/**
 * @brief ucclRequest is a handle provided by the user to post a request to UCCL RDMAEndpoint.
 * It is the responsibility of the user to manage the memory of ucclRequest. UCCL RDMAEndpoint 
 * will not free the memory of ucclRequest. UCCL fills the ucclRequest with the result of the 
 * request. The user can use the ucclRequest to check the status of the request.
 */
struct ucclRequest {
    enum ReqType type;
    int n;
    PollCtx *poll_ctx;
    void *context;
    void *req_pool;
    union {
        struct {
            int data_len[kMaxRecv];
            uint64_t data[kMaxRecv];
            struct FifoItem *elems;
            struct ibv_send_wr wr;
            struct ibv_sge sge;
            struct ibv_qp *qp;
        } recv;
        struct {
            int data_len;
            uint64_t laddr;
            uint64_t raddr;
            uint32_t lkey;
            uint32_t rkey;
            uint32_t rid;
            int tx_events;
        } send;
    };
};

/**
 * @brief Each RDMAContext has a pool of RecvRequest.
 * After the recevier posting an async recv ucclRequest to an engine, the engine will allocate a RecvRequest
 * from its RDMAContext. Then, when receiving the data, the engine will locate the RecvRequest and then
 * further find the ucclRequest.
 */
struct RecvRequest {
    enum type {
        UNUSED = 0,
        RECV,
    };
    enum type type;
    struct ucclRequest *ureq;
    uint32_t received_bytes[kMaxRecv];
    uint32_t fin_msg;
};

/// @ref ncclIbNetCommBase
struct alignas(32) NetCommBase {
    // Pointing to rdma_ctx_->fifo_mr_->addr.
    struct RemFifo *fifo;

    // CQ for Fifo QP and GPU flush QP and RC QP.
    struct ibv_cq *flow_cq;

    // Fifo QP based on Reliable Connection (RC).
    struct ibv_qp *fifo_qp;
    // Local PSN for Fifo.
    uint32_t fifo_local_psn;
    // Memory region for Fifo.
    struct ibv_mr *fifo_mr;

    // RC UP for small messages bypassing UcclEngine.
    struct ibv_qp *rc_qp;
    uint32_t rc_local_psn;

    uint64_t remote_fifo_addr;
    uint32_t remote_fifo_rkey;
};

/// @ref ncclIbSendComm
struct SendComm {
    struct NetCommBase base;
    // Track outstanding FIFO requests.
    struct ucclRequest *fifo_ureqs[kMaxReq][kMaxRecv];
    uint64_t fifo_head;
};

/// @ref ncclIbRecvComm
struct RecvComm {
    struct NetCommBase base;

    // QP for GPU flush.
    struct ibv_qp *gpu_flush_qp;
    // Memory region for GPU flush.
    struct ibv_mr *gpu_flush_mr;
    struct ibv_sge gpu_flush_sge;
    // GPU flush buffer
    int gpu_flush;
};

class RXTracking {
        // 1 means always send immediate ack.
        static constexpr uint32_t kMAXWQE = 4;
        static constexpr uint32_t kMAXBytes = kMAXWQE * kChunkSize;
    public:

        std::set<UINT_CSN> ready_csn_;
        
        RXTracking() = default;
        ~RXTracking() = default;

        // Immediate Acknowledgement.
        inline void cumulate_wqe(void) { cumulative_wqe_++;}
        inline void cumulate_bytes(uint32_t bytes) { cumulative_bytes_ += bytes;}
        inline void encounter_ooo(void) { ooo_ = true;}
        
        /**
         * @brief Send ack immediately if the following conditions are met:
         * 1. Out-of-order packets are received.
         * 2. The number of received WQE reaches kMAXWQE.
         * 3. The number of received bytes reaches kMAXBytes.
         */
        inline bool need_imm_ack(void) { return ooo_ || cumulative_wqe_ == kMAXWQE || cumulative_bytes_ >= kMAXBytes;}
        /**
         * @brief After sending immediate ack, clear the states.
         */
        inline void clear_imm_ack(void) {
            ooo_ = false;
            cumulative_wqe_ = 0;
            cumulative_bytes_ = 0;
        }
    
    private:
        bool ooo_ = false;
        uint32_t cumulative_wqe_ = 0;
        uint32_t cumulative_bytes_ = 0;
};

class TXTracking {
    public:
        struct ChunkTrack {
            struct ucclRequest *ureq;
            uint32_t csn;
            struct wr_ex *wr_ex;
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

        uint64_t ack_transmitted_chunks(uint32_t num_acked_chunks);

        inline void track_chunk(struct ucclRequest *ureq, uint32_t csn, struct wr_ex * wr_ex, uint64_t timestamp) {
            unacked_chunks_.push_back({ureq, csn, wr_ex, timestamp});
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

/**
 * @brief UCQPWrapper is a wrapper for ibv_qp with additional information for 
 * implementing reliable data transfer.
 */
struct UCQPWrapper {
    struct ibv_qp *qp;
    uint32_t local_psn;
    // # of chunks in the timing wheel for in-order within the same QP.
    uint32_t in_wheel_cnt_ = 0;
    // A counter for occasionally posting IBV_SEND_SIGNALED flag.
    uint32_t signal_cnt_ = 0;
    // Congestion control state.
    swift::Pcb pcb;
    // States for tracking sent chunks.
    TXTracking txtracking;
    // States for tracking received chunks.
    RXTracking rxtracking;
    // We use list_empty(&qpw->ack.ack_link) to check if it has pending ACK to send.
    struct ack_item ack;
    bool rto_armed = false;
};

/**
 * @brief UCCL SACK Packet Header for each QP.
 * Multiple SACKs are packed in a single packet transmitted through the Ctrl QP.
 */
struct __attribute__((packed)) UcclSackHdr {
    be16_t qpidx;  // QP index.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    be64_t remote_queueing;   // t_ack_sent (SW) - t_remote_nic_rx (HW)
    be64_t sack_bitmap[kSackBitmapSize /
                       swift::Pcb::kSackBitmapBucketSize];  // Bitmap of the
                                                            // SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
};
static const size_t kUcclSackHdrLen = sizeof(UcclSackHdr);
static_assert(kUcclSackHdrLen == 32, "UcclSackHdr size mismatch");
static_assert(CtrlChunkBuffPool::kPktSize >= kUcclSackHdrLen, "CtrlChunkBuffPool::PktSize must be larger than UcclSackHdr");

class UcclEngine;

/**
 * @brief RDMA context for a remote peer on an engine, which is produced by RDMAFactory. It contains:
 *   - (UC QP): Multiple Unreliable Connection QPs and a shared CQ. All UC QPs share the same SRQ.
 *   - (Ctrl QP): A high-priority QP for control messages and a dedicated CQ, PD, and MR.
 *   - (Retr QP): A QP for retransmission and a dedicated CQ, PD, and MR.
 */
class RDMAContext {
    public:
        constexpr static int kTotalQP = kPortEntropy + 2;
        constexpr static int kCtrlMRSize = CtrlChunkBuffPool::kChunkSize * CtrlChunkBuffPool::kNumChunk;
        /// TODO: How to determine the size of retransmission MR?
        constexpr static int kRetrMRSize = RetrChunkBuffPool::kRetrChunkSize * RetrChunkBuffPool::kNumChunk;
        constexpr static int kCQSize = 4096;
        // 256-bit SACK bitmask => we can track up to 256 packets
        static constexpr std::size_t kReassemblyMaxSeqnoDistance = kSackBitmapSize;

        // Track outstanding RECV requests.
        struct RecvRequest reqs_[kMaxReq];

        inline uint64_t get_recvreq_id(struct RecvRequest *req) {
            return req - reqs_;
        }

        inline struct RecvRequest *get_recvreq_by_id(int id) {
            return &reqs_[id];
        }

        inline void free_recvreq(struct RecvRequest *req) {
            VLOG(3) << "free_recvreq: " << req;
            memset(req, 0, sizeof(struct RecvRequest));
        }

        /**
         * @brief Get an unused request, if no request is available, return nullptr.
         * @return struct RecvRequest* 
         */
        inline struct RecvRequest *alloc_recvreq(void) {
            for (int i = 0; i < kMaxReq; i++) {
                auto *req = &reqs_[i];
                if (req->type == RecvRequest::UNUSED) {
                    VLOG(3) << "alloc_recvreq: " << req;
                    return req;
                }
            }
            VLOG(3) << "alloc_recvreq: nullptr";
            return nullptr;
        }

        TimerManager *rto_;

        // Try to arm a timer for the given QP. If the timer is already armed, do nothing.
        inline void arm_timer_for_qp(struct UCQPWrapper *qpw) {
            if (!qpw->rto_armed) {
                rto_->arm_timer({this, qpw});
                qpw->rto_armed = true;
            }
        }

        // Try to rearm a timer for the given QP. If the timer is not armed, arm it.
        // If the timer is already armed, rearm it.
        inline void rearm_timer_for_qp(struct UCQPWrapper *qpw) {
            if (qpw->rto_armed) {
                rto_->rearm_timer({this, qpw});
            } else {
                arm_timer_for_qp(qpw);
            }
        }

        inline void mark_qp_timeout(struct UCQPWrapper *qpw) {
            qpw->rto_armed = false;
        }

        inline void disarm_timer_for_qp(struct UCQPWrapper *qpw) {
            if (qpw->rto_armed) {
                rto_->disarm_timer({this, qpw});
                qpw->rto_armed = false;
            }
        }

        // Remote RDMA context.
        struct RemoteRDMAContext remote_ctx_;

        // Protection domain for all RDMA resources.
        struct ibv_pd *pd_ = nullptr;

        // QPs for data transfer based on Unreliable Connection (UC).
        struct UCQPWrapper uc_qps_[kPortEntropy];
        // UC QPN to index mapping.
        std::unordered_map<uint32_t, int> qpn2idx_;
        
        // Shared CQ for all UC QPs.
        struct ibv_cq_ex *send_cq_ex_;
        struct ibv_cq_ex *recv_cq_ex_;
        struct ibv_srq *srq_;
        
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
        uint32_t inflight_retr_chunks_ = 0;

        // Global timing wheel for all UC QPs.
        TimingWheel wheel_;

        // The device index that this context belongs to.
        int dev_;

        // RDMA device context per device.
        struct ibv_context *context_;
        // MTU of this device.
        ibv_mtu mtu_;
        uint32_t mtu_bytes_;
        // GID index of this device.
        uint8_t sgid_index_;

        // Buffer pool for control chunks.
        std::optional<CtrlChunkBuffPool> ctrl_chunk_pool_;

        // Buffer pool for retransmission headers.
        std::optional<RetrHdrBuffPool> retr_hdr_pool_;

        // Buffer pool for retransmission chunks.
        std::optional<RetrChunkBuffPool> retr_chunk_pool_;

        // Buffer pool for work request extension items.
        std::optional<WrExBuffPool> wr_ex_pool_;

        // Pre-allocated WQEs for consuming retransmission chunks.
        struct ibv_recv_wr retr_wrs_[kMaxBatchCQ];

        // WQE for sending ACKs.
        struct ibv_send_wr tx_ack_wr_;
        
        // Pre-allocated WQEs for receiving ACKs.
        struct ibv_recv_wr rx_ack_wrs_[kPostRQThreshold];
        // Pre-allocted SGEs for receiving ACKs.
        struct ibv_sge rx_ack_sges_[kPostRQThreshold];
        uint32_t post_ctrl_rq_cnt_ = 0;

        // Pre-allocated WQEs for consuming immediate data.
        struct ibv_recv_wr imm_wrs_[kPostRQThreshold];
        uint32_t post_srq_cnt_ = 0;

        // When the Retr chunk pool exhausts, we can't post enough WQEs to the Retr RQ.
        uint32_t fill_retr_rq_cnt_ = 0;

        double ratio_;
        double offset_;

        inline void update_clock(double ratio, double offset) {
            ratio_ = ratio;
            offset_ = offset;
        }

        // Convert NIC clock to host clock (TSC).
        inline uint64_t convert_nic_to_host(uint64_t host_clock, uint64_t nic_clock) {
            return ratio_ * nic_clock + offset_;
        }
        
        // Select a QP index in a round-robin manner.
        inline uint32_t select_qpidx_rr(void) {
            static uint32_t next_qp_idx = 0;
            return next_qp_idx++ % kPortEntropy;
        }

        // Select a QP index randomly.
        inline uint32_t select_qpidx_rand() {
            static thread_local std::mt19937 generator(std::random_device{}());
            std::uniform_int_distribution<uint32_t> distribution(0, kPortEntropy - 1);
            return distribution(generator);
        }

        // Select a QP index in a power-of-two manner.
        inline uint32_t select_qpidx_pow2(void) {
            auto q1 = select_qpidx_rand();
            auto q2 = select_qpidx_rand();

            // Return the QP with lower RTT.
            return uc_qps_[q1].pcb.timely.prev_rtt_ < uc_qps_[q2].pcb.timely.prev_rtt_ ? q1 : q2;
        }

        void tx_messages(struct ucclRequest *ureq);

        int supply_rx_buff(struct ucclRequest *ureq);

        /**
         * @brief Poll the completion queues for all UC QPs.
         * SQ and RQ use separate completion queues.
         */
        inline int poll_uc_cq(void) { int work = sender_poll_uc_cq(); work += receiver_poll_uc_cq(); return work;}
        int sender_poll_uc_cq(void);
        int receiver_poll_uc_cq(void);

        /**
         * @brief Poll the completion queue for the Ctrl QP.
         * SQ and RQ use the same completion queue.
         */
        int poll_ctrl_cq(void);

        /**
         * @brief Poll the completion queue for the Retr QP.
         * SQ and RQ use the same completion queue.
         */
        int poll_retr_cq(void);

        /**
         * @brief Check if we need to post enough recv WQEs to the Ctrl QP.
         * @param force Force to post WQEs.
         */
        void check_ctrl_rq(bool force = false);

        /**
         * @brief Check if we need to post enough recv WQEs to the SRQ.
         * @param force Force to post WQEs.
         */
        void check_srq(bool force = false);

        /**
         * @brief Retransmit a chunk for the given UC QP.
         * @param qpw 
         * @param wr_ex 
         */
        void retransmit_chunk(struct UCQPWrapper *qpw, struct wr_ex *wr_ex);

        /**
         * @brief Receive a chunk from the flow.
         * @param ack_list If this QP needs ACK, add it to the list.
         */
        void rx_chunk(struct list_head *ack_list);

        /**
         * @brief Receive a retransmitted chunk from the flow.
         * @param ack_list If this QP needs ACK, add it to the list.
         */
        void rx_retr_chunk(struct list_head *ack_list);

        /**
         * @brief Receive a barrier from the flow.
         * @param ack_list If this QP needs ACK, add it to the list.
         */
        void rx_barrier(struct list_head *ack_list);

        /**
         * @brief Rceive an ACK from the Ctrl QP.
         * @param pkt_addr The position of the ACK packet in the ACK chunk.
         */
        void rx_ack(uint64_t pkt_addr);

        /**
         * @brief Craft an ACK for a UC QP using the given WR index.
         * 
         * @param qpidx 
         * @param chunk_addr
         * @param num_sge
         */
        void craft_ack(int qpidx, uint64_t chunk_addr, int num_sge);

        /**
         * @brief Flush all ACKs in the batch.
         * 
         * @param num_ack 
         * @param chunk_addr
         */
        void flush_acks(int num_ack, uint64_t chunk_addr);

        /**
         * @brief Transmit a batch of chunks queued in the timing wheel.
         */
        void burst_timing_wheel(void);

        /**
         * @brief Try to update the CSN for the given UC QP.
         * @param qpw 
         */
        void try_update_csn(struct UCQPWrapper *qpw);

        /**
         * @brief Periodically checks the state of the flow and performs
         * necessary actions.
         *
         * This method is called periodically to check the state of the flow,
         * update the RTO timer, retransmit unacknowledged messages, and
         * potentially remove the flow or notify the application about the
         * connection state.
         *
         * @return Returns true if the flow should continue to be checked
         * periodically, false if the flow should be removed or closed.
         */
        bool periodic_check();

        /**
         * @brief Retransmit chunks for the given QP.
         * @param qpw 
         * @param rto 
         */
        void __retransmit(struct UCQPWrapper *qpw, bool rto);
        inline void fast_retransmit(struct UCQPWrapper *qpw) { __retransmit(qpw, false); }
        inline void rto_retransmit(struct UCQPWrapper *qpw) { __retransmit(qpw, true); }

        std::string to_string();

        RDMAContext(TimerManager *rto, int dev, union CtrlMeta meta);

        ~RDMAContext(void);
        
        friend class RDMAFactory;
};

struct FactoryDevice {
    char ib_name[64];
    std::string local_ip_str;
    
    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    
    uint8_t ib_port_num;
    uint8_t gid_idx;
    union ibv_gid gid;

    struct ibv_pd *pd;

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
    
    public:

        ~RDMAFactory() {
            devices_.clear();
            gid_2_dev_map.clear();
        }

        /**
         * @brief Initialize RDMA device for the given GID index.
         * @param gid_idx GID index, which usually starts from 0.
         */
        static void init_dev(int gid_idx);
        /**
         * @brief Create a Context object for the given device using the given meta.
         * @param dev 
         * @param meta 
         * @return RDMAContext* 
         */
        static RDMAContext *CreateContext(TimerManager *rto, int dev, union CtrlMeta meta);
        static inline struct FactoryDevice *get_factory_dev(int dev) {
            DCHECK(dev >= 0 && dev < rdma_ctl->devices_.size());
            return &rdma_ctl->devices_[dev];
        }
        
        std::string to_string(void) const;
};

static inline uint16_t util_rdma_extract_local_subnet_prefix(uint64_t subnet_prefix)
{
    return (be64toh(subnet_prefix) & 0xffff);
}

static inline int modify_qp_rtr_gpuflush(struct ibv_qp *qp, int dev)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN;

    memset(&attr, 0, sizeof(attr));

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = factory_dev->port_attr.active_mtu;
    if (USE_ROCE) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid = factory_dev->gid;
        attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
        attr.ah_attr.grh.hop_limit = 0xff;
    } else {
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = factory_dev->port_attr.lid;
    }

    attr.ah_attr.port_num = IB_PORT_NUM;
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
        VLOG(6) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline int modify_qp_rtr(struct ibv_qp *qp, int dev, struct RemoteRDMAContext *remote_ctx, uint32_t remote_qpn, uint32_t remote_psn, uint8_t sl)
{
    struct ibv_qp_attr attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = factory_dev->port_attr.active_mtu;
    if (USE_ROCE) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = IB_PORT_NUM;
        attr.ah_attr.grh.dgid = remote_ctx->remote_gid;
        attr.ah_attr.grh.sgid_index = factory_dev->gid_idx;
        attr.ah_attr.grh.hop_limit = 0xff;
    } else {
        if (util_rdma_extract_local_subnet_prefix(factory_dev->gid.global.subnet_prefix) != 
            util_rdma_extract_local_subnet_prefix(remote_ctx->remote_gid.global.subnet_prefix)) {
                LOG(ERROR) << "Only support same subnet communication for now.";
        }
        attr.ah_attr.is_global = 0;
        attr.ah_attr.port_num = IB_PORT_NUM;
        attr.ah_attr.dlid = remote_ctx->remote_port_attr.lid;
    }
    attr.ah_attr.sl = sl;
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
        VLOG(6) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline int modify_qp_rts(struct ibv_qp *qp, uint32_t local_psn, bool rc)
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
        VLOG(6) << oss.str();
    }
    
    return ibv_modify_qp(qp, &attr, attr_mask);
}

static inline void util_rdma_create_qp(struct ibv_context *context, struct ibv_qp **qp, enum ibv_qp_type qp_type, bool cq_ex, bool ts,
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
            if constexpr (kTestNoHWTimestamp)
                cq_ex_attr.wc_flags &= ~IBV_WC_EX_WITH_COMPLETION_TIMESTAMP;
            cq_ex_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
            cq_ex_attr.flags = IBV_CREATE_CQ_ATTR_SINGLE_THREADED | IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
            auto cq_ex = (struct ibv_cq_ex **)cq;
            *cq_ex= ibv_create_cq_ex(context, &cq_ex_attr);
            UCCL_INIT_CHECK(*cq_ex != nullptr, "ibv_create_cq_ex failed");
        } else {
            *cq = ibv_create_cq(context, cqsize, nullptr, nullptr, 0);
            UCCL_INIT_CHECK(*cq != nullptr, "ibv_create_cq failed");
        }
    }
    
    // Creating MR
    if (addr == nullptr) {
        addr = mmap(nullptr, mr_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        UCCL_INIT_CHECK(addr != MAP_FAILED, "mmap failed");
    }
    memset(addr, 0, mr_size);

    *mr = ibv_reg_mr(pd, addr, mr_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0));
    UCCL_INIT_CHECK(*mr != nullptr, "ibv_reg_mr failed");

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.send_cq = *cq;
    qp_init_attr.recv_cq = *cq;
    qp_init_attr.qp_type = qp_type;

    qp_init_attr.cap.max_send_wr = max_send_wr;
    qp_init_attr.cap.max_recv_wr = max_recv_wr;
    qp_init_attr.cap.max_send_sge = max_send_sge;
    qp_init_attr.cap.max_recv_sge = max_recv_sge;
    // kMaxRecv * sizeof(struct FifoItem)
    qp_init_attr.cap.max_inline_data = kMaxInline;

    // Creating QP
    *qp = ibv_create_qp(pd, &qp_init_attr);
    UCCL_INIT_CHECK(*qp != nullptr, "ibv_create_qp failed");

    // Modifying QP state to INIT
    struct ibv_qp_attr qp_attr;
    int attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_NUM;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | ((qp_type == IBV_QPT_RC) ? IBV_ACCESS_REMOTE_READ : 0);

    UCCL_INIT_CHECK(ibv_modify_qp(*qp, &qp_attr, attr_mask) == 0, "ibv_modify_qp failed");
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