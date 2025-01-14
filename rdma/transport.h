#pragma once

#include <netdb.h>
#include <bitset>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <infiniband/verbs.h>

#include <glog/logging.h>

#include "transport_cc.h"
#include "transport_config.h"
#include "util.h"
#include "util_shared_pool.h"
#include "util_endian.h"
#include "util_latency.h"
#include "util_rss.h"
#include "util_timer.h"
#include "util_rdma.h"

namespace uccl {

typedef uint64_t FlowID;

struct ConnID {
    FlowID flow_id;       // Used for UcclRDMAEngine to look up UcclFlow.
    uint32_t engine_idx;  // Used for RDMAEndpoint to locate the right engine.
    int boostrap_id;      // Used for bootstrap connection with the peer.
};

struct Mhandle {
    struct ibv_mr *mr;
};

/**
 * @class Channel
 * @brief A channel is a command queue for application threads to submit rx and
 * tx requests to the UcclFlow. A channel is only served by one UcclFlow, but
 * could be shared by multiple app threads if needed.
 */
class Channel {
    constexpr static uint32_t kChannelSize = 1024;

   public:
    struct Msg {
        enum Op : uint8_t {
            kTx,
            kRx,
            kFlush,
        };
        Op opcode;
        FlowID flow_id;
        struct ucclRequest *ureq;
        
        union {
            // kTx
            struct {
                void *data;
                size_t size;
                struct ibv_mr *mr;
            } tx;
            // kRx
            struct {
                void *data[kMaxRecv];
                size_t size[kMaxRecv];
                struct ibv_mr *mr[kMaxRecv];
                int n;
            } rx;
            // kFlush
            struct {
                void *data[kMaxRecv];
                size_t size[kMaxRecv];
                struct ibv_mr *mr[kMaxRecv];
                int n;
            } flush;
        };
        // Wakeup handler
        PollCtx *poll_ctx;
    };
    static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

    struct CtrlMsg {
        enum Op : uint8_t {
            // Endpoint --> Engine
            kInstallFlowRDMA = 0,
            kSyncFlowRDMA,
            kRegMR,
            kRegMRDMABUF,
            kDeregMR,
            // Engine --> Endpoint
            kCompleteFlowRDMA,
            kCompleteRegMR,
        };
        Op opcode;
        FlowID flow_id;
        
        struct XchgMeta meta;
        // Wakeup handler
        PollCtx *poll_ctx;
    };
    static_assert(sizeof(CtrlMsg) % 4 == 0,
                  "channelMsg must be 32-bit aligned");

    Channel() {
        tx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
        rx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
        ctrl_cmdq_ = create_ring(sizeof(CtrlMsg), kChannelSize);
        ctrl_rspq_ = create_ring(sizeof(CtrlMsg), kChannelSize);
    }

    ~Channel() {
        free(tx_cmdq_);
        free(rx_cmdq_);
        free(ctrl_cmdq_);
        free(ctrl_rspq_);
    }

    jring_t *tx_cmdq_;
    jring_t *rx_cmdq_;
    // Endpoint --> Engine
    jring_t *ctrl_cmdq_;
    // Engine --> Endpoint
    jring_t *ctrl_rspq_;
};

/**
 * Uccl SACK Packet Header.
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

class UcclFlow;
class UcclRDMAEngine;
class RDMAEndpoint;

/**
 * @class UcclFlow, a connection between a local and a remote endpoint.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a bidirectional connection between two hosts, uniquely identified
 * by a TCP-negotiated `FlowID'.
 *
 * A flow is always associated with a single `Channel' object which serves as
 * the communication interface with the application to which the flow belongs.
 *
 * On normal operation, a flow is:
 *    - Receiving network packets from the NIC, which then converts to messages
 *      and enqueues to the `Channel', so that they reach the application.
 *    - Receiving messages from the application (via the `Channel'), which then
 *      converts to network packets and sends them out to the remote recipient.
 */
class UcclFlow {
    const static uint32_t kMaxReadyMsgbufs = MAX_UNACKED_PKTS;
    constexpr static int kMaxBatchCQ = 32;
    // 256-bit SACK bitmask => we can track up to 256 packets
    static constexpr std::size_t kReassemblyMaxSeqnoDistance = kSackBitmapSize;
   public:
    /**
     * @brief Construct a new Uccl Flow on RDMA
     * 
     * @param engine 
     * @param channel 
     * @param flow_id 
     * @param rdma_ctx 
     */
    UcclFlow(UcclRDMAEngine *engine, Channel *channel, FlowID flow_id, struct RDMAContext *rdma_ctx): 
        engine_(engine), channel_(channel), flow_id_(flow_id), rdma_ctx_(rdma_ctx) {
            for (int i = 0; i < kMaxBatchCQ; i++) {
                retr_wrs_[i].num_sge = 1;
                retr_wrs_[i].sg_list = nullptr;
                retr_wrs_[i].next = (i == kMaxBatchCQ - 1) ? nullptr : &retr_wrs_[i + 1];
            }

            for (int i = 0; i < kPostRQThreshold; i++) {
                imm_wrs_[i].num_sge = 0;
                imm_wrs_[i].sg_list = nullptr;
                imm_wrs_[i].next = (i == kPostRQThreshold - 1) ? nullptr : &imm_wrs_[i + 1];

                rx_ack_sges_[i].lkey = rdma_ctx_->ctrl_chunk_pool_->get_lkey();
                rx_ack_sges_[i].length = CtrlChunkBuffPool::kChunkSize;
                rx_ack_wrs_[i].sg_list = &rx_ack_sges_[i];
                rx_ack_wrs_[i].num_sge = 1;
                rx_ack_wrs_[i].next = (i == kPostRQThreshold - 1) ? nullptr : &rx_ack_wrs_[i + 1];
            }

            tx_ack_wr_.num_sge = 1;
            tx_ack_wr_.next = nullptr;
            tx_ack_wr_.opcode = IBV_WR_SEND_WITH_IMM;
            tx_ack_wr_.send_flags = IBV_SEND_SIGNALED;
        };

    ~UcclFlow() {}

    friend class UcclRDMAEngine;

    inline void shutdown() { 
        for (int i = 0; i < kPortEntropy; i++) {
            rdma_ctx_->uc_qps_[i].pcb.rto_disable(); 
        }
    }

    /**
     * @brief Supply a buffer for the flow to receive data into.
     * @param rx_work 
     */
    void app_supply_rx_buf(Channel::Msg &rx_work);

    /**
     * @brief Flush the receive buffer.
     * @param rx_work 
     */
    void flush_rx_buf(Channel::Msg &rx_work);

    /**
     * @brief Transmit a message described by the tx_work.
     * @param tx_work 
     * @return Return true if this tx_work has been successfully transmitted.
     */
    bool tx_messages(Channel::Msg &tx_work);

    /**
     * @brief Receive a chunk from the flow.
     * @param ack_list 
     */
    void rx_chunk(struct list_head *ack_list);

    /**
     * @brief Receive a retransmitted chunk from the flow.
     * 
     * @param ack_list 
     */
    void rx_retr_chunk(struct list_head *ack_list);

    /**
     * @brief Receive a barrier from the flow.
     * @param ack_list
     */
    void rx_barrier(struct list_head *ack_list);

    /**
     * @brief Poll the completion queue for the FIFO QP.
     * @return Return true if polling is done for this flow, Engine should remove it from the polling list.
     */
    bool poll_fifo_cq(void);

    /**
     * @brief Poll the completion queues for all UC QPs.
     */
    inline int poll_uc_cq(void) { return rdma_ctx_->is_send_ ? sender_poll_uc_cq() : receiver_poll_uc_cq(); }
    int sender_poll_uc_cq(void);
    int receiver_poll_uc_cq(void);

    /**
     * @brief Poll the completion queue for the Ctrl QP.
     */
    inline int poll_ctrl_cq(void) { return rdma_ctx_->is_send_ ? sender_poll_ctrl_cq() : receiver_poll_ctrl_cq(); }
    int sender_poll_ctrl_cq(void);
    int receiver_poll_ctrl_cq(void);

    /**
     * @brief Poll the completion queue for the Retr QP.
     */
    inline int poll_retr_cq(void) { return rdma_ctx_->is_send_ ? sender_poll_retr_cq() : receiver_poll_retr_cq(); }
    int sender_poll_retr_cq(void);
    int receiver_poll_retr_cq(void);

    /**
     * @brief Only used for testing RC.
     */
    void test_rc_poll_cq(void);

    /**
     * @brief Retransmit a chunk for the given UC QP.
     * @param qpw 
     * @param wr_ex 
     */
    void retransmit_chunk(struct UCQPWrapper *qpw, struct wr_ex *wr_ex);

    /**
     * @brief Check if we need to post enough recv WQEs to the SRQ.
     */
    void check_srq(bool force = false);

    /**
     * @brief Check if we need to post enough recv WQEs to the Ctrl QP.
     * @param force 
     */
     inline void check_ctrl_rq(bool force = false) { rdma_ctx_->is_send_ ? sender_check_ctrl_rq(force) : receiver_check_ctrl_rq(); }
     void sender_check_ctrl_rq(bool force = false);
     void receiver_check_ctrl_rq(void);

    /**
     * @brief Rceive an ACK from the Ctrl QP.
     * 
     * @param pkt_addr
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

    void burst_timing_wheel(void);

    /**
     * @brief Try to update the CSN for the given UC QP.
     * @param qpw 
     */
    void try_update_csn(struct UCQPWrapper *qpw);

    /**
     * @brief The receiver is ready, post multiple messages to NIC.
     * @param slot Slot in FIFO.
     */
    void post_multi_messages(int slot);
    
    /**
     * @brief Post a single message to NIC. If needed, it will be queued in the timing wheel.
     * @param req 
     * @param slot 
     * @param mid 
     */
    void post_single_message(struct FlowRequest *req, struct FifoItem &slot, uint32_t mid);

    /**
     * @brief Only used for testing RC.
     */
    void test_rc_post_multi_messages(int slot);

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

    std::string to_string();

   private:

    void __retransmit(struct UCQPWrapper *qpw, bool rto);
    inline void fast_retransmit(struct UCQPWrapper *qpw) { __retransmit(qpw, false); }
    inline void rto_retransmit(struct UCQPWrapper *qpw) { __retransmit(qpw, true); }

    // <Slot, i>
    std::deque<std::pair<int, int> > pending_tx_msgs_;

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

    /**
     * @brief Post multiple recv requests to a FIFO queue for remote peer to use RDMA WRITE.
     * These requests are transmitted through the underlyding fifo QP (RC).
     * @param req Pointer to the FlowRequest structure.
     * @param data Array of data buffers.
     * @param size Array of buffer sizes.
     * @param n Number of buffers.
     * @param mr Memory region for the buffers.
     */
    void post_fifo(struct FlowRequest *req, void **data, size_t *size, int n, struct ibv_mr **mr);

    // Which Engine this flow belongs to.
    UcclRDMAEngine *engine_;

    // Context for RDMA resources.
    RDMAContext *rdma_ctx_;

    // The channel this flow belongs to.
    Channel *channel_;
    // FlowID of this flow.
    FlowID flow_id_;

    // Measure the distribution of probed RTT.
    Latency rtt_stats_;
    uint64_t rtt_probe_count_ = 0;

    friend class UcclRDMAEngine;
    friend class RDMAEndpoint;
};

/**
 * @brief Class `UcclRDMAEngine' abstracts the main Uccl engine which supports RDMA. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class UcclRDMAEngine {
   public:
    // Slow timer (periodic processing) interval in microseconds.
    const size_t kSlowTimerIntervalUs = 4000;  // 4ms

    UcclRDMAEngine() = delete;
    UcclRDMAEngine(UcclRDMAEngine const &) = delete;

    /**
     * @brief Construct a new UcclRDMAEngine object.
     * @param dev           Device index.
     * @param engine_id     Engine index.
     * @param channel       Uccl channel the engine will be responsible for.
     * For now, we assume an engine is responsible for a single channel, but
     * future it may be responsible for multiple channels.
     */
    UcclRDMAEngine(int dev, int engine_id, Channel *channel)
        : engine_idx_(engine_id),
          dev_(dev),
          channel_(channel),
          last_periodic_tsc_(rdtsc()),
          last_sync_clock_tsc_(rdtsc()),
          periodic_ticks_(0),
          kSlowTimerIntervalTsc_(us_to_cycles(kSlowTimerIntervalUs, freq_ghz)) {
            auto context = RDMAFactory::get_factory_dev(dev_)->context;
            struct ibv_values_ex values;
            values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
            ibv_query_rt_values_ex(context, &values);
            auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;
            last_nic_clock_ = nic_clock;
            last_host_clock_ = rdtsc();
    }

    /**
     * @brief Handling async recv requests from Endpoint for all flows.
     */
    void handle_rx_work(void);

    /**
     * @brief Handling aysnc send requests from Endpoint for all flows.
     */
    void handle_tx_work(void);

    /**
     * @brief Handling all completion events for all flows, including:
     * High-priority completion events from Ctrl QPs.
     * Datapath completion events from UC QPs.
     * Occasinal completion events from FIFO CQs.
     */
    void handle_completion(void);

    /**
     * @brief Only used for testing RC.
     */
    void test_rc_handle_completion(void);

    void handle_timing_wheel(void);

    /**
     * @brief Add a flow to the list for polling FIFO CQs in future.
     * @param flow 
     */
    inline void add_fifo_cq_polling(UcclFlow *flow) {
        fifo_cq_list_.push_back(flow);
    }

    /**
     * @brief This is the main event cycle of the Uccl engine.
     * It is called by a separate thread running the Uccl engine.
     * On each iteration, the engine processes incoming packets in the RX
     * queue and enqueued messages in all channels that it is responsible
     * for. This method is not thread-safe.
     */
    void run();

    /**
     * @brief Method to perform periodic processing. This is called by the
     * main engine cycle (see method `Run`).
     */
    void periodic_process();

    /**
     * @brief Creating underlying QPs, MRs, PDs, and CQs for the flow and set 
     * QP state to INIT.
     * @param ctrl_work 
     */
    void handle_install_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work);
    
    /**
     * @brief Modifying QP state to RTR and RTS. 
     * @param ctrl_work 
     */
    void handle_sync_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work);

    /**
     * @brief Registering a memory region.
     * @param ctrl_work 
     */
    void handle_regmr_on_engine_rdma(Channel::CtrlMsg &ctrl_work);

    /**
     * @brief Registering a memory region with DMA-BUF support.
     * @param ctrl_work 
     */
    void handle_regmr_dmabuf_on_engine_rdma(Channel::CtrlMsg &ctrl_work);

    /**
     * @brief Deregistering a memory region.
     * @param ctrl_work 
     */
    void handle_deregmr_on_engine_rdma(Channel::CtrlMsg &ctrl_work);

    inline bool need_sync(uint64_t now) {
        return now - last_sync_clock_tsc_ > ns_to_cycles(kSyncClockIntervalNS, freq_ghz);
    }

    inline void handle_clock_synchronization(void) {
        if constexpr (kTestRC) return;
        auto host_clock = rdtsc();
        if (need_sync(host_clock)) {
            auto context = RDMAFactory::get_factory_dev(dev_)->context;
            struct ibv_values_ex values;
            values.comp_mask = IBV_VALUES_MASK_RAW_CLOCK;
            ibv_query_rt_values_ex(context, &values);

            auto nic_clock = values.raw_clock.tv_sec * 1e9 + values.raw_clock.tv_nsec;

            // Update ratio and offset
            ratio_ = (1.0 * (int64_t)host_clock - (int64_t)last_host_clock_) / ((int64_t)nic_clock - (int64_t)last_nic_clock_);
            offset_ = host_clock - ratio_ * nic_clock;
            
            last_sync_clock_tsc_ = host_clock;
        }
    }

    // Convert NIC clock to host clock (TSC).
    inline uint64_t convert_nic_to_host(uint64_t host_clock, uint64_t nic_clock) {
        if (need_sync(host_clock)) handle_clock_synchronization();
        return ratio_ * nic_clock + offset_;
    }

    // Called by application to shutdown the engine. App will need to join
    // the engine thread.
    inline void shutdown() { shutdown_ = true; }

    std::string status_to_string();

   protected:

    /**
     * @brief Iterate throught the list of flows, check and handle RTOs.
     */
    void handle_rto();

    /**
     * @brief This method polls active channels for all control plane
     * requests and processes them. It is called periodically.
     */
    void process_ctl_reqs();

   private:
    // Device index
    int dev_;
    // Engine index
    uint32_t engine_idx_;
    // UcclFlow map
    std::unordered_map<FlowID, UcclFlow *> active_flows_map_;
    // Control plane channel with RDMAEndpoint.
    Channel *channel_;
    // FIFO CQs that need to be polled.
    std::list<UcclFlow *> fifo_cq_list_;
    // Pending tx work due to receiver not ready.
    std::deque<Channel::Msg> pending_tx_work_;
    // Timestamp of last periodic process execution.
    uint64_t last_periodic_tsc_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_;
    // Slow timer interval in TSC.
    uint64_t kSlowTimerIntervalTsc_;

    // Timestamp of last clock synchronization.
    uint64_t last_sync_clock_tsc_;
    uint64_t last_host_clock_;
    uint64_t last_nic_clock_;
    double ratio_ = 0;
    double offset_ = 0;

    // Whether shutdown is requested.
    std::atomic<bool> shutdown_{false};
};

// We only support RoCEv2 for now. Lid is not used.
struct RDMADevice {
    struct ibv_context *context;
    // Device name.
    char ib_name[64];
    // We only support one port per device, this should always be 1.
    uint8_t ib_port_num;
    // GID index.
    uint8_t gid_idx;
    // GID.
    union ibv_gid gid;
    // Local IP address.
    std::string local_ip_str;
};

/**
 * @class RDMAEndpoint
 * @brief application-facing interface, communicating with `UcclRDMAEngine' through
 * `Channel'. Each connection is identified by a unique flow_id, and uses
 * multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues. Note that all IB devices are managed by a single RDMAEndpoint.
 */
class RDMAEndpoint {
    constexpr static uint32_t kMaxInflightMsg = 1024 * 256;
    constexpr static uint16_t kBootstrapPort = 30000;
    constexpr static uint32_t kStatsTimerIntervalSec = 2;

    // The first CPU to run the engine thread belongs to the RDMAEndpoint.
    // The range of CPUs for one device to run engine threads is 
    // [engine_cpu_start_ + i*dev, engine_cpu_start_ + i*dev + num_engines_per_dev_). 
    int engine_cpu_start_;

    // RDMA devices.
    int num_devices_;
    struct RDMADevice rdma_dev_list_[MAX_IB_DEVICES];

    int num_engines_per_dev_;
    // Per-engine communication channel
    Channel *channel_vec_[NUM_ENGINES * MAX_IB_DEVICES];
    std::vector<std::unique_ptr<UcclRDMAEngine>> engine_vec_;
    std::vector<std::unique_ptr<std::thread>> engine_th_vec_;

    // Number of flows on each engine, indexed by engine_idx.
    std::mutex engine_load_vec_mu_;
    std::array<int, NUM_ENGINES> engine_load_vec_ = {0};

    SharedPool<PollCtx *, true> *ctx_pool_;
    uint8_t *ctx_pool_buf_;

    // int listen_fd_;
    int listen_fds_[NUM_DEVICES];

    std::mutex fd_map_mu_;
    // Mapping from unique (within this engine) flow_id to the boostrap fd.
    std::unordered_map<FlowID, int> fd_map_;

   public:
    RDMAEndpoint(const uint8_t *gid_idx_list, int num_devices, int num_engines_per_dev, int engine_cpu_start);
    ~RDMAEndpoint();

    // Connect to a remote address; thread-safe
    ConnID uccl_connect(int dev, std::string remote_ip);
    // Explicitly specify the engine_id to install the flow.
    ConnID uccl_connect(int dev, int engine_id, std::string remote_ip);
    // Accept a connection from a remote address; thread-safe
    ConnID uccl_accept(int dev, std::string &remote_ip);
    // Explicitly specify the engine_id to install the flow.
    ConnID uccl_accept(int dev, int engine_id, std::string &remote_ip);
    
    // Register a memory region.
    int uccl_regmr(ConnID flow_id, void *data, size_t len, int type, struct Mhandle **mhandle);
    // Register a DMA-BUF memory region.
    int uccl_regmr_dmabuf(ConnID flow_id, void *data, size_t len, int type, int offset, int fd, struct Mhandle **mhandle);
    // Deregister a memory region.
    void uccl_deregmr(ConnID flow_id, struct Mhandle *mhandle);

    // Post a buffer to engine for sending data asynchronously.
    PollCtx *uccl_send_async(ConnID flow_id, struct Mhandle *mhandle, const void *data,
                             const size_t size);
    // Post n buffers to engine for receiving data asynchronously.
    PollCtx *uccl_recv_async(ConnID flow_id, struct Mhandle **mhandles, void **data, int *size, int n);

    // Post a buffer to engine for sending data asynchronously.
    PollCtx *uccl_send_async(ConnID flow_id, struct Mhandle *mhandle, const void *data,
                             const size_t size, struct ucclRequest *ureq);
    // Post n buffers to engine for receiving data asynchronously.
    PollCtx *uccl_recv_async(ConnID flow_id, struct Mhandle **mhandles, void **data, int *size, int n, 
                            struct ucclRequest *ureq);

    // Ensure that all received data is visible to GPU.
    PollCtx *uccl_flush(ConnID flow_id, struct Mhandle **mhandles, void **data, int *size, int n);

    bool uccl_wait(PollCtx *ctx);
    bool uccl_poll(PollCtx *ctx);
    bool uccl_poll_once(PollCtx *ctx);

   private:

    void install_flow_on_engine_rdma(int dev, FlowID flow_id,
                                     uint32_t local_engine_idx, int bootstrap_fd, bool is_send);
    inline void put_load_on_engine(int engine_id);
    inline int find_least_loaded_engine_idx_and_update();
    inline void fence_and_clean_ctx(PollCtx *ctx);

    inline int receive_message(int sockfd, void *buffer, size_t n_bytes) {
        int bytes_read = 0;
        int r;
        while (bytes_read < n_bytes) {
            // Make sure we read exactly n_bytes
            r = read(sockfd, buffer + bytes_read, n_bytes - bytes_read);
            if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
                CHECK(false) << "ERROR reading from socket";
            }
            if (r > 0) {
                bytes_read += r;
            }
        }
        return bytes_read;
    }

    inline int send_message(int sockfd, const void *buffer, size_t n_bytes) {
        int bytes_sent = 0;
        int r;
        while (bytes_sent < n_bytes) {
            // Make sure we write exactly n_bytes
            r = write(sockfd, buffer + bytes_sent, n_bytes - bytes_sent);
            if (r < 0 && !(errno == EAGAIN || errno == EWOULDBLOCK)) {
                CHECK(false) << "ERROR writing to socket";
            }
            if (r > 0) {
                bytes_sent += r;
            }
        }
        return bytes_sent;
    }

    inline void net_barrier(int bootstrap_fd) {
        bool sync = true;
        int ret = send_message(bootstrap_fd, &sync, sizeof(bool));
        ret = receive_message(bootstrap_fd, &sync, sizeof(bool));
        DCHECK(ret == sizeof(bool) && sync);
    }

    std::thread stats_thread_;
    void stats_thread_fn();
    std::mutex stats_mu_;
    std::condition_variable stats_cv_;
    std::atomic<bool> shutdown_{false};

    friend class UcclFlow;
};

}  // namespace uccl
