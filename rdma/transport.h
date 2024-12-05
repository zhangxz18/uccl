#pragma once

#include <glog/logging.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    FlowID flow_id;       // Used for UcclEngine to look up UcclFlow.
    uint32_t engine_idx;  // Used for Endpoint to locate the right engine.
    int boostrap_id;      // Used for bootstrap connection with the peer.
};

struct alignas(64) PollCtx {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> fence;  // Sync rx/tx memcpy visibility.
    std::atomic<bool> done;   // Sync cv wake-up.
    uint64_t timestamp;       // Timestamp for request issuing.
    PollCtx() : fence(false), done(false), timestamp(0) {};
    ~PollCtx() { clear(); }
    void clear() {
        mu.~mutex();
        cv.~condition_variable();
        fence = false;
        done = false;
        timestamp = 0;
    }
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
            kTx = 0,
            kTxComp = 1,
            kRx = 2,
            kRxComp = 3,
        };
        Op opcode;
        FlowID flow_id;
        void *data;
        size_t len;
        size_t *len_p;
        // Wakeup handler
        PollCtx *poll_ctx;
    };
    static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

    struct CtrlMsg {
        enum Op : uint8_t {
            kInstallFlow = 0,
        };
        Op opcode;
        FlowID flow_id;
        uint32_t remote_ip;
        char remote_mac[ETH_ALEN];
        char padding[2];
        uint32_t remote_engine_idx;
        // Wakeup handler
        PollCtx *poll_ctx;
    };
    static_assert(sizeof(CtrlMsg) % 4 == 0,
                  "channelMsg must be 32-bit aligned");

    Channel() {
        tx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
        rx_cmdq_ = create_ring(sizeof(Msg), kChannelSize);
        ctrl_cmdq_ = create_ring(sizeof(CtrlMsg), kChannelSize);
    }

    ~Channel() {
        free(tx_cmdq_);
        free(rx_cmdq_);
        free(ctrl_cmdq_);
    }

    jring_t *tx_cmdq_;
    jring_t *rx_cmdq_;
    jring_t *ctrl_cmdq_;
};

class UcclFlow;
class UcclEngine;
class Endpoint;

/**
 * @class UcclFlow, a connection between a local and a remote endpoint.
 * @brief Class to abstract the components and functionality of a single flow.
 * A flow is a bidirectional connection between two hosts, uniquely identified
 * by a TCP-negotiated `FlowID', Protocol is always UDP.
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

   public:
    /**
     * @brief Construct a new flow running on AFXDP socket.
     *
     * @param local_addr Local IP address.
     * @param remote_addr Remote IP address.
     * @param local_l2_addr Local L2 address.
     * @param remote_l2_addr Remote L2 address.
     * @param FlowID Connection ID for the flow.
     */
    UcclFlow(const uint32_t local_addr, const uint32_t remote_addr,
             const char local_l2_addr[ETH_ALEN],
             const char remote_l2_addr[ETH_ALEN], uint32_t local_engine_idx,
             uint32_t remote_engine_idx, Channel *channel,
             FlowID flow_id)
        : local_addr_(local_addr),
          remote_addr_(remote_addr),
          local_engine_idx_(local_engine_idx),
          remote_engine_idx_(remote_engine_idx),
          channel_(channel),
          flow_id_(flow_id),
          pcb_() {
        memcpy(local_l2_addr_, local_l2_addr, ETH_ALEN);
        memcpy(remote_l2_addr_, remote_l2_addr, ETH_ALEN);
    }
    ~UcclFlow() {}

    friend class UcclEngine;

    std::string to_string() const;
    inline void shutdown() { pcb_.rto_disable(); }

    /**
     * @brief Push the received packet onto the ingress queue of the flow.
     * Decrypts packet if required, stores the payload in the relevant channel
     * shared memory space, and if the message is ready for delivery notifies
     * the application.
     *
     * If this is a transport control packet (e.g., ACK) it only updates
     * transport-related parameters for the flow.
     */
    void rx_messages();

    void rx_supply_app_buf(Channel::Msg &rx_work);

    /**
     * @brief Push a Message from the application onto the egress queue of
     * the flow. Segments the message, and encrypts the packets, and adds
     * all packets onto the egress queue. Caller is responsible for freeing
     * the MsgBuf object.
     *
     * @param msg Pointer to the first message buffer on a train of buffers,
     * aggregating to a partial or a full Message.
     */
    void tx_messages(Channel::Msg &tx_work);

    void process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                              uint64_t ts4);

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

   private:

    void fast_retransmit();
    void rto_retransmit();

    /**
     * @brief Helper function to transmit a number of packets from the queue
     * of pending TX data.
     */
    void transmit_pending_packets();

    struct pending_tx_msg_t {
        Channel::Msg tx_work;
        size_t cur_offset = 0;
    };

    std::deque<pending_tx_msg_t> pending_tx_msgs_;

    /**
     * @brief Deserialize a chunk of data from the application buffer and append
     * to the tx tracking.
     */
    void deserialize_and_append_to_txtracking();

    void prepare_l2header(uint8_t *pkt_addr) const;
    void prepare_l3header(uint8_t *pkt_addr, uint32_t payload_bytes) const;
    void prepare_l4header(uint8_t *pkt_addr, uint32_t payload_bytes,
                          uint16_t dst_port) const;

    inline uint16_t get_next_dst_port() {
        return dst_ports_[next_port_idx_++ % kPortEntropy];
    }

    // The following is used to fill packet headers.
    uint32_t local_addr_;
    uint32_t remote_addr_;
    char local_l2_addr_[ETH_ALEN];
    char remote_l2_addr_[ETH_ALEN];
    // Which engine (also NIC queue and xsk) this flow belongs to.
    uint32_t local_engine_idx_;
    uint32_t remote_engine_idx_;

    // The channel this flow belongs to.
    Channel *channel_;
    // FlowID of this flow.
    FlowID flow_id_;
    // Destination ports with remote_engine_idx_ as the target queue_id.
    std::vector<uint16_t> dst_ports_;
    // Index in dst_ports_ for the next port to use.
    uint32_t next_port_idx_ = 0;

    // Swift CC protocol control block.
    swift::Pcb pcb_;
    // Measure the distribution of probed RTT.
    Latency rtt_stats_;
    uint64_t rtt_probe_count_ = 0;

    friend class UcclEngine;
    friend class Endpoint;
};

/**
 * @brief Class `UcclEngine' abstracts the main Uccl engine. This engine
 * contains all the functionality need to be run by the stack's threads.
 */
class UcclEngine {
   public:
    // Slow timer (periodic processing) interval in microseconds.
    const size_t kSlowTimerIntervalUs = 2000;  // 2ms
    UcclEngine() = delete;
    UcclEngine(UcclEngine const &) = delete;

    /**
     * @brief Construct a new UcclEngine object.
     *
     * @param queue_id      RX/TX queue index to be used by the engine.
     * @param channel       Uccl channel the engine will be responsible for.
     * For now, we assume an engine is responsible for a single channel, but
     * future it may be responsible for multiple channels.
     */
    UcclEngine(int queue_id, Channel *channel, const std::string local_addr,
               const std::string local_l2_addr)
        : local_addr_(htonl(str_to_ip(local_addr))),
          local_engine_idx_(queue_id),
          channel_(channel),
          last_periodic_tsc_(rdtsc()),
          periodic_ticks_(0),
          kSlowTimerIntervalTsc_(us_to_cycles(kSlowTimerIntervalUs, freq_ghz)) {
        DCHECK(str_to_mac(local_l2_addr, local_l2_addr_));

        rdma_ctx_ = RDMAFactory::CreateContext(queue_id);
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

    void handle_install_flow_on_engine(Channel::CtrlMsg &ctrl_work);

    void handle_install_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work);

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
    uint32_t local_addr_;
    char local_l2_addr_[ETH_ALEN];
    // Engine index, also NIC queue ID and xsk index.
    uint32_t local_engine_idx_;
    // RDMA context used for send/recv packets.
    RDMAContext *rdma_ctx_;
    // UcclFlow map
    std::unordered_map<FlowID, UcclFlow *> active_flows_map_;
    // Control plane channel with Endpoint.
    Channel *channel_;
    // Timestamp of last periodic process execution.
    uint64_t last_periodic_tsc_;
    // Clock ticks for the slow timer.
    uint64_t periodic_ticks_;
    // Slow timer interval in TSC.
    uint64_t kSlowTimerIntervalTsc_;
    // Whether shutdown is requested.
    std::atomic<bool> shutdown_{false};
};

/**
 * @class Endpoint
 * @brief application-facing interface, communicating with `UcclEngine' through
 * `Channel'. Each connection is identified by a unique flow_id, and uses
 * multiple src+dst port combinations to leverage multiple paths. Under the
 * hood, we leverage TCP to boostrap our connections. We do not consider
 * multi-tenancy for now, assuming this endpoint exclusively uses the NIC and
 * its all queues.
 */
class Endpoint {
    constexpr static uint32_t kMaxInflightMsg = 1024 * 256;
    constexpr static uint16_t kBootstrapPort = 30000;
    constexpr static uint32_t kStatsTimerIntervalSec = 2;

    // The first CPU to run the engine thread belongs to the Endpoint.
    // The range of CPUs to run the engine thread is [engine_cpu_start_, engine_cpu_start_ + num_queues_).
    int engine_cpu_start_;

    std::string local_ip_str_;
    std::string local_mac_str_;

    int num_queues_;
    Channel *channel_vec_[NUM_QUEUES];
    std::vector<std::unique_ptr<UcclEngine>> engine_vec_;
    std::vector<std::unique_ptr<std::thread>> engine_th_vec_;

    // Number of flows on each engine, indexed by engine_idx.
    std::mutex engine_load_vec_mu_;
    std::array<int, NUM_QUEUES> engine_load_vec_ = {0};

    SharedPool<PollCtx *, true> *ctx_pool_;
    uint8_t *ctx_pool_buf_;

    int listen_fd_;

    std::mutex bootstrap_fd_map_mu_;
    // Mapping from unique (within this engine) flow_id to the boostrap fd.
    std::unordered_map<FlowID, int> bootstrap_fd_map_;

   public:
    Endpoint(const char *interface_name, int num_engines, int engine_cpu_start);
    ~Endpoint();

    // Connecting to a remote address; thread-safe
    ConnID uccl_connect(std::string remote_ip);
    // Accepting a connection from a remote address; thread-safe
    ConnID uccl_accept(std::string &remote_ip);

    // Sending the data by leveraging multiple port combinations.
    bool uccl_send(ConnID flow_id, const void *data, const size_t len,
                   bool busypoll = false);
    // Receiving the data by leveraging multiple port combinations.
    bool uccl_recv(ConnID flow_id, void *data, size_t *len_p,
                   bool busypoll = false);

    // Sending the data by leveraging multiple port combinations.
    PollCtx *uccl_send_async(ConnID flow_id, const void *data,
                             const size_t len);
    // Receiving the data by leveraging multiple port combinations.
    PollCtx *uccl_recv_async(ConnID flow_id, void *data, size_t *len_p);

    bool uccl_wait(PollCtx *ctx);
    bool uccl_poll(PollCtx *ctx);
    bool uccl_poll_once(PollCtx *ctx);

   private:
    void install_flow_on_engine(FlowID flow_id, const std::string &remote_ip,
                                uint32_t local_engine_idx, int bootstrap_fd);
    void install_flow_on_engine_rdma(FlowID flow_id, const std::string &remote_ip,
                                     uint32_t local_engine_idx, int bootstrap_fd);
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
