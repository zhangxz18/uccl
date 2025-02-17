#pragma once

#include <arpa/inet.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <inttypes.h>
#include <linux/if_ether.h>
#include <linux/if_link.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <memory.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <bitset>
#include <deque>
#include <memory>
#include <mutex>
#include <set>

#include "transport_config.h"
#include "transport_header.h"
#include "util.h"
#include "util_list.h"
#include "util_timer.h"

namespace uccl {

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
    BuffPool(uint32_t num_elements, size_t element_size,
             struct ibv_mr *mr = nullptr,
             void (*init_cb)(uint64_t buff) = nullptr)
        : num_elements_(num_elements), element_size_(element_size), mr_(mr) {
        if (mr_) {
            base_addr_ = mr->addr;
        } else {
            base_addr_ = mmap(nullptr, num_elements_ * element_size_,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base_addr_ == MAP_FAILED)
                throw std::runtime_error(
                    "Failed to allocate memory for BuffPool.");
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

    inline bool empty(void) { return head_ == tail_; }

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

class PktHdrBuffPool : public BuffPool {
   public:
    // This should cover EFA_GRH_SIZE + data packet hdr
    // or EFA_GRH_SIZE + ack packet hdr + sack packet hdr.
    static constexpr uint32_t kPktHdrSize = 256;
    static constexpr uint32_t kNumPktHdr = NUM_FRAMES;
    static_assert((kNumPktHdr & (kNumPktHdr - 1)) == 0,
                  "kNumPktHdr must be power of 2");

    PktHdrBuffPool(struct ibv_mr *mr) : BuffPool(kNumPktHdr, kPktHdrSize, mr) {}

    ~PktHdrBuffPool() = default;
};

class PktDataBuffPool : public BuffPool {
   public:
    static constexpr uint32_t kPktDataSize = EFA_MTU;
    static constexpr uint32_t kNumPktData = NUM_FRAMES;
    static_assert((kNumPktData & (kNumPktData - 1)) == 0,
                  "kNumPkt must be power of 2");

    PktDataBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumPktData, kPktDataSize, mr) {}

    ~PktDataBuffPool() = default;
};

class FrameDesc {
    // Describing the packet frame address and length.
    uint64_t pkt_hdr_addr_;   // in CPU memory.
    uint64_t pkt_data_addr_;  // in GPU memory.
    uint32_t pkt_hdr_len_;    // the length of packet hdr.
    uint32_t pkt_data_len_;   // the length of packet data.
    uint16_t dest_qpn_;       // dest QP to use for this frame.
    uint8_t dest_gid_idx_;    // dest gid to use for this frame.

    // Flags to denote the message buffer state.
    static const uint8_t UCCL_MSGBUF_FLAGS_SYN = (1 << 0);
    static const uint8_t UCCL_MSGBUF_FLAGS_FIN = (1 << 1);
    static const uint8_t UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE = (1 << 2);
    uint8_t msg_flags_;

    struct list_head frame_link_;

    FrameDesc(uint64_t pkt_hdr_addr, uint64_t pkt_data_addr,
              uint32_t pkt_hdr_len, uint32_t pkt_data_len, uint8_t msg_flags_)
        : pkt_hdr_addr_(pkt_hdr_addr),
          pkt_data_addr_(pkt_data_addr),
          pkt_hdr_len_(pkt_hdr_len),
          pkt_data_len_(pkt_data_len),
          msg_flags_(msg_flags) {
        INIT_LIST_HEAD(&frame_link_);
        dest_qpn_ = UINT16_MAX;
        dest_gid_idx_ = UINT8_MAX;
    }

   public:
    static FrameDesc *Create(uint64_t frame_desc_addr, uint64_t pkt_hdr_addr,
                             uint64_t pkt_data_addr, uint32_t pkt_hdr_len,
                             uint32_t pkt_data_len, uint8_t msg_flags_ = 0) {
        return new (reinterpret_cast<void *>(frame_desc_addr)
            FrameDesc(pkt_hdr_addr, pkt_data_addr, pkt_hdr_len, pkt_data_len, msg_flags);
    }
    uint32_t get_pkt_hdr_addr() const { return pkt_hdr_addr_; }
    uint64_t get_pkt_data_addr() const { return pkt_data_addr_; }
    uint8_t get_pkt_hdr_len() const { return pkt_hdr_len_; }
    uint8_t get_pkt_data_len() const { return pkt_data_len_; }

    uint16_t get_dest_qpn() const { return dest_qpn_; }
    void set_dest_qpn(uint16_t dest_qpn) const { dest_qpn_ = dest_qpn; }

    uint8_t get_dest_gid_idx() const { return dest_gid_idx_; }
    void set_dest_gid_idx(uint8_t dest_gid_idx) const {
        dest_gid_idx_ = dest_gid_idx;
    }

    uint16_t msg_flags() const { return msg_flags_; }
    void set_msg_flags(uint16_t flags) { msg_flags_ = flags; }
    void add_msg_flags(uint16_t flags) { msg_flags_ |= flags; }

    // Returns true if this is the first in a message.
    bool is_first() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_SYN) != 0; }
    // Returns true if this is the last in a message.
    bool is_last() const { return (msg_flags_ & UCCL_MSGBUF_FLAGS_FIN) != 0; }

    // Returns the next message buffer index in the chain.
    FrameDesc *next() const {
        return reinterpret_cast<FrameDesc *>(frame_link_.next);
    }
    // Set the next message buffer index in the chain.
    void set_next(FrameDesc *next) {
        list_add(&next->frame_link_, &frame_link_);
    }

    void mark_first() { add_msg_flags(UCCL_MSGBUF_FLAGS_SYN); }
    void mark_last() { add_msg_flags(UCCL_MSGBUF_FLAGS_FIN); }

    void mark_txpulltime_free() {
        add_msg_flags(UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE);
    }
    void mark_not_txpulltime_free() {
        msg_flags_ &= ~UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE;
    }
    bool is_txpulltime_free() {
        return (msg_flags_ & UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE) != 0;
    }

    void clear_fields() {
        pkt_hdr_addr_ = 0;
        pkt_data_addr_ = 0;
        pkt_hdr_len_ = 0;
        pkt_data_len_ = 0;
        msg_flags_ = 0;
        dest_qpn_ = UINT16_MAX;
        dest_gid_idx_ = UINT8_MAX;
        INIT_LIST_HEAD(&frame_link_);
    }

    std::string to_string() {
        std::stringstream s;
        s << "pkt_hdr_addr: 0x" << std::hex << pkt_hdr_addr_
          << " pkt_data_addr: 0x" << std::hex << pkt_data_addr_
          << " pkt_hdr_len: " << std::dec << pkt_hdr_len_
          << " pkt_data_len: " << std::dec << pkt_data_len_
          << " dest_qpn: " << std::dec << dest_qpn_
          << " dest_gid_idx: " << std::dec << dest_gid_idx_
          << " msg_flags: " << std::dec << std::bitset<8>(msg_flags_);
        return s.str();
    }

    std::string print_chain() {
        std::stringstream s;
        struct list_head *pos, *n;
        list_for_each_safe(pos, n, &frame_link_) {
            auto *cur_desc = list_entry(pos, struct FrameDesc, frame_link_);
            s << cur_desc->to_string() << "\n";
        }
        return s.str();
    }
};

class FrameDescBuffPool : public BuffPool {
   public:
    static constexpr uint32_t kFrameDescSize = sizeof(FrameDesc);
    static constexpr uint32_t kNumFrameDesc = NUM_FRAMES;
    static_assert((kNumFrameDesc & (kNumFrameDesc - 1)) == 0,
                  "kNumFrameDesc must be power of 2");

    FrameDescBuffPool() : BuffPool(kNumFrameDesc, kFrameDescSize, nullptr) {}

    ~FrameDescBuffPool() = default;
};

struct EFADevice {
    char ib_name[64];
    std::string local_ip_str;

    struct ibv_context *context;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;

    uint8_t EFA_PORT_NUM;
    uint8_t gid_idx;
    union ibv_gid gid;

    struct ibv_pd *pd;

    // DMA-BUF support
    bool dma_buf_support;
};

class EFASocket;
class EFAFactory;
extern EFAFactory efa_ctl;

class EFAFactory {
   public:
    std::vector<struct EFADevice> devices_;

    // int gid_idx --> int dev
    std::unordered_map<int, int> gid_2_dev_map;

    std::mutex socket_q_lock_;
    std::deque<EFASocket *> socket_q_;

    // Not thread-safe; should be called just once.
    static void Init();
    // gid_idx from GID_INDEX_LIST;
    // socket_id from [0, ..., kNumEnginesPerDev-1].
    static EFASocket *CreateSocket(int gid_idx, int socket_id);
    // Getting a pointer to the struct EFADevice.
    static struct EFADevice *GetEFADevice(int gid_idx);

    static void Shutdown();

   private:
    static void InitDev(int gid_idx);
};

class EFASocket {
    struct ibv_context *context_;
    struct ibv_pd *pd_;

    struct ibv_cq *send_cq_;
    struct ibv_cq *recv_cq_;
    struct ibv_qp *qp_list_[kMaxPath];

    struct ibv_cq *ctrl_cq_;
    struct ibv_qp *ctrl_qp_;

    // For fast CQ polling.
    struct ibv_wc wc_[kMaxBatchCQ];

    // remote_gid_idx -> struct ibv_ah*.
    struct ibv_ah *ah_list_[255];

    EFASocket(int gid_idx, int socket_id);

    struct ibv_qp *create_qp(struct ibv_cq *send_cq, struct ibv_cq *recv_cq,
                             uint32_t send_cq_size, uint32_t recv_cq_size);

    // !!! User must call this to create ah before sending packets.
    void create_ah(uint8_t remote_gid_idx, union ibv_gid remote_gid);

    uint32_t next_qp_idx_for_send_;
    inline uint32_t get_next_qp_idx_for_send() {
        next_qp_idx_for_send_ = (next_qp_idx_for_send_++) % kMaxQPForSend;
        return next_qp_idx_for_send_;
    }

   public:
    uint32_t gid_idx_;
    uint32_t socket_id_;

    PktHdrBuffPool *pkt_hdr_pool_;
    PktDataBuffPool *pkt_data_pool_;
    FrameDescBuffPool *frame_desc_pool_;

    inline uint32_t gid_idx() const { return gid_idx_; }
    inline uint32_t socket_id() const { return socket_id_; }

    // dest_qpn and dest_gid_idx is specified in the FrameDesc; src_qp is
    // determined by EFASocket internally to evenly spread the load. This
    // function also dynamically chooses normal qp and ctrl_qp based on the size
    // of pkt_hdr_len_ and pkt_data_len_.
    uint32_t send_packet(FrameDesc *frame);
    uint32_t send_packets(std::vector<FrameDesc *> &frames);
    // This polls send_cq_;
    std::vector<FrameDesc *> poll_send_cq(uint32_t bugget);

    // We embedded FrameDesc* into wr_id, so we can retrieve it later.
    // This polls recv_cq_;
    std::vector<FrameDesc *> recv_packets(uint32_t budget);

    // This will internally free FrameDesc used to send acks.
    std::vector<FrameDesc *> recv_acks_and_poll_ctrl_cq(uint32_t budget);

    void refill_recv_queue_data(uint32_t budget, uint32_t qp_idx);
    void refill_recv_queue_ctrl(uint32_t budget);

    std::string to_string();
    void shutdown();
    ~EFASocket();

    inline uint32_t send_queue_free_entries() {
        return kMaxSendWr - unpolled_send_wrs_;
    }
    inline uint64_t send_queue_estimated_latency_ns() {
        return unpolled_send_wrs_ * EFA_MTU * 1000000000UL / kLinkBandwidth;
    }

    inline uint64_t pop_pkt_hdr() {
        uint64_t pkt_hdr_addr;
        auto ret = pkt_hdr_pool_->alloc_buff(&pkt_hdr_addr);
        DCHECK(ret == 0);
        return pkt_hdr_addr;
    }
    inline void push_pkt_hdr(uint64_t pkt_hdr_addr) {
        pkt_hdr_pool_->free_buff(pkt_hdr_addr);
    }
    inline void get_pkt_hdr_lkey() { return pkt_hdr_pool_->get_lkey(); }

    inline uint64_t pop_pkt_data() {
        uint64_t pkt_data_addr;
        auto ret = pkt_data_pool_->alloc_buff(&pkt_data_addr);
        DCHECK(ret == 0);
        return pkt_data_addr;
    }
    inline void push_pkt_data(uint64_t pkt_data_addr) {
        pkt_data_pool_->free_buff(pkt_data_addr);
    }
    inline void get_pkt_data_lkey() { return pkt_data_pool_->get_lkey(); }

    inline uint64_t pop_frame_desc() {
        uint64_t pkt_frame_desc;
        auto ret = frame_desc_pool_->alloc_buff(&pkt_frame_desc);
        DCHECK(ret == 0);
        return pkt_frame_desc;
    }
    inline void push_frame_desc(uint64_t pkt_frame_desc) {
        frame_desc_pool_->free_buff(pkt_frame_desc);
    }

   private:
    uint32_t unpolled_send_wrs_;
    uint32_t recv_queue_wrs_;

    std::chrono::time_point<std::chrono::high_resolution_clock> last_stat_;
    inline static std::atomic<uint64_t> out_packets_ = 0;
    inline static std::atomic<uint64_t> out_bytes_ = 0;
    inline static std::atomic<uint64_t> in_packets_ = 0;
    inline static std::atomic<uint64_t> in_bytes_ = 0;

    friend class EFAFactory;
};

/**
 * @brief This helper function gets the Infiniband name from the GID index.
 *
 * @param gid_idx
 * @param ib_name
 * @return int
 */
static inline int util_efa_get_ib_name_from_gididx(int gid_idx, char *ib_name) {
    sprintf(ib_name, "%s", EFA_DEVICE_NAME_LIST[gid_idx].c_str());
    return 0;
}

/**
 * @brief This helper function gets the IP address of the device from gid_index.
 *
 * @param gid_idx
 * @return int
 */
static inline int util_efa_get_ip_from_gididx(int gid_idx, std::string *ip) {
    *ip = get_dev_ip(ENA_DEVICE_NAME_LIST[gid_idx].c_str());
    return *ip == "" ? -1 : 0;
}

}  // namespace uccl