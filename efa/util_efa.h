#pragma once

#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <glog/logging.h>
#include <ifaddrs.h>
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
#include "util_list.h"
#include "util_shared_pool.h"
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

class UDPktHdrBuffPool : public BuffPool {
   public:
    static constexpr uint32_t kPktHdrSize = 64 + UD_ADDITION;
    static constexpr uint32_t kNumPktHdr = NUM_FRAMES;
    static_assert((kNumPktHdr & (kNumPktHdr - 1)) == 0,
                  "kNumPktHdr must be power of 2");

    UDPktHdrBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumPktHdr, kPktHdrSize, mr) {}

    ~UDPktHdrBuffPool() = default;
};

class UDPktDataBuffPool : public BuffPool {
   public:
    static constexpr uint32_t kPktDataSize =
        EFA_MAX_PAYLOAD - UDPktHdrBuffPool::kPktHdrSize;
    static constexpr uint32_t kNumPktData = NUM_FRAMES;
    static_assert((kNumPktData & (kNumPktData - 1)) == 0,
                  "kNumPkt must be power of 2");

    UDPktDataBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumPktData, kPktDataSize, mr) {}

    ~UDPktDataBuffPool() = default;
};

class FrameDesc {
    // Describing the packet frame address and length.
    uint64_t pkt_data_addr_;  // in GPU memory.
    uint64_t pkt_hdr_addr_;   // in CPU memory.
    uint32_t pkt_data_len_;
    uint16_t dest_qp_;  // dest QP to use for this frame.

    // Flags to denote the message buffer state.
    static const uint8_t UCCL_MSGBUF_FLAGS_SYN = (1 << 0);
    static const uint8_t UCCL_MSGBUF_FLAGS_FIN = (1 << 1);
    static const uint8_t UCCL_MSGBUF_FLAGS_TXPULLTIME_FREE = (1 << 2);
    uint8_t msg_flags_;

    struct list_head frame_link_;

    FrameDesc(uint64_t pkt_data_addr, uint64_t pkt_hdr_addr,
              uint32_t pkt_data_len, uint8_t msg_flags_)
        : pkt_data_addr_(pkt_data_addr),
          pkt_hdr_addr_(pkt_hdr_addr),
          pkt_data_len_(pkt_data_len),
          msg_flags_(msg_flags) {
        INIT_LIST_HEAD(&frame_link_);
        dest_qp_ = UINT16_MAX;
    }

   public:
    static FrameDesc *Create(uint64_t frame_desc_addr, uint64_t pkt_data_addr,
                             uint64_t pkt_hdr_addr, uint32_t pkt_data_len,
                             uint8_t msg_flags_ = 0) {
        return new (reinterpret_cast<void *>(frame_desc_addr)
            FrameDesc(pkt_data_addr, pkt_hdr_addr, pkt_data_len, msg_flags);
    }
    uint64_t get_pkt_data_addr() const { return pkt_data_addr_; }
    uint32_t get_pkt_hdr_addr() const { return pkt_hdr_addr_; }
    uint8_t *get_pkt_data_len() const { return pkt_data_len_; }

    uint16_t get_dest_qp() const { return dest_qp_; }
    void set_dest_qp(uint16_t dest_qp) const { dest_qp_ = dest_qp; }

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
        next_ = nullptr;
        frame_offset_ = 0;
        umem_buffer_ = nullptr;
        frame_len_ = 0;
        msg_flags_ = 0;
    }

    std::string to_string() {
        std::stringstream s;
        s << "pkt_data_addr: 0x" << std::hex << pkt_data_addr_
          << " pkt_data_len: " << std::dec << pkt_data_len_
          << " pkt_hdr_addr: 0x" << std::hex << pkt_hdr_addr_
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

    FrameDescBuffPool(struct ibv_mr *mr)
        : BuffPool(kNumFrameDesc, kFrameDescSize, mr) {}

    ~FrameDescBuffPool() = default;
};

class EFASocket;
class EFAFactory;
extern EFAFactory efa_ctl;

class EFAFactory {
   public:
    std::mutex socket_q_lock_;
    std::deque<EFASocket *> socket_q_;

    static void Init();
    static EFASocket *CreateSocket(int socket_id);
    static void Shutdown();
};

class EFASocket {
    // socket_id starts from 0, equal to socket_id.
    EFASocket(int socket_id);

    void destroy_efa_socket();
    int create_efa_socket();

   public:
    uint32_t socket_id_;
    UDPktDataBuffPool *pkt_data_pool_;
    UDPktHdrBuffPool *pkt_hdr_pool_;
    FrameDescBuffPool *frame_desc_pool_;

    inline uint32_t socket_id() const { return socket_id_; }
    // dest_qp is specified in the FrameDesc; src_qp is determined by EFASocket
    // internally to evenly spread the load
    uint32_t send_packet(FrameDesc *frame);
    uint32_t send_packets(std::vector<FrameDesc *> &frames);
    std::vector<FrameDesc *> recv_packets(uint32_t budget);

    uint32_t pull_complete_queue();

    void populate_recv_queue(uint32_t budget);
    inline void populate_recv_queue_to_full() {
        populate_recv_queue(FILL_RING_SIZE - fill_queue_entries_);
    }

    inline uint32_t send_queue_free_entries(uint32_t nb = UINT32_MAX) {
        return xsk_prod_nb_free(&send_queue_, nb);
    }
    inline uint64_t send_queue_estimated_latency_ns() {
        auto send_queue_pending_entries =
            TX_RING_SIZE - send_queue_free_entries();
        return send_queue_pending_entries * AFXDP_MTU * 1000000000UL /
               kLinkBandwidth;
    }

    std::string to_string();
    void shutdown();
    ~EFASocket();

    friend class EFAFactory;

    // #define FRAME_POOL_DEBUG

#ifdef FRAME_POOL_DEBUG
    std::set<uint64_t> free_frames_;
#endif

    inline uint64_t pop_frame() {
#ifdef FRAME_POOL_DEBUG
        auto frame_offset = frame_pool_->pop();
        CHECK(free_frames_.erase(frame_offset) == 1);
        FrameDesc::clear_fields(frame_offset, umem_buffer_);
        return frame_offset;
#else
        auto frame_offset = frame_pool_->pop();
        FrameDesc::clear_fields(frame_offset, efa_ctl.umem_buffer_);
        return frame_offset;
#endif
    }

    inline void push_frame(uint64_t frame_offset) {
#ifdef FRAME_POOL_DEBUG
        if (free_frames_.find(frame_offset) == free_frames_.end()) {
            FrameDesc::clear_fields(frame_offset, umem_buffer_);
            free_frames_.insert(frame_offset);
            frame_pool_->push(frame_offset);
        } else {
            CHECK(false) << "Frame offset " << std::hex << frame_offset
                         << " size " << std::dec
                         << FrameDesc::get_msgbuf_ptr(frame_offset,
                                                      umem_buffer_)
                                ->get_frame_len()
                         << " already in free_frames_";
        }
#else
        FrameDesc::clear_fields(frame_offset, efa_ctl.umem_buffer_);
        frame_pool_->push(frame_offset);
#endif
    }

   private:
    uint32_t unpulled_tx_pkts_;
    uint32_t fill_queue_entries_;

    std::chrono::time_point<std::chrono::high_resolution_clock> last_stat_;
    inline static std::atomic<uint64_t> out_packets_ = 0;
    inline static std::atomic<uint64_t> out_bytes_ = 0;
    inline static std::atomic<uint64_t> in_packets_ = 0;
    inline static std::atomic<uint64_t> in_bytes_ = 0;
};

}  // namespace uccl