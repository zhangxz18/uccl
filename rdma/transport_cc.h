#pragma once

#include <glog/logging.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "util.h"
#include "timely.h"
#include "timing_wheel.h"

namespace uccl {
namespace swift {

struct pending_retr_chunk {
    uint64_t remote_addr;
    uint64_t chunk_addr;
    uint32_t chunk_len;
    uint32_t imm_data;
};

/**
 * @brief Swift Congestion Control (SWCC) protocol control block.
 */
struct Pcb {
    static constexpr std::size_t kInitialCwnd = 256;
    static constexpr std::size_t kSackBitmapBucketSize = sizeof(uint64_t) * 8;
    static constexpr std::size_t kRtoMaxRexmitConsectutiveAllowed = 32;
    static constexpr int kRtoExpireThresInTicks = 3;  // in slow timer ticks.
    static constexpr int kRtoDisabled = -1;
    Pcb()
        : timely(freq_ghz, kLinkBandwidth) {}

    Timely timely;

    void mutliplicative_decrease() { ecn_alpha /= 2; }
    void additive_increase() {
        ecn_alpha += 0.1;
        if (ecn_alpha > 1.0) ecn_alpha = 1.0;
    }

    UINT_CSN seqno() const { return snd_nxt; }
    UINT_CSN get_snd_nxt() {
        UINT_CSN seqno = snd_nxt;
        snd_nxt += 1;
        return seqno;
    }

    std::string to_string() const {
        auto avg_rtt_diff = timely.get_avg_rtt_diff();
        auto rate_gbps = timely.get_rate_gbps();
        std::string s; s.clear();
        s += "[CC] snd_nxt: " + std::to_string(snd_nxt.to_uint32()) +
             " snd_una: " + std::to_string(snd_una.to_uint32()) +
             " rcv_nxt: " + std::to_string(rcv_nxt.to_uint32()) +
             " fast/rto rexmits: " + std::to_string(stats_fast_rexmits) + "/" + std::to_string(stats_rto_rexmits) +
             Format(" prev_rtt: %.2lf us ", timely.get_avg_rtt()) + 
             Format(" rate: %.2lf Gbps ", rate_gbps);
        return s;
    }

    UINT_CSN ackno() const { return rcv_nxt; }
    bool max_rto_rexmits_consectutive_reached() const {
        return rto_rexmits_consectutive >= kRtoMaxRexmitConsectutiveAllowed;
    }
    bool rto_disabled() const { return rto_timer == kRtoDisabled; }
    bool rto_expired() const { return rto_timer >= kRtoExpireThresInTicks; }

    UINT_CSN get_rcv_nxt() const { return rcv_nxt; }
    void advance_rcv_nxt(UINT_CSN n) { rcv_nxt += n; }
    void advance_rcv_nxt() { rcv_nxt += 1; }

    /************* RTO related *************/
    void rto_enable() { rto_timer = 0; }
    void rto_disable() { rto_timer = kRtoDisabled; }
    void rto_reset() { rto_enable(); }
    void rto_maybe_reset() {
        if (snd_una == snd_nxt)
            rto_disable();
        else
            rto_reset();
    }
    void rto_advance() { rto_timer++; }

    /************* RTO related *************/

    void barrier_bitmap_shift_left_one() {
        constexpr size_t barrier_bitmap_bucket_max_idx = 
            kSackBitmapSize / kSackBitmapBucketSize - 1;

        for (size_t i = 0; i < barrier_bitmap_bucket_max_idx; i++) {
            // Shift the current each bucket to the left by 1 and take the most
            // significant bit from the next bucket
            uint64_t &barrier_bitmap_left_bucket = barrier_bitmap[i];
            const uint64_t barrier_bitmap_right_bucket = barrier_bitmap[i + 1];

            barrier_bitmap_left_bucket = (barrier_bitmap_left_bucket >> 1) |
                                         (barrier_bitmap_right_bucket << 63);
        }

        // Special handling for the right most bucket
        uint64_t &barrier_bitmap_right_most_bucket =
            barrier_bitmap[barrier_bitmap_bucket_max_idx];
        barrier_bitmap_right_most_bucket >>= 1;

        barrier_bitmap_count--;

        // Increment the shift count.
        shift_count++;
    }

    void sack_bitmap_shift_left_one() {
        constexpr size_t sack_bitmap_bucket_max_idx =
            kSackBitmapSize / kSackBitmapBucketSize - 1;

        for (size_t i = 0; i < sack_bitmap_bucket_max_idx; i++) {
            // Shift the current each bucket to the left by 1 and take the most
            // significant bit from the next bucket
            uint64_t &sack_bitmap_left_bucket = sack_bitmap[i];
            const uint64_t sack_bitmap_right_bucket = sack_bitmap[i + 1];

            sack_bitmap_left_bucket = (sack_bitmap_left_bucket >> 1) |
                                      (sack_bitmap_right_bucket << 63);
        }

        // Special handling for the right most bucket
        uint64_t &sack_bitmap_right_most_bucket =
            sack_bitmap[sack_bitmap_bucket_max_idx];
        sack_bitmap_right_most_bucket >>= 1;

        sack_bitmap_count--;
    }

    void barrier_bitmap_bit_set(const size_t index) {
        const size_t barrier_bitmap_bucket_idx = index / kSackBitmapBucketSize;
        const size_t barrier_bitmap_idx_in_bucket = index % kSackBitmapBucketSize;

        LOG_IF(FATAL, index >= kSackBitmapSize)
            << "Index out of bounds: " << index;

        barrier_bitmap[barrier_bitmap_bucket_idx] |=
            (1ULL << barrier_bitmap_idx_in_bucket);

        barrier_bitmap_count++;
    }

    void sack_bitmap_bit_set(const size_t index) {
        const size_t sack_bitmap_bucket_idx = index / kSackBitmapBucketSize;
        const size_t sack_bitmap_idx_in_bucket = index % kSackBitmapBucketSize;

        LOG_IF(FATAL, index >= kSackBitmapSize)
            << "Index out of bounds: " << index;

        sack_bitmap[sack_bitmap_bucket_idx] |=
            (1ULL << sack_bitmap_idx_in_bucket);

        sack_bitmap_count++;
    }

    uint64_t t_remote_nic_rx{0};
    uint64_t t_remote_host_rx{0};

    uint32_t target_delay{0};
    UINT_CSN snd_nxt{0};
    UINT_CSN snd_una{0};
    UINT_CSN snd_ooo_acks{0};
    UINT_CSN rcv_nxt{0};
    uint64_t sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
    uint8_t sack_bitmap_count{0};
    // Sender copy of SACK bitmap for retransmission.
    uint64_t tx_sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
    uint8_t tx_sack_bitmap_count{0};
    // The starting CSN of the copy of SACK bitmap.
    uint32_t base_csn{0};

    // For RDMA retransmission.
    uint64_t barrier_bitmap[kSackBitmapSize / kSackBitmapBucketSize] {0};
    uint8_t barrier_bitmap_count{0};
    std::unordered_map<uint64_t, struct pending_retr_chunk> pending_retr_chunks;
    // Incremented when a bitmap is shifted left by 1.
    // Even if increment every one microsecond, it will take 584542 years to overflow.
    uint64_t shift_count;
    uint16_t cwnd{kInitialCwnd};
    uint16_t duplicate_acks{0};
    int rto_timer{kRtoDisabled};
    uint16_t rto_rexmits_consectutive{0};
    double ecn_alpha{1.0};
    
    // Stats
    uint32_t stats_fast_rexmits{0};
    uint32_t stats_rto_rexmits{0};
    uint32_t stats_accept_retr{0};
    uint32_t stats_accept_barrier{0};
    uint32_t stats_chunk_drop{0};
    uint32_t stats_barrier_drop{0};
    uint32_t stats_retr_chunk_drop{0};
    uint32_t stats_ooo{0};
    uint32_t stats_real_ooo{0};
    uint32_t stats_maxooo{0};
};

}  // namespace swift
}  // namespace uccl
