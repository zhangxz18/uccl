/**
 * @file eqds.h
 * @brief EQDS congestion control [NSDI'22]
 */

#pragma once

#include <iomanip>
#include <list>

#include <infiniband/verbs.h>

#include "transport_config.h"
#include "util.h"
#include "util_latency.h"
#include "util_timer.h"
#include "util_jring.h"

namespace uccl {

namespace eqds {

typedef uint32_t PullQuanta;

#define PULL_QUANTUM 512
#define PULL_SHIFT 9

static inline uint32_t unquantize(uint32_t pull_quanta) {
    return pull_quanta << PULL_SHIFT;
}

static inline PullQuanta quantize_floor(uint32_t bytes) {
    return bytes >> PULL_SHIFT;
}

static inline PullQuanta quantize_ceil(uint32_t bytes) {
    return (bytes + PULL_QUANTUM - 1) >> PULL_SHIFT;
}

// Per-QP congestion control state for EQDS.
struct EQDSCC {

    static constexpr PullQuanta INIT_PULL_QUANTA = 1000000000;
    static constexpr uint32_t kEQDSMaxCwnd = 200000; // Bytes
    
    /********************************************************************/
    /************************ Sender-side states ************************/
    /********************************************************************/
    
    // Last received highest credit in PullQuanta.
    PullQuanta pull_ = INIT_PULL_QUANTA;

    // Receive request credit in PullQuanta, but consume it in bytes
    int32_t credit_pull_ = 0;
    int32_t credit_spec_ = kEQDSMaxCwnd;

    bool in_speculating_;
    
    uint32_t unsent_bytes_;

    /********************************************************************/
    /************************ Receiver-side states ************************/
    /********************************************************************/

    bool in_pull_ = false;

    PullQuanta highest_pull_target_;

    inline int32_t credit() { return credit_pull_ + credit_spec_; }

    // Called when transmitting a chunk.
    // Return true if we can transmit the chunk. Otherwise,
    // sender should pause sending this message until credit is received.
    inline bool spend_credit(uint32_t chunk_size) {

        return true;

        if (credit_pull_ > 0) {
            credit_pull_ -= chunk_size;
            return true;
        } else if (in_speculating_ && credit_spec_ > 0) {
            credit_spec_ -= chunk_size;
            return true;
        }

        // XXX
        credit_spec_ -= chunk_size;
        return false;
    }

    inline void receive_credit(PullQuanta pullno) {
        if (pullno > pull_) {
            PullQuanta extra_credit = pullno - pull_;
            credit_pull_ += unquantize(extra_credit);
            if (credit_pull_ > kEQDSMaxCwnd) {
                credit_pull_ = kEQDSMaxCwnd;
            }
            pull_ = pullno;
        }
    }

    inline void stop_speculating() {
        in_speculating_ = false;
    }

    inline bool can_continue_send() {
        if (credit() <= 0) return false;

        // Nothing to send
        if (unsent_bytes_ == 0) return false;

        return true;
    }

    inline void try_continue_send() {

    }

    inline PullQuanta compute_pull_target() {
        PullQuanta target;
        return target;
    }

    inline void handle_pull_target(PullQuanta pull_target) {
        if (pull_target > highest_pull_target_) {
            highest_pull_target_ = pull_target;
        }
    }

};

class EQDSChannel {
    static constexpr uint32_t kChannelSize = 1024;

public:
    struct Msg {
        enum Op: uint8_t {
            kRequestPull,
        };
        Op opcode;
        EQDSCC *eqds_cc;
    };
    static_assert(sizeof(Msg) % 4 == 0, "channelMsg must be 32-bit aligned");

    EQDSChannel() {
        cmdq_ = create_ring(sizeof(Msg), kChannelSize);
    }

    ~EQDSChannel() {
        free(cmdq_);
    }

    jring_t *cmdq_;
};

class EQDS {
public:

    EQDSChannel channel_;

    // Make progress on the pacer.
    void run_pacer(void);

    // Grant credit to the sender of this flow.
    void grant_credit(void);

    // Handle pull target requested by the sender of this flow.
    void handle_pull_target(uint32_t pull_target /* unit of chunks */);

    // For original EQDS, it stalls the pacer when ECN ratio reaches a threshold (i.e., 10%).
    // Here we use resort to RTT-based stall.
    void update_cc_state(void);

    // [Thread-safe] Request pacer to grant credit to this flow.
    inline void request_pull(EQDSCC *eqds_cc) {        
        EQDSChannel::Msg msg = {
            .opcode = EQDSChannel::Msg::Op::kRequestPull,
            .eqds_cc = eqds_cc,
        };
        while (jring_mp_enqueue_bulk(channel_.cmdq_, &msg, 1, nullptr) != 1) {}
    }

    EQDS(int dev): dev_(dev), channel_() {
        // Initialize the pacer thread.
        pacer_th_ = std::thread([this] {
            // Pin the pacer thread to a specific CPU.
            pin_thread_to_cpu(PACER_CPU_START + dev_);
            while (!shutdown_) {
                run_pacer();
            }
        });

    }

    ~EQDS() {}

    // Shutdown the EQDS pacer thread.
    inline void shutdown(void) {
        shutdown_ = true;
        pacer_th_.join();
    }

private:
    std::thread pacer_th_;
    int dev_;

    std::list<EQDSCC *> active_senders_;
    std::list<EQDSCC *> idle_senders_;

    bool shutdown_ = false;
};

}
};