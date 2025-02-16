/**
 * @file eqds.h
 * @brief EQDS congestion control [NSDI'22]
 */

#pragma once

#include <iomanip>
#include <list>

#include <infiniband/verbs.h>

#include "util.h"
#include "util_latency.h"
#include "util_timer.h"
#include "util_jring.h"

namespace uccl {

// Per-flow congestion control state for EQDS.
struct EQDSFlowCC {
    static constexpr uint32_t kMaxCwnd = 200000;
    
    // Already in pull queue or not.
    bool in_pull = false;

    // Receive request credit in pull_quanta, but consume it in bytes
    int32_t credit_pull_ = 0;
    int32_t credit_spec_ = kMaxCwnd;

    inline int32_t credit() { return credit_pull_ + credit_spec_; }

    inline bool spend_credit(uint32_t chunk_size) {
        return true;
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
        EQDSFlowCC *flow_cc;
        PollCtx *poll_ctx;
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

    // Request pacer to grant credit to this flow.
    // This function is thread-safe.
    inline void request_pull(EQDSFlowCC *flow_cc, PollCtx *poll_ctx) {
        if (flow_cc->in_pull) return;
        flow_cc->in_pull = true;
        
        EQDSChannel::Msg msg = {
            .opcode = EQDSChannel::Msg::Op::kRequestPull,
            .flow_cc = flow_cc,
            .poll_ctx = poll_ctx,
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

    std::list<EQDSFlowCC *> active_senders_;
    std::list<EQDSFlowCC *> idle_senders_;

    bool shutdown_ = false;
};

};