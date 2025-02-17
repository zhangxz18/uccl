#include "eqds.h"
#include "transport_config.h"

namespace uccl {

namespace eqds {

// Make progress on the pacer.
void EQDS::run_pacer(void) 
{
    EQDSChannel::Msg msg;
    while (jring_sc_dequeue_bulk(channel_.cmdq_, &msg, 1, nullptr) == 1) {
        switch (msg.opcode) {
            case EQDSChannel::Msg::kRequestPull:
                for (int i = 0; i < kPortEntropy; i++) {
                    active_senders_.push_back(&msg.eqds_qpcc[i]);
                    msg.eqds_qpcc[i].in_pull_ = true;
                    std::atomic_thread_fence(std::memory_order_acquire);
                }
                VLOG(5) << "Registered in pacer pull queue.";
                break;
            default:
                LOG(ERROR) << "Unknown opcode: " << msg.opcode;
                break;
        }
    }
}

// Grant credit to the sender of this flow.
void EQDS::grant_credit(void) 
{

}

// Handle pull target requested by the sender of this flow.
void EQDS::handle_pull_target(uint32_t pull_target /* unit of chunks */)
{

}

// For original EQDS, it stalls the pacer when ECN ratio reaches a threshold (i.e., 10%).
// Here we use resort to RTT-based stall.
void EQDS::update_cc_state(void)
{

}
}

};