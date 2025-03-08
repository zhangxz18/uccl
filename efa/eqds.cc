#include "eqds.h"
#include "transport_config.h"
#include "util_list.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>

namespace uccl {

namespace eqds {

// Make progress on the pacer.
void EQDS::run_pacer(void) {
    auto now = rdtsc();
    handle_pull_request();

    // It is our responsibility to poll Tx completion events.
    handle_poll_cq();

    if (now - last_pacing_tsc_ >= pacing_interval_tsc_) {
        handle_grant_credit();
        last_pacing_tsc_ = now;
    }
}

// Handle registration requests.
void EQDS::handle_pull_request(void)
{

}

// Handle Credit CQ TX events.
void EQDS::handle_poll_cq(void)
{

}

// Grant credits to senders.
void EQDS::handle_grant_credit(void)
{

}

int CreditQPContext::poll_credit_cq(void)
{
    return 0;
}

}; // namesapce eqds
}; // namespace uccl