#include "util_rdma.h"

#include "transport_config.h"

namespace uccl {

    RDMAFactory rdma_ctl;

    void RDMAFactory::init(const char *interface_name, uint64_t num_queues) {
        strcpy(rdma_ctl.interface_name_, interface_name);
    }

    RDMAContext *RDMAFactory::CreateContext(int queue_id) {
        return new RDMAContext();
    }

    void RDMAFactory::shutdown(void) {
        // Do nothing
    }

}