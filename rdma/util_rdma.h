#ifndef UTIL_RDMA_H
#define UTIL_RDMA_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>

class RDMAContext;
class RDMAFactory;
extern RDMAFactory rdma_ctl;

namespace uccl {

    class RDMAContext {
        public:

        friend class RDMAFactory;
    };

    class RDMAFactory {

        char interface_name_[256];
        public:
            static void init(const char *interface_name, uint64_t num_queues);
            static RDMAContext *CreateContext(int queue_id);
            static void shutdown(void);
    };

}

#endif