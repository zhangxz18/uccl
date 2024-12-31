/**
 * @file rdma_test.cc
 * @brief Test for NCCL plugin.
 */

#include <iostream>
#include <dlfcn.h>

#include "nccl_net.h"

ncclNet_v8_t* ncclNet;

int main() {

    void* handle = dlopen("./libnccl-net.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Can't load library " << dlerror() << std::endl;
        return 1;
    }

    ncclNet = (ncclNet_v8_t*)dlsym(handle, "ncclNetPlugin_v8");
    if (!ncclNet) {
        std::cerr << "Can't find symbol: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    int result = ncclNet->init(nullptr);
    std::cout << result << std::endl;

    dlclose(handle);
    return 0;
}