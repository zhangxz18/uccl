/**
 * @file rdma_test.cc
 * @brief Test for NCCL plugin.
 */

#include <assert.h>
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

    int ret = ncclNet->init(nullptr);
    assert(ret == 0);
    std::cout << ret << std::endl;

    int num_devices;
    ret = ncclNet->devices(&num_devices);
    assert(ret == 0);

    ncclNetProperties_v8_t properties;
    ret = ncclNet->getProperties(0, &properties);
    assert(ret == 0);
    
    std::cout << "# of devices: " << num_devices << std::endl;


    dlclose(handle);
    return 0;
}