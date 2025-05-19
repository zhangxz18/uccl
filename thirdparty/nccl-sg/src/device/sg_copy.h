#ifndef __SG_COPY__
#define __SG_COPY__

#include <assert.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "sg_copy_if.h"

__device__ __forceinline__ uint64_t ld_volatile(uint64_t *ptr) {
    uint64_t ans;
    asm volatile("ld.volatile.global.u64 %0, [%1];"
                 : "=l"(ans)
                 : "l"(ptr)
                 : "memory");
    return ans;
}

__device__ __forceinline__ void fence_acq_rel_sys() {
#if __CUDA_ARCH__ >= 700
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
    asm volatile("membar.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys(uint64_t *ptr, uint64_t val) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(ptr), "l"(val)
                 : "memory");
#else
    asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(ptr), "l"(val)
                 : "memory");
#endif
}

template <typename X, typename Y, typename Z = decltype(X() + Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
    return (x + y - 1) / y;
}

#endif  // __SG_COPY__