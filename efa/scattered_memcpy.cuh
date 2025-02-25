#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/:
// CUDA 12.1 supports upto 32764 as kernel parameters.

// Supporting 128 * 9KB = 1.152MB net chunk size.
static const uint32_t MAX_SCATTERED_COPIES = 128;
static const uint32_t THREADS_PER_COPY = 256;

typedef struct {
    uint64_t dst[MAX_SCATTERED_COPIES];
    uint64_t src[MAX_SCATTERED_COPIES];
    uint64_t size[MAX_SCATTERED_COPIES];
} copy_param_t;

// Launch wrapper (exposed with C linkage) that is callable from a .cc file.
void launchScatteredMemcpy(const copy_param_t *p);
void launchScatteredMemcpyAsync(const copy_param_t *params,
                                cudaStream_t stream);
int pollScatteredMemcpy(cudaStream_t stream);

#ifdef __cplusplus
}
#endif
