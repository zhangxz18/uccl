#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// An A100 SM can have at most 32 thread blocks.
static const uint32_t THREAD_BLOCKS = 32;
// NCCL uses 512 for current gen, and 256 for previous gen.
static const uint32_t THREADS_PER_BLOCK = 1024;
// Supporting 128 * 9KB = 1.152MB net chunk size.
static const uint32_t MAX_SCATTERED_COPIES = 128;

static const uint32_t THREADS_PER_COPY =
    THREADS_PER_BLOCK * THREAD_BLOCKS / MAX_SCATTERED_COPIES;

typedef struct {
    uint64_t dst[MAX_SCATTERED_COPIES];
    uint64_t src[MAX_SCATTERED_COPIES];
    uint64_t size[MAX_SCATTERED_COPIES];
} copy_param_t;

static const uint32_t param_size = sizeof(copy_param_t);
// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/:
// CUDA 12.1 supports upto 32764 as kernel parameters.
static_assert(param_size <= 32764, "param_size must be <= 32764");

// Launch wrapper (exposed with C linkage) that is callable from a .cc file.
void launchScatteredMemcpy(const copy_param_t *p);
// 1 vs 128 MAX_SCATTERED_COPIES -> 2.6 vs 3.6 us async launch time.
void launchScatteredMemcpyAsync(const copy_param_t *params,
                                cudaStream_t stream);
// 0.56 us for pollScatteredMemcpy.
int pollScatteredMemcpy(cudaStream_t stream);

#ifdef __cplusplus
}
#endif
