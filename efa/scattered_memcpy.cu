#include <cuda_runtime.h>
#include <stdio.h>

#include "scattered_memcpy.cuh"

__global__ void kernelScatteredMemcpy(__grid_constant__ const copy_param_t p) {
    int copy_idx = blockIdx.x;  // Each block handles one copy operation
    int thread_id = threadIdx.x;

    if (copy_idx >= MAX_SCATTERED_COPIES || p.size[copy_idx] == 0) return;

    char *src_ptr = (char *)p.src[copy_idx];
    char *dst_ptr = (char *)p.dst[copy_idx];
    uint64_t total_size = p.size[copy_idx];

    // Use uint64_t for efficient memory transfers (8 bytes at a time)
    uint64_t *src_u64 = (uint64_t *)src_ptr;
    uint64_t *dst_u64 = (uint64_t *)dst_ptr;
    uint64_t num_full_u64 = total_size / 8;
    uint64_t thread_offset = thread_id;

    // Parallelized copying of 64-bit chunks
    for (uint64_t i = thread_offset; i < num_full_u64; i += blockDim.x) {
        dst_u64[i] = src_u64[i];
    }

    // Handle remaining bytes (unaligned part)
    if (thread_id == 0) {
        char *src_tail = src_ptr + (num_full_u64 * 8);
        char *dst_tail = dst_ptr + (num_full_u64 * 8);
        for (uint64_t i = 0; i < (total_size % 8); i++) {
            dst_tail[i] = src_tail[i];
        }
    }
}

void launchScatteredMemcpy(const copy_param_t *params) {
    int num_copies = MAX_SCATTERED_COPIES;

    // Configure CUDA kernel launch
    int threadsPerBlock = THREADS_PER_COPY;
    int numBlocks = num_copies;  // One block per copy

    // Launch the kernel
    kernelScatteredMemcpy<<<numBlocks, threadsPerBlock>>>(*params);

    // Wait for kernel to complete.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
    }
}

void launchScatteredMemcpyAsync(const copy_param_t *params,
                                cudaStream_t stream) {
    int num_copies = MAX_SCATTERED_COPIES;

    // Configure CUDA kernel launch
    int threadsPerBlock = THREADS_PER_COPY;
    int numBlocks = num_copies;  // One block per copy

    // Launch the kernel
    kernelScatteredMemcpy<<<numBlocks, threadsPerBlock, 0, stream>>>(*params);
}

int pollScatteredMemcpy(cudaStream_t stream) {
    cudaError_t err = cudaStreamQuery(stream);
    if (err != cudaSuccess && err != cudaErrorNotReady) {
        fprintf(stderr, "cudaStreamQuery failed: %s\n",
                cudaGetErrorString(err));
        exit(0);
    }
    return err == cudaSuccess;
}
