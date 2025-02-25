#include <cuda_runtime.h>
#include <stdio.h>

#include "scattered_memcpy.cuh"

__global__ void kernelScatteredMemcpy(uint32_t num_copies,
                                      __grid_constant__ const copy_param_t p) {
    // Total threads in the grid.
    int total_threads = gridDim.x * blockDim.x;
    // Compute our unique global thread id.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Number of threads per copy.
    int threads_per_copy = total_threads / num_copies;

    // Map each thread to a copy.
    int copy_idx = global_id / threads_per_copy;
    if (copy_idx >= num_copies) return;  // In case of rounding

    // Compute local thread index within the group assigned to this copy.
    int local_thread_idx = global_id % threads_per_copy;

    // Retrieve parameters for this copy.
    uint64_t total_size = p.size[copy_idx];
    if (total_size == 0) return;

    char* src_ptr = (char*)p.src[copy_idx];
    char* dst_ptr = (char*)p.dst[copy_idx];

    // Copy 8-byte chunks first (if possible)
    uint64_t num_full = total_size / 8;
    uint64_t* src_u64 = (uint64_t*)src_ptr;
    uint64_t* dst_u64 = (uint64_t*)dst_ptr;

    // Each thread in the group copies its portion of 64-bit words.
    for (uint64_t i = local_thread_idx; i < num_full; i += threads_per_copy) {
        dst_u64[i] = src_u64[i];
    }

    // Handle the remaining tail bytes (if any)
    uint64_t tail_start = num_full * 8;
    // Let only one thread in the copy group (e.g. local_thread_idx == 0) copy
    // the tail.
    if (local_thread_idx == 0) {
        for (uint64_t i = tail_start; i < total_size; i++) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

void launchScatteredMemcpy(uint32_t num_copies, const copy_param_t* params) {
    // Launch the kernel
    kernelScatteredMemcpy<<<THREAD_BLOCKS, THREADS_PER_BLOCK>>>(num_copies,
                                                                *params);

    // Wait for kernel to complete.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
    }
}

void launchScatteredMemcpyAsync(uint32_t num_copies, const copy_param_t* params,
                                cudaStream_t stream) {
    // Launch the kernel
    kernelScatteredMemcpy<<<THREAD_BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
        num_copies, *params);
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
