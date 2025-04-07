#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>

#define cudaCheckErrors(msg)                                        \
    do {                                                            \
        cudaError_t __err = cudaGetLastError();                     \
        if (__err != cudaSuccess) {                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
                    cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");             \
            exit(1);                                                \
        }                                                           \
    } while (0)

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

// Yang: 128 max scattered IOVs
#define kMaxIovs 128
struct alignas(8) Iov {
    void *srcs[kMaxIovs];
    void *dsts[kMaxIovs];
    int lens[kMaxIovs];
    int iov_n;
    int gpu_idx;  // for debugging
    int step;     // for debugging
};

#define kMaxFifoDepth 1024
struct IovFifo {
    uint64_t head;   // GPU writes finished index
    uint64_t tail;   // CPU posts working index
    uint64_t abort;  // Telling the kernel to abort
    struct Iov iovs[kMaxFifoDepth];
};

__device__ void kernelScatteredMemcpy(struct Iov *iov) {
    // Total threads in the grid.
    int total_threads = gridDim.x * blockDim.x;
    // Compute our unique global thread id.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Number of threads per copy.
    int threads_per_copy = total_threads / iov->iov_n;

    // Map each thread to a copy.
    int copy_idx = global_id / threads_per_copy;
    if (copy_idx >= iov->iov_n) return;  // In case of rounding

    // Compute local thread index within the group assigned to this copy.
    int local_thread_idx = global_id % threads_per_copy;

    // Retrieve parameters for this copy.
    uint64_t total_size = iov->lens[copy_idx];
    if (total_size == 0) return;

    char *src_ptr = (char *)iov->srcs[copy_idx];
    char *dst_ptr = (char *)iov->dsts[copy_idx];

    // Copy 8-byte chunks first (if possible)
    uint64_t num_full = total_size / 8;
    uint64_t *src_u64 = (uint64_t *)src_ptr;
    uint64_t *dst_u64 = (uint64_t *)dst_ptr;

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

// TODO: per-warp fifo queue (to support multiple SMs), only loading Iovs up to
// iov_n, implementing prims_simple style copy.
__global__ void persistKernel(struct IovFifo *fifo) {
    __shared__ uint64_t cached_tail;
    __shared__ uint64_t abort_flag;
    __shared__ struct Iov *cur_iov;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // This print is necessary to make sure other kernels can run.
    if (global_tid == 0) {
        printf("Persist kernel: block %d, thread %d\n", bid, tid);
    }

    // Initing per-threadblock variables
    if (tid == 0) cached_tail = (uint64_t)-1;
    __syncthreads();

    // We should avoid all thread loading the global memory at the same, as this
    // will cause severe performance drop.
    // while (ld_volatile(&fifo->abort) == 0) {
    while (true) {
        // Each thread block loads new work from CPU.
        if (tid == 0) {
            uint64_t cur_tail;
            do {
                abort_flag = ld_volatile(&fifo->abort);
                if (abort_flag) break;

                cur_tail = ld_volatile(&fifo->tail);
            } while (cur_tail != cached_tail + 1);

            cached_tail = cur_tail;
            cur_iov = fifo->iovs + cached_tail % kMaxFifoDepth;

            // TODO: using volatile load?
        }
        __syncthreads();
        if (abort_flag) return;

        kernelScatteredMemcpy(cur_iov);

        __syncthreads();

        // Post the finished work to the GPU
        if (tid == 0) {
            fence_acq_rel_sys();
            st_relaxed_sys(&fifo->head, cached_tail);
        }
    }
}

__global__ void emptykernel() {
    if (threadIdx.x == 0)
        printf("Empty kernel: block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

static constexpr int kTestIters = 10000;
static constexpr int kTestIovs = 128;

// make cuda_persist_kernel
// CUDA_MODULE_LOADING=EAGER ./cuda_persist_kernel
// 11 us for 1 test iov, 82 us for 128 test iovs
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaCheckErrors("cudaStreamCreate failed");
    cudaStreamCreate(&stream2);
    cudaCheckErrors("cudaStreamCreate failed");

    struct IovFifo *__fifo;
    cudaHostAlloc(&__fifo, sizeof(struct IovFifo), cudaHostAllocMapped);
    cudaCheckErrors("cudaMallocManaged failed");

    volatile struct IovFifo *fifo = (volatile struct IovFifo *)__fifo;

    // Initialize the fifo
    fifo->head = (uint64_t)-1;
    fifo->tail = (uint64_t)-1;
    for (int i = 0; i < kMaxFifoDepth; i++) {
        fifo->iovs[i].iov_n = -1;
    }
    fifo->abort = 0;

    // Preallocate a iov work item.
    struct Iov *cpu_iov = (struct Iov *)malloc(sizeof(struct Iov));
    for (int i = 0; i < kMaxIovs; i++) {
        cudaMalloc(&cpu_iov->srcs[i], 8888);
        cudaMalloc(&cpu_iov->dsts[i], 8888);
        cpu_iov->lens[i] = 8888;
    }
    cpu_iov->iov_n = kTestIovs;

    persistKernel<<<1, 512, 0, stream1>>>(__fifo);
    cudaCheckErrors("persistKernel failed");

    // Test concurrent kernel launch.
    emptykernel<<<4, 512, 0, stream2>>>();
    cudaCheckErrors("emptykernel failed");
    cudaStreamSynchronize(stream2);
    cudaCheckErrors("cudaStreamSynchronize failed");

    for (int i = 0; i < kTestIters; i++) {
        // CPU dispatches work to GPU.
        volatile struct Iov *gpu_iov = fifo->iovs + i % kMaxFifoDepth;
        for (int j = 0; j < kMaxIovs; j++) {
            gpu_iov->srcs[j] = cpu_iov->srcs[j];
            gpu_iov->dsts[j] = cpu_iov->dsts[j];
            gpu_iov->lens[j] = cpu_iov->lens[j];
        }
        gpu_iov->iov_n = cpu_iov->iov_n;
        fifo->tail = i;
        __sync_synchronize();

        // CPU side work.
        auto start = std::chrono::high_resolution_clock::now();
        while (fifo->head != i) {
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (i % 1000 == 0)
            printf("CPU wait time: %ld us\n", elapsed_us.count());
    }

    // Tell the kernel to abort.
    fifo->abort = 1;
    __sync_synchronize();

    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");

    cudaFreeHost(__fifo);
    cudaCheckErrors("cudaFreeHost failed");

    return 0;
}
