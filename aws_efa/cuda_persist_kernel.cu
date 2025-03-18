#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

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

__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
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

__device__ __forceinline__ void st_relaxed_sys(uint64_t* ptr, uint64_t val) {
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
    void* iov_addrs[kMaxIovs];
    int iov_lens[kMaxIovs];
    int dst_offsets[kMaxIovs];
    int iov_n;
    int gpu_idx;  // for debugging
    int step;     // for debugging
};

#define kMaxFifoDepth 1024
struct IovFifo {
    uint64_t head;  // GPU writes finished index
    uint64_t tail;  // CPU posts working index
    struct Iov iovs[kMaxFifoDepth];
};

__global__ void persistKernel(struct IovFifo* fifo, int steps) {
    __shared__ uint64_t cached_tail;
    __shared__ struct Iov cached_iov;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    // Initing per-threadblock variables
    if (tid == 0) cached_tail = (uint64_t)-1;
    __syncthreads();

    int i = 0;
    while (i++ < steps) {
        // Each thread block loads new work from CPU.
        if (tid == 0) {
            uint64_t cur_tail;
            do {
                cur_tail = ld_volatile(&fifo->tail);
            } while (cur_tail != cached_tail + 1);

            cached_tail = cur_tail;
            cached_iov = fifo->iovs[cached_tail % kMaxFifoDepth];
            // TODO: using volatile load?
        }
        __syncthreads();

        // do something with cached_iov
        if (global_tid == 0) {
            if (cached_tail % 1000 == 0) {
                printf(
                    "PersistKernel: block %d, thread %d, cached_tail %lu, "
                    "iov_n %d\n",
                    bid, tid, cached_tail, cached_iov.iov_n);
            }
        }

        __syncthreads();

        // Post the finished work to the GPU
        if (global_tid == 0) {
            fence_acq_rel_sys();
            st_relaxed_sys(&fifo->head, cached_tail);
        }
    }
}

__global__ void emptykernel() {
    if (threadIdx.x == 0)
        printf("Emptykernel: block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

static constexpr int kTestIters = 10000;

// make cuda_persist_kernel
// CUDA_MODULE_LOADING=EAGER ./cuda_persist_kernel
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaCheckErrors("cudaStreamCreate failed");
    cudaStreamCreate(&stream2);
    cudaCheckErrors("cudaStreamCreate failed");

    struct IovFifo* __fifo;
    cudaHostAlloc(&__fifo, sizeof(struct IovFifo), cudaHostAllocMapped);
    cudaCheckErrors("cudaMallocManaged failed");

    volatile struct IovFifo* fifo = (volatile struct IovFifo*)__fifo;

    // Initialize the fifo
    fifo->head = (uint64_t)-1;
    fifo->tail = (uint64_t)-1;
    for (int i = 0; i < kMaxFifoDepth; i++) {
        fifo->iovs[i].iov_n = -1;
    }

    persistKernel<<<4, 512, 0, stream1>>>(__fifo, kTestIters);
    cudaCheckErrors("persistKernel failed");

    // Test concurrent kernel launch.
    emptykernel<<<4, 512, 0, stream2>>>();
    cudaCheckErrors("emptykernel failed");
    cudaStreamSynchronize(stream2);
    cudaCheckErrors("cudaStreamSynchronize failed");

    for (int i = 0; i < kTestIters; i++) {
        fifo->iovs[i % kMaxFifoDepth].iov_n = i;
        fifo->tail = i;
        __sync_synchronize();

        // CPU side work.

        while (fifo->head != i) {
        }
    }

    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");

    cudaFreeHost(__fifo);
    cudaCheckErrors("cudaFreeHost failed");

    return 0;
}
