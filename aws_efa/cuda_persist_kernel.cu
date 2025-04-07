#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <tuple>

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

__global__ void emptykernel() {
    if (threadIdx.x == 0)
        printf("Empty kernel: block %d, thread %d\n", blockIdx.x, threadIdx.x);
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

#define kMaxFifoDepth 8
struct IovFifo {
    uint64_t head;   // GPU writes finished index
    uint64_t tail;   // CPU posts working index
    uint64_t abort;  // Telling the kernel to abort
    struct Iov iovs[kMaxFifoDepth];
};

// TODO: per-warp fifo queue (to support multiple SMs), only loading Iovs up to
// iov_n, implementing prims_simple style copy.
__global__ void persistKernel(struct IovFifo **fifo_vec) {
    __shared__ uint64_t cached_tail;
    __shared__ uint64_t abort_flag;
    __shared__ struct Iov *cur_iov;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_tid = bid * blockDim.x + tid;

    struct IovFifo *fifo = fifo_vec[bid];

    // This impossible print is necessary to make sure other kernels run.
    if (global_tid == -1) {
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

class iovMultiFifo {
    static constexpr int kMaxFifos = 32;

    int num_fifo_;
    volatile struct IovFifo *fifo_vec_[kMaxFifos];
    uint64_t fifo_slot_idx_[kMaxFifos];  // monotonic increasing index

   public:
    iovMultiFifo(int num_fifo) : num_fifo_(num_fifo) {
        for (int i = 0; i < num_fifo; i++) {
            cudaHostAlloc(&fifo_vec_[i], sizeof(struct IovFifo),
                          cudaHostAllocMapped);
            cudaCheckErrors("cudaMallocManaged failed");

            // Initialize the fifo
            volatile struct IovFifo *fifo = fifo_vec_[i];
            fifo->head = (uint64_t)-1;
            fifo->tail = (uint64_t)-1;
            for (int j = 0; j < kMaxFifoDepth; j++) {
                fifo->iovs[j].iov_n = -1;
            }
            fifo->abort = 0;

            fifo_slot_idx_[i] = 0;
        }
    }
    ~iovMultiFifo() {
        for (int i = 0; i < num_fifo_; i++) {
            cudaFreeHost((void *)fifo_vec_[i]);
            cudaCheckErrors("cudaFreeHost failed");
        }
    }

    struct IovFifo **get_fifo_vec() { return (struct IovFifo **)fifo_vec_; }

    std::tuple<uint64_t, volatile struct Iov *> reserve_fifo_slot(
        int fifo_idx) {
        auto slot_idx = fifo_slot_idx_[fifo_idx]++;
        auto reserved_iov =
            fifo_vec_[fifo_idx]->iovs + slot_idx % kMaxFifoDepth;
        return {slot_idx, reserved_iov};
    }

    void dispatch_task(int fifo_idx) {
        auto slot_idx = fifo_slot_idx_[fifo_idx] - 1;
        fifo_vec_[fifo_idx]->tail = slot_idx;
        __sync_synchronize();
    }

    bool check_completion(int fifo_idx, uint64_t slot_idx) {
        assert(slot_idx < fifo_slot_idx_[fifo_idx]);
        // The intial -1 will always be less than 0.
        return (int64_t)fifo_vec_[fifo_idx]->head >= (int64_t)slot_idx;
    }

    void abort(int fifo_idx) {
        fifo_vec_[fifo_idx]->abort = 1;
        __sync_synchronize();
    }
};

static constexpr int kTestIters = 10000;
static constexpr int kTestIovs = 128;
static constexpr int kNumBlocks = 4;

// make cuda_persist_kernel
// CUDA_MODULE_LOADING=EAGER ./cuda_persist_kernel
// 11 us for 1 test iov, 82 us for 128 test iovs
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaCheckErrors("cudaStreamCreate failed");
    cudaStreamCreate(&stream2);
    cudaCheckErrors("cudaStreamCreate failed");

    // Preallocate a iov work item.
    struct Iov *cpu_iov = (struct Iov *)malloc(sizeof(struct Iov));
    for (int i = 0; i < kMaxIovs; i++) {
        cudaMalloc(&cpu_iov->srcs[i], 8888);
        cudaMalloc(&cpu_iov->dsts[i], 8888);
        cpu_iov->lens[i] = 8888;
    }

    // Create a iovMultiFifo object.
    iovMultiFifo *fifo = new iovMultiFifo(kNumBlocks);

    persistKernel<<<kNumBlocks, 512, 0, stream1>>>(fifo->get_fifo_vec());
    cudaCheckErrors("persistKernel failed");

    // Test concurrent kernel launch.
    emptykernel<<<4, 512, 0, stream2>>>();
    cudaCheckErrors("emptykernel failed");
    cudaStreamSynchronize(stream2);
    cudaCheckErrors("cudaStreamSynchronize failed");

    for (int i = 0; i < kTestIters; i++) {
        auto fifo_idx = i % kNumBlocks;

        // Reserve a slot in the FIFO.
        auto [slot_idx, gpu_iov] = fifo->reserve_fifo_slot(fifo_idx);
        for (int j = 0; j < kTestIovs; j++) {
            gpu_iov->srcs[j] = cpu_iov->srcs[j];
            gpu_iov->dsts[j] = cpu_iov->dsts[j];
            gpu_iov->lens[j] = cpu_iov->lens[j];
        }
        gpu_iov->iov_n = kTestIovs;

        // CPU dispatches work to GPU.
        fifo->dispatch_task(fifo_idx);

        // CPU side work.
        auto start = std::chrono::high_resolution_clock::now();
        while (fifo->check_completion(fifo_idx, slot_idx) == false) {
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (i % 1000 == 0) {
            auto elapsed_us =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start);
            auto bw_GBps = kTestIovs * 8888 * 1.0 / elapsed_us.count() / 1000;
            printf("CPU wait time: %ld us, bw: %lf GBps\n", elapsed_us.count(),
                   bw_GBps);
        }
    }

    // Tell the kernel to abort.
    for (int i = 0; i < kNumBlocks; i++) {
        fifo->abort(i);
    }

    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");

    delete fifo;
    return 0;
}
