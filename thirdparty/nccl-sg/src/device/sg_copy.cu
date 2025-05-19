#include "sg_copy.h"

__device__ void kernelScatteredMemcpy(struct Iov *iov) {
    typedef float2 T;
    static constexpr int kCpAsycDepth = 8;
    __shared__ T smem[kNumThPerBlock * kCpAsycDepth];

    // Each SM is an independent worker.
    int nthreads = blockDim.x;
    int tid = threadIdx.x;
    int iov_n = iov->iov_n;

    // Number of threads per copy: A100 has 8 * 128bit mem transactions.
    // https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
    int nthreads_per_iov = max(8, nthreads / iov_n);
    // Ignoring some non-rounded threads
    if (tid > nthreads_per_iov * iov_n) return;

    int iov_n_per_iter = nthreads / nthreads_per_iov;
    int start_iov = tid / nthreads_per_iov;

    for (int i = start_iov; i < iov_n; i += iov_n_per_iter) {
        // Map each thread to a iov copy.
        int iov_idx = i;
        // Compute local tid within the th group assigned to this iov copy.
        int local_tid = tid % nthreads_per_iov;

        // Retrieve parameters for this copy.
        char *src_ptr = (char *)iov->srcs[iov_idx];
        char *dst_ptr = (char *)iov->dsts[iov_idx];
        int iov_len = iov->lens[iov_idx];
        if (iov_len == 0) return;

        // Copy t-byte chunks first (if possible)
        int num_full = iov_len / sizeof(T);
        T *src_T = (T *)src_ptr;
        T *dst_T = (T *)dst_ptr;

        int depth = 0;
        // Each thread in the group copies its portion of data.
        for (int j = local_tid; j < num_full; j += nthreads_per_iov) {
            // dst_T[j] = src_T[j];

            void *smemBytePtr = (void *)&smem[tid + nthreads * depth++];
            const void *gmemBytePtr = (const void *)&src_T[j];
            __pipeline_memcpy_async(smemBytePtr, gmemBytePtr, sizeof(T));

            if (depth == kCpAsycDepth || j + nthreads_per_iov >= num_full) {
                __pipeline_commit();
                __pipeline_wait_prior(0);
                // Copy the data from shared memory to global memory
                for (int k = 0; k < depth; k++) {
                    dst_T[j - (depth - 1 - k) * nthreads_per_iov] =
                        smem[tid + nthreads * k];
                }
                depth = 0;
            }
        }

        // Let only one thread in the copy group (e.g. local_tid == 0) copy
        // the tail.
        if (local_tid == 0) {
            // Handle the remaining tail bytes (if any)
            int tail_start = num_full * 8;
            for (int j = tail_start; j < iov_len; j++) {
                dst_ptr[j] = src_ptr[j];
            }
        }
    }
}

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
    if (tid == 0) {
        abort_flag = 0;
        cached_tail = (uint64_t)-1;
    }
    __syncthreads();

    // We should avoid all thread loading the global memory at the same, as this
    // will cause severe performance drop.
    // while (ld_volatile(&fifo->abort) == 0) {
    while (true) {
        // Each thread block loads new work from CPU.
        if (tid == 0) {
            uint64_t cur_tail;
            do {
                cur_tail = ld_volatile(&fifo->tail);

                if (cur_tail == kAbortTailValue) {
                    // The CPU has posted a abort signal.
                    abort_flag = 1;
                    break;
                }
            } while ((int64_t)cur_tail < (int64_t)(cached_tail + 1));

            // Processing one iov at a time.
            cur_tail = cached_tail + 1;

            cached_tail = cur_tail;
            cur_iov = fifo->iovs + cached_tail % kFifoCap;
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

void iovMultiFifo::launchSGCopyKernel() {
    persistKernel<<<kNumThBlocks, kNumThPerBlock, 0, sg_stream>>>(
        get_fifo_vec());
    cudaCheckErrors("persistKernel failed");
}
