#include <stdio.h>
#include <unistd.h>

#define ITERS 4
#define DSIZE 65536
#define nTPB 256

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

__device__ uint __smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ volatile int blkcnt1 = 0;
__device__ volatile int blkcnt2 = 0;
__device__ volatile int itercnt = 0;

__device__ void my_compute_function(int *buf, int idx, int data) {
    buf[idx] = data;  // put your work code here
}

__global__ void testkernel(int *buffer1, int *buffer2,
                           volatile int *buffer1_ready,
                           volatile int *buffer2_ready, const int buffersize,
                           const int iterations) {
    // assumption of persistent block-limited kernel launch
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int iter_count = 0;
    // persistent until iterations complete
    // while (iter_count < iterations) {
    while (true) {  // persistent
        int *buf =
            (iter_count & 1) ? buffer2 : buffer1;  // ping pong between buffers
        volatile int *bufrdy =
            (iter_count & 1) ? (buffer2_ready) : (buffer1_ready);
        volatile int *blkcnt = (iter_count & 1) ? (&blkcnt2) : (&blkcnt1);
        int my_idx = idx;
        while (iter_count - itercnt > 1);  // don't overrun buffers on device
        while (*bufrdy == 2);              // wait for buffer to be consumed
        printf("SM %d, block %d, thread %d\n", __smid(), blockIdx.x,
               threadIdx.x);
        while (my_idx < buffersize) {  // perform the "work"
            my_compute_function(buf, my_idx, iter_count);
            my_idx += gridDim.x * blockDim.x;  // grid-striding loop
        }
        __syncthreads();  // wait for my block to finish
        __threadfence();  // make sure global buffer writes are "visible"
        if (!threadIdx.x) atomicAdd((int *)blkcnt, 1);  // mark my block done
        if (!idx) {                       // am I the master block/thread?
            while (*blkcnt < gridDim.x);  // wait for all blocks to finish
            *blkcnt = 0;
            *bufrdy = 2;             // indicate that buffer is ready
            __threadfence_system();  // push it out to mapped memory
            itercnt++;
        }
        iter_count++;
    }
}

__global__ void emptykernel() {
    // do nothing
    printf("Emptykernel: SM %d, block %d, thread %d\n", __smid(), blockIdx.x,
           threadIdx.x);
}

int validate(const int *data, const int dsize, const int val) {
    for (int i = 0; i < dsize; i++)
        if (data[i] != val) {
            printf("mismatch at %d, was: %d, should be: %d\n", i, data[i], val);
            return 0;
        }
    return 1;
}

// code from:
// https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste

int main() {
    int device;
    cudaGetDevice(&device);

    int concurrentKernels;
    cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels,
                           device);

    if (concurrentKernels) {
        printf("✅ GPU supports concurrent kernel execution!\n");
    } else {
        printf("❌ GPU does NOT support concurrent kernel execution.\n");
    }

    int *h_buf1, *d_buf1, *h_buf2, *d_buf2;
    volatile int *m_bufrdy1, *m_bufrdy2;
    // buffer and "mailbox" setup
    cudaHostAlloc(&h_buf1, DSIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_buf2, DSIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&m_bufrdy1, sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc(&m_bufrdy2, sizeof(int), cudaHostAllocMapped);
    cudaCheckErrors("cudaHostAlloc fail");
    cudaMalloc(&d_buf1, DSIZE * sizeof(int));
    cudaMalloc(&d_buf2, DSIZE * sizeof(int));
    cudaCheckErrors("cudaMalloc fail");
    cudaStream_t streamk, streamc;
    cudaStreamCreate(&streamk);
    cudaStreamCreate(&streamc);
    cudaCheckErrors("cudaStreamCreate fail");
    *m_bufrdy1 = 0;
    *m_bufrdy2 = 0;
    cudaMemset(d_buf1, 0xFF, DSIZE * sizeof(int));
    cudaMemset(d_buf2, 0xFF, DSIZE * sizeof(int));
    cudaCheckErrors("cudaMemset fail");
    // inefficient crutch for choosing number of blocks
    // int nblock = 0;
    // cudaDeviceGetAttribute(&nblock, cudaDevAttrMultiProcessorCount, 0);
    // cudaCheckErrors("get multiprocessor count fail");
    int nblock = 1;
    testkernel<<<nblock, nTPB, 0, streamk>>>(d_buf1, d_buf2, m_bufrdy1,
                                             m_bufrdy2, DSIZE, ITERS);
    cudaCheckErrors("kernel launch fail");

    cudaStream_t streame;
    cudaStreamCreate(&streame);
    cudaCheckErrors("cudaStreamCreate fail");
    int cnt = 0;
    while (cnt++ < 1000) emptykernel<<<nblock, nTPB, 0, streame>>>();
    cudaCheckErrors("kernel launch fail");
    cudaStreamSynchronize(streame);
    cudaCheckErrors("cudaStreamSync fail");

    volatile int *bufrdy;
    int *hbuf, *dbuf;
    for (int i = 0; i < ITERS; i++) {
        if (i & 1) {  // ping pong on the host side
            bufrdy = m_bufrdy2;
            hbuf = h_buf2;
            dbuf = d_buf2;
        } else {
            bufrdy = m_bufrdy1;
            hbuf = h_buf1;
            dbuf = d_buf1;
        }
        // int qq = 0; // add for failsafe - otherwise a machine failure can
        // hang
        while ((*bufrdy) != 2);  // use this for a failsafe:  if (++qq >
                                 // 1000000) {printf("bufrdy = %d\n", *bufrdy);
                                 // return 0;} // wait for buffer to be full;
        cudaMemcpyAsync(hbuf, dbuf, DSIZE * sizeof(int), cudaMemcpyDeviceToHost,
                        streamc);
        cudaStreamSynchronize(streamc);
        cudaCheckErrors("cudaMemcpyAsync fail");
        *bufrdy = 0;  // release buffer back to device
        if (!validate(hbuf, DSIZE, i)) {
            printf("validation failure at iter %d\n", i);
            exit(1);
        }
    }
    printf("Completed %d iterations successfully\n", ITERS);
}
