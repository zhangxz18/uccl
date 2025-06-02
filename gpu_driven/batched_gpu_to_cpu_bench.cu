/* batched_gpu_to_cpu_bench.cu
 *
 * make
 * CUDA_MODULE_LOADING=EAGER ./batched_gpu_to_cpu_bench
 * 
 * clock rate: 1410000 kHz
 * 
 * Per-block avg latency:
 *   Block 0 : 5.619 µs over 10000 ops
 *   Block 1 : 5.870 µs over 10000 ops
 *   Block 2 : 5.769 µs over 10000 ops
 *   Block 3 : 5.857 µs over 10000 ops
 *   Block 4 : 5.774 µs over 10000 ops
 *   Block 5 : 5.579 µs over 10000 ops
 *   Block 6 : 5.566 µs over 10000 ops
 *   Block 7 : 5.572 µs over 10000 ops

 * Overall avg GPU-measured latency  : 5.701 µs
 * End-to-end Wall-clock time        : 80.022 ms
 * Throughput                        : 1.00 Mops/s
 * 
 */

#include <assert.h>
#include <cuda_pipeline.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <immintrin.h>
#include <chrono>
#include <tuple>
#include <vector>
#include <thread>
#include <atomic>

static constexpr int kNumThBlocks = 8;
static constexpr int kNumThPerBlock = 1;
static constexpr int kIterations = 10000;
static constexpr int kBatchSize = 8; // Higher throughput but higher latency.

static constexpr uint32_t kQueueSize = 128;
static constexpr uint32_t kQueueMask = kQueueSize - 1;

#define MEASURE_PER_OP_LATENCY

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

struct alignas(128) Fifo {
    unsigned long long head; // Next slot to produce
    unsigned long long tail; // Next slot to consume
    volatile uint64_t buf[kQueueSize]; // Payload buffer (8 bytes). 
};

#ifdef MEASURE_PER_OP_LATENCY
__device__ unsigned long long cycle_accum[kNumThBlocks] = {0};
__device__ unsigned int op_count[kNumThBlocks] = {0};
#endif

__global__ void gpu_issue_batched_commands(Fifo *fifos) {

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    Fifo *my_fifo = &fifos[bid];
    if (tid != 0) {
        return;
    }

#ifdef MEASURE_PER_OP_LATENCY
    uint32_t complete = 0;
    __shared__ unsigned long long cycle_accum_smem;
    __shared__ unsigned int       op_count_smem;
    cycle_accum_smem = 0ull;
    op_count_smem    = 0u;
#endif

    extern __shared__ unsigned long long start_cycle_smem[];

    for (int it = 0; it < kIterations; it += kBatchSize)
    {
        uint32_t my_hdr;
        uint32_t cur_tail;

        const unsigned int todo = 
            (it + kBatchSize <= kIterations) ? kBatchSize :
            (kIterations - it);

        while (true) {
            uint32_t cur_head = my_fifo->head;
            cur_tail = my_fifo->tail;
            if (cur_head - cur_tail + todo <= kQueueSize) {
                if (atomicAdd_system(&(my_fifo->head), todo) == cur_head) {
                    my_hdr = cur_head; 
                    break; // Successfully reserved a slot
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < todo; ++i) {
            uint32_t idx = (my_hdr + i) & kQueueMask;
            unsigned long long t0 = clock64();
            uint64_t cmd = (static_cast<uint64_t>(bid) << 32) | (it + i + 1);

            start_cycle_smem[idx] = t0;
            my_fifo->buf[idx] = cmd;
        }
        __threadfence_system();
#ifdef MEASURE_PER_OP_LATENCY
        while (complete < my_hdr + todo) {
            uint32_t cidx = complete & kQueueMask;
            if (complete < my_fifo->tail) {
                unsigned long long t1 = clock64();
                unsigned long long cycles = t1 - start_cycle_smem[cidx];
                cycle_accum_smem += cycles;
                op_count_smem++;
                complete++;
            } else {
                break;
            }
        }
#endif
    }
#ifdef MEASURE_PER_OP_LATENCY
    while (complete < kIterations) {
        uint32_t cidx = complete & kQueueMask;
        while (complete >= my_fifo->tail) { /* spin */ }

        unsigned long long t1 = clock64();
        cycle_accum_smem += (t1 - start_cycle_smem[cidx]);
        ++op_count_smem;
        ++complete;
    }

    cycle_accum[bid] = cycle_accum_smem;
    op_count[bid] = op_count_smem;
#endif
}

static inline bool pin_thread_to_cpu(int cpu) {
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu < 0 || cpu >= num_cpus) return false;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);

    pthread_t current_thread = pthread_self();

    return !pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}

void cpu_consume(Fifo *fifo, int block_idx) {
    // printf("CPU thread for block %d started\n", block_idx);
    pin_thread_to_cpu(block_idx);

    uint64_t my_tail = 0;
    for (int seen = 0; seen < kIterations; ++seen) {
        while (fifo->head == my_tail)  { /* spin */ }
        uint64_t idx = my_tail & kQueueMask;
        uint64_t cmd;
        do { 
            cmd = fifo->buf[idx]; 
            _mm_pause();  // Avoid hammering the cacheline. 
        } while (cmd == 0);

        // printf("CPU thread for block %d, idx: %d, consuming cmd %llu\n", 
            //    block_idx, idx, static_cast<unsigned long long>(cmd));

        uint64_t expected_cmd = (static_cast<uint64_t>(block_idx) << 32) | (seen + 1);
        if (cmd != expected_cmd) {
            fprintf(stderr, "Error: block %d, expected cmd %llu, got %llu\n",
                    block_idx, static_cast<unsigned long long>(expected_cmd), 
                    static_cast<unsigned long long>(cmd));
            exit(1);
        }
        fifo->buf[idx] = 0;
        // std::atomic_thread_fence(std::memory_order_release);
        my_tail++;
        fifo->tail = my_tail;
    }
}


int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaCheckErrors("cudaStreamCreate failed");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("clock rate: %d kHz\n", prop.clockRate);

    Fifo *fifos;
    cudaHostAlloc(&fifos, sizeof(Fifo) * kNumThBlocks, cudaHostAllocMapped);

    for (int i = 0; i < kNumThBlocks; ++i) {
        fifos[i].head = 0;
        fifos[i].tail = 0;
        for (uint32_t j = 0; j < kQueueSize; ++j) {
            fifos[i].buf[j] = 0; // Initialize the buffer
        }
    }

    // Launch one CPU polling thread per block
    std::vector<std::thread> cpu_threads;
    for (int i = 0; i < kNumThBlocks; ++i) {
        cpu_threads.emplace_back(cpu_consume, &fifos[i], i);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t shmem_bytes = kQueueSize * sizeof(unsigned long long);
    gpu_issue_batched_commands<<<kNumThBlocks, kNumThPerBlock, shmem_bytes, stream1>>>(fifos);
    cudaCheckErrors("gpu_issue_command kernel failed");

    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");
    auto t1 = std::chrono::high_resolution_clock::now();

    for (auto& t : cpu_threads) {
        t.join();
    }

#ifdef MEASURE_PER_OP_LATENCY
    unsigned long long h_cycles[kNumThBlocks];
    unsigned int       h_ops[kNumThBlocks];
    cudaMemcpyFromSymbol(h_cycles,cycle_accum,sizeof(h_cycles));
    cudaMemcpyFromSymbol(h_ops,   op_count,   sizeof(h_ops));
#endif


    unsigned int tot_ops=0;
#ifdef MEASURE_PER_OP_LATENCY
    double total_us=0; 
    unsigned long long tot_cycles=0;
    printf("\nPer-block avg latency:\n");
    for(int b=0;b<kNumThBlocks;++b){
        double us = (double)h_cycles[b]*1000.0/prop.clockRate/h_ops[b];
        printf("  Block %d : %.3f µs over %u ops\n",b,us,h_ops[b]);
        total_us += us;
        tot_cycles += h_cycles[b]; tot_ops += h_ops[b];
    }
#else
    tot_ops = kNumThBlocks * kIterations;
#endif
    double wall_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    double throughput = (double)(kNumThBlocks*kIterations)/(wall_ms*1000.0);

#ifdef MEASURE_PER_OP_LATENCY
    printf("\nOverall avg GPU-measured latency  : %.3f µs\n",
           (double)tot_cycles*1000.0/prop.clockRate/tot_ops);
    printf("Total cycles                       : %llu\n", tot_cycles);
#endif
    printf("Total ops                          : %u\n", tot_ops);
    printf("End-to-end Wall-clock time        : %.3f ms\n", wall_ms);
    printf("Throughput                        : %.2f Mops/s\n", throughput);


    cudaFreeHost(fifos);
    cudaCheckErrors("cudaFreeHost failed");
    cudaStreamDestroy(stream1);
    cudaCheckErrors("cudaStreamDestroy failed");
}