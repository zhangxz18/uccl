#include <assert.h>
#include <cuda_pipeline.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <tuple>
#include <vector>
#include <thread>

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

struct GPUSignal {
    volatile uint64_t cmd;
    volatile uint64_t ack;
};


__global__ void gpu_issue_command(GPUSignal *signal, int iterations) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < iterations; i++) {
            unsigned long long start = clock64();
            signal->cmd = i + 1;  

            while (signal->ack != (i+1)) {
                __nanosleep(10);
            }
            unsigned long long end = clock64();
            printf("Command %d issued, acked in %llu cycles\n", i + 1, end - start);
        }
    }
}



void cpu_polling(GPUSignal *signal, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        while (signal->cmd != (i + 1)) {
            std::this_thread::yield();
        }
        signal->ack = i + 1; 
    }
}

// make -j
// CUDA_MODULE_LOADING=EAGER ./gpu_to_cpu_bench
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaCheckErrors("cudaStreamCreate failed");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("clock rate: %d\n", prop.clockRate);

    GPUSignal *signal;
    cudaHostAlloc(&signal, sizeof(GPUSignal), cudaHostAllocMapped);
    signal->cmd = 0;
    signal->ack = 0;

    int iterations = 1000;
    std::thread cpu_thread(cpu_polling, signal, iterations);

    gpu_issue_command<<<1, 32, 0, stream1>>>(signal, iterations);
    cudaCheckErrors("gpu_issue_command kernel failed");
    cudaStreamSynchronize(stream1);
    cudaCheckErrors("cudaStreamSynchronize failed");

    cpu_thread.join();

    cudaFreeHost(signal);
    cudaCheckErrors("cudaFreeHost failed");
    cudaStreamDestroy(stream1);
    cudaCheckErrors("cudaStreamDestroy failed");

}
