#include "scattered_memcpy.cuh"

#include <cuda_runtime.h>
#include <string.h>

#include <chrono>
#include <iostream>

#include "transport_config.h"

#define TEST_RUNS 100

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    copy_param_t *h_params = new copy_param_t();

    // Allocate and initialize GPU memory for copies
    for (int i = 0; i < MAX_SCATTERED_COPIES; i++) {
        uint64_t size = EFA_MAX_PAYLOAD;
        h_params->size[i] = size;

        cudaMalloc((void **)&h_params->src[i], size);
        cudaMalloc((void **)&h_params->dst[i], size);

        // Initialize source memory with some data
        uint8_t *h_src = new uint8_t[size];
        memset(h_src, i % 256, size);
        cudaMemcpy((void *)h_params->src[i], h_src, size,
                   cudaMemcpyHostToDevice);
        delete[] h_src;
    }

    uint64_t total_bytes = 0;
    for (int i = 0; i < MAX_SCATTERED_COPIES; i++) {
        total_bytes += h_params->size[i];
    }

    // Warm up.
    launchScatteredMemcpy(h_params);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_RUNS; i++) {
        launchScatteredMemcpy(h_params);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           end_time - start_time)
                           .count();
    double time_us = duration_ns / 1000.0 / TEST_RUNS;
    double bandwidth_gbps = total_bytes / (time_us * 1e3);

    std::cout << "CUDA Scattered Memcpy Performance:" << std::endl;
    std::cout << "  Launch Time: " << time_us << " us" << std::endl;
    std::cout << "  Memory Bandwidth: " << bandwidth_gbps << " GB/s"
              << std::endl;

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < TEST_RUNS; i++) {
        launchScatteredMemcpyAsync(h_params, stream);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end_time - start_time)
                      .count();
    time_us = duration_ns / 1000.0 / TEST_RUNS;
    bandwidth_gbps = total_bytes / (time_us * 1e3);

    std::cout << "Async CUDA Scattered Memcpy Performance:" << std::endl;
    std::cout << "  Launch Time: " << time_us << " us" << std::endl;
    std::cout << "  Memory Bandwidth: " << bandwidth_gbps << " GB/s"
              << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    int finished_cnt = 0, poll_cnt = 0;
    while (finished_cnt < TEST_RUNS) {
        finished_cnt += pollScatteredMemcpy(stream);
        poll_cnt++;
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      end_time - start_time)
                      .count();
    time_us = duration_ns / 1000.0 / poll_cnt;
    std::cout << "  Polling Time: " << time_us << " us" << std::endl;

    for (int i = 0; i < MAX_SCATTERED_COPIES; i++) {
        cudaFree((void *)h_params->src[i]);
        cudaFree((void *)h_params->dst[i]);
    }

    return 0;
}
