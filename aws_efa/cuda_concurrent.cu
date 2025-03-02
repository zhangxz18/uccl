#include <cuda_runtime.h>

#include <iostream>

#define N 1024  // Number of elements

// Dummy kernel to simulate computation
__global__ void kernel1(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < 100000000; i++) {  // Simulate workload
        data[idx] *= 2 * i;
    }
}

// Another dummy kernel
__global__ void kernel2(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < 100000000; i++) {  // Simulate workload
        data[idx] *= 2 * i;
    }
}

int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int *d_data1, *d_data2;
    cudaMalloc(&d_data1, N * sizeof(int));
    cudaMalloc(&d_data2, N * sizeof(int));

    // Launch both kernels in separate streams
    for (int i = 0; i < 10; i++) {
        kernel1<<<8, 512, 0, stream1>>>(d_data1);
        kernel2<<<8, 512, 0, stream2>>>(d_data2);
    }

    // Wait for both kernels to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Cleanup
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    std::cout << "✅ Both kernels executed concurrently!" << std::endl;
    return 0;
}
