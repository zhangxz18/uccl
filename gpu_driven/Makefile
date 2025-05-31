CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lfabric -libverbs -lefa -lcudart -lpthread -lcuda -std=c++17
CUDAFLAGS = -arch=sm_80

all: pcie_bench gpu_to_cpu_bench

pcie_bench: pcie_bench.cu
	nvcc -O3 -g pcie_bench.cu -o pcie_bench $(CXXFLAGS) $(CUDAFLAGS)

gpu_to_cpu_bench: gpu_to_cpu_bench.cu
	nvcc -O3 -g gpu_to_cpu_bench.cu -o gpu_to_cpu_bench $(CXXFLAGS) $(CUDAFLAGS)

clean:
	rm $(CUDA_BIN)