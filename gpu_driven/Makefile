CUDA_HOME:=/usr/local/cuda
CXXFLAGS = -I /opt/amazon/efa/include -L /opt/amazon/efa/lib -I$(CUDA_HOME)/include -L ${CUDA_HOME}/lib64 -lfabric -libverbs -lefa -lcudart -lpthread -lcuda -std=c++17
CUDAFLAGS = -arch=sm_80

all: pcie_bench

pcie_bench: pcie_bench.cu
	nvcc -O3 -g pcie_bench.cu -o pcie_bench $(CXXFLAGS) $(CUDAFLAGS)

clean:
	rm $(CUDA_BIN)