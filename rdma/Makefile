ABS_ROOT_PATH := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

KERNEL = $(shell uname -r)
INC = -I ../lib/install/include
LIBS = ../lib/install/lib/libxdp.a ../lib/install/lib/libbpf.a -lglog -lgflags -lgtest -lz -lelf
LIBS_SHARED = -Wl,-rpath=$(ABS_ROOT_PATH)/../lib/xdp-tools/lib/libxdp/ \
	   		  -Wl,-rpath=$(ABS_ROOT_PATH)/../lib/libbpf/src/ \
	   		  -L../lib/xdp-tools/lib/libxdp -lxdp -L../lib/libbpf/src -lbpf \
	   		  -lglog -lgflags -lgtest -lz -lelf
override CXXFLAGS += -O3 -g -std=c++17 -Wno-pointer-arith -Wno-interference-size -fPIC
CLANGFLAGS = -O3 -g
DEPS = *.h
PLUGIN_SO = libnccl-net.so

lib_src = $(wildcard *.cc)
lib_src := $(filter-out %_main.cc,$(lib_src))
lib_src := $(filter-out %_test.cc,$(lib_src))
lib_src := $(filter-out %_plugin.cc,$(lib_src))
lib_obj = $(lib_src:.cc=.o)

test_src = $(wildcard *_test.cc)
test_bin = $(test_src:.cc=)

.PHONY: build
build: $(test_bin) $(lib_obj) $(PLUGIN_SO)

rdma_test: rdma_test.cc $(DEPS) $(lib_obj)
	g++ rdma_test.cc -o rdma_test $(lib_obj) $(INC) $(LIBS) $(CXXFLAGS)

%.o: %.cc $(DEPS)
	g++ -c $< -o $@ $(INC) $(CXXFLAGS)

NCCL_HOME:=../nccl
CUDA_HOME:=/usr/local/cuda
NCCL_INC:= -I$(NCCL_HOME)/build/include -I$(NCCL_HOME)/src/include -I$(CUDA_HOME)/include

$(PLUGIN_SO): nccl_plugin.cc $(DEPS) $(lib_obj)
	g++ $(NCCL_INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) nccl_plugin.cc $(lib_obj) $(INC) $(LIBS_SHARED) $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(main_bin) $(PLUGIN_SO)
