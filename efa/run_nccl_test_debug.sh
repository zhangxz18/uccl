# !/bin/bash

LIBNCCL_PATH="/opt/uccl_rdma/nccl/build/lib/libnccl.so"
PLUGIN_PATH="/opt/uccl_rdma/efa/libnccl-net.so"

export LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}"
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET_DISABLE=0
export NCCL_MAX_NCHANNELS=2
export NCCL_MIN_NCHANNELS=2
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export GLOG_logtostderr=1
export UCCL_ENGINE_QUIET=1

# sudo -E /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --output=nccl_test_report \
#     --gpu-metrics-device=0 

# /opt/uccl_rdma/nccl-tests/build/alltoall_perf -b 1M -e 1M -f 2 -g 1 -w 0 -n 1 -t 2

# /opt/uccl_rdma/nccl-tests/build/alltoall_perf -b 1M -e 1M -f 2 -g 1 -w 0 -n 1 -t 2 >& alltoall_debug.log

compute-sanitizer --tool memcheck /opt/uccl_rdma/nccl-tests/build/alltoall_perf -b 1M -e 1M -f 2 -g 1 -w 0 -n 1 -t 2 >& alltoall_debug.log

# gdb --args /opt/uccl_rdma/nccl-tests/build/alltoall_perf -b 1M -e 1M -f 2 -g 1 -w 0 -n 1 -t 2
