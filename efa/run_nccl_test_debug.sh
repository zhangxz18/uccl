# !/bin/bash

source ../shared.sh
NODES=$(get_nodes "../nodes.txt")

TEST="uccl_rdma"
# TEST="uccl_rdma_zc"
# TEST="srd"

# BIN_PATH="/opt/uccl_rdma/nccl-tests/build/alltoall_perf"
BIN_PATH="/opt/uccl_rdma/nccl-tests/build/all_reduce_perf"

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

LIBNCCL_PATH="/opt/${TEST}/nccl/build/lib/libnccl.so"
PLUGIN_PATH="/opt/${TEST}/efa/libnccl-net.so"

# This works best for allreduce (tree and ring) on 8/32 GPUs.
COPY_CHANNELS=8
CHANNELS_NET_PEER=1

# This works best for alltoall on 8 GPUs.
# COPY_CHANNELS=8
# CHANNELS_NET_PEER=1

# For allreduce with nvlink, use larger buffer to catch up with NCCL-SRD with larger buffers, and avoid performance outliers.
# CHUNK_SIZE=524288
# BUFFSIZE=8388608

# For others, just use default buffer size.
# CHUNK_SIZE=131072
# BUFFSIZE=1048576
CHUNK_SIZE=524288
BUFFSIZE=8388608

NVLINK_DISABLE=0

if [ "$TEST" = "srd" ]; then
    LIBNCCL_PATH="/opt/uccl_rdma_zc/nccl/build/lib/libnccl.so"
    PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"
    CHUNK_SIZE=524288
    BUFFSIZE=8388608
fi

mpirun --bind-to none -np 4 -N 1 --host ${NODES} \
    --tag-output --merge-stderr-to-stdout \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ens32 \
    -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
    -x NCCL_P2P_DISABLE=${NVLINK_DISABLE} \
    -x NCCL_SHM_DISABLE=${NVLINK_DISABLE} \
    -x NCCL_NET_DISABLE=0 \
    -x CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    -x NCCL_PROTO=Simple \
    -x NCCL_ALGO=Tree \
    -x NCCL_MAX_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x GLOG_logtostderr=0 \
    -x NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml \
    -x UCCL_ENGINE_QUIET=1 \
    ${BIN_PATH} \
    -b 1K -e 1G -f 2 -w 5 -n 10 -c 1 -g 1 -t 8 \
    >&alltoall_debug.log

    # gdb -ex run --args \
    # -x NCCL_DEBUG=INFO \
    # -x CUDA_MODULE_LOADING=EAGER \
    # -x NCCL_GDRCOPY_FLUSH_ENABLE=1 \

# export LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}"
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_NET_DISABLE=0
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export NCCL_GDRCOPY_FLUSH_ENABLE=1
# export NCCL_PROTO=Simple
# export NCCL_ALGO=Tree
# export NCCL_MAX_NCHANNELS=${COPY_CHANNELS}
# export NCCL_MIN_NCHANNELS=${COPY_CHANNELS}
# export NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE}
# export NCCL_BUFFSIZE=${BUFFSIZE}
# export NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER}
# export NCCL_NET_GDR_LEVEL=SYS
# export GLOG_logtostderr=0
# export NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml
# export UCCL_ENGINE_QUIET=1

# compute-sanitizer --tool memcheck ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 2 >& alltoall_debug.log

# sudo -E /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --output=nccl_test_report --gpu-metrics-device=0 \
#     ${BIN_PATH} -b 1K -e 1K -f 2 -w 0 -n 1 -g 1 -t 8

# gdb --args ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 2

# export NCCL_DEBUG=INFO
