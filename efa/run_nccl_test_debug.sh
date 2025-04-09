# !/bin/bash

TEST="uccl_rdma"
# TEST="uccl_rdma_zc"

LIBNCCL_PATH="/opt/${TEST}/nccl/build/lib/libnccl.so"
PLUGIN_PATH="/opt/${TEST}/efa/libnccl-net.so"
# BIN_PATH="/opt/${TEST}/nccl-tests/build/alltoall_perf"
BIN_PATH="/opt/${TEST}/nccl-tests/build/all_reduce_perf"

COPY_CHANNELS=1
if [ "$TEST" = "uccl_rdma_zc" ]; then
    COPY_CHANNELS=2
fi

mpirun --bind-to none -np 1 -N 1 --host localhost \
    --tag-output --merge-stderr-to-stdout \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ens32 \
    -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
    -x NCCL_P2P_DISABLE=1 \
    -x NCCL_SHM_DISABLE=1 \
    -x NCCL_NET_DISABLE=0 \
    -x CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    -x NCCL_GDRCOPY_FLUSH_ENABLE=1 \
    -x NCCL_PROTO=simple \
    -x NCCL_ALGO=Ring \
    -x NCCL_MAX_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_P2P_NET_CHUNKSIZE=131072 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_BUFFSIZE=1048576 \
    -x GLOG_logtostderr=0 \
    -x NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml \
    -x UCCL_ENGINE_QUIET=1 \
    -x CUDA_MODULE_LOADING=EAGER \
    ${BIN_PATH} \
    -b 1K -e 1M -f 2 -w 5 -n 10 -c 1 -g 1 -t 8 \
    >& alltoall_debug.log

    # -x NCCL_DEBUG=INFO \

# export LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_NET_DISABLE=0
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export NCCL_GDRCOPY_FLUSH_ENABLE=1
# export NCCL_PROTO=simple
# export NCCL_ALGO=Ring
# export NCCL_MAX_NCHANNELS=${COPY_CHANNELS}
# export NCCL_MIN_NCHANNELS=${COPY_CHANNELS}
# export NCCL_P2P_NET_CHUNKSIZE=131072
# export NCCL_NCHANNELS_PER_NET_PEER=1
# export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_BUFFSIZE=1048576
# export GLOG_logtostderr=0
# export NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml
# export UCCL_ENGINE_QUIET=1

# compute-sanitizer --tool memcheck ${BIN_PATH} -b 1K -e 1K -f 2 -w 0 -n 1 -g 1 -t 2 >& alltoall_debug.log

# sudo -E /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --output=nccl_test_report --gpu-metrics-device=0 \
#     ${BIN_PATH} -b 1K -e 1K -f 2 -w 0 -n 1 -g 1 -t 8

# gdb --args ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 2
