# !/bin/bash

# TEST="uccl_rdma"
TEST="uccl_rdma_zc"

LIBNCCL_PATH="/opt/${TEST}/nccl/build/lib/libnccl.so"
PLUGIN_PATH="/opt/${TEST}/efa/libnccl-net.so"
BIN_PATH="/opt/${TEST}/nccl-tests/build/alltoall_perf"

COPY_CHANNELS=8
if [ "$TEST" = "uccl_rdma_zc" ]; then
    COPY_CHANNELS=2
fi

mpirun --bind-to none -np 1 -N 1 --host localhost \
    --tag-output --merge-stderr-to-stdout \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ens32 \
    -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
    -x NCCL_DEBUG=INFO \
    -x NCCL_P2P_DISABLE=1 \
    -x NCCL_SHM_DISABLE=1 \
    -x NCCL_NET_DISABLE=0 \
    -x CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    -x NCCL_MAX_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_P2P_NET_CHUNKSIZE=131072 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_BUFFSIZE=8388608 \
    -x GLOG_logtostderr=1 \
    -x NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml \
    -x UCCL_ENGINE_QUIET=1 \
    -x CUDA_MODULE_LOADING=EAGER \
    ${BIN_PATH} \
    -b 1K -e 1G -f 2 -w 5 -n 100 -g 1 -t 8 \
    >& alltoall_debug.log


# export LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}"
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_NET_DISABLE=0
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export NCCL_MAX_NCHANNELS=1
# export NCCL_MIN_NCHANNELS=1
# export NCCL_NET_GDR_LEVEL=SYS
# export NCCL_P2P_NET_CHUNKSIZE=524288
# export NCCL_BUFFSIZE=8388608
# export GLOG_logtostderr=1
# export NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml
# export UCCL_ENGINE_QUIET=1

# ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 8
# sudo -E /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --output=nccl_test_report --gpu-metrics-device=0 
# compute-sanitizer --tool memcheck ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 2 >& alltoall_debug.log
# gdb --args ${BIN_PATH} -b 1M -e 1M -f 2 -w 0 -n 1 -g 1 -t 2
