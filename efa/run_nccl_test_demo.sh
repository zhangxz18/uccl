# !/bin/bash

source ../shared.sh
NODES=$(get_nodes "../nodes.txt")

TEST=${1:-srd}

BIN=${2:-all_reduce_perf} # or alltoall_perf
BIN_PATH="/opt/uccl_rdma/nccl-tests/build/${BIN}"

NVLINK=0
NVLINK_DISABLE="$((1-NVLINK))"

if [ "$BIN" = "all_reduce_perf" ]; then
    if [ "$TEST" = "srd" ]; then
        LIBNCCL_PATH="/opt/uccl_rdma_zc/nccl/build/lib/libnccl.so"
        PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"
        COPY_CHANNELS=8
        CHANNELS_NET_PEER=1
        CHUNK_SIZE=524288
        BUFFSIZE=8388608 # AWS recommended value
        # BUFFSIZE=1048576
        # PXN_DISABLE=0 is critical for srd alltoall with nvlink performance.
        PXN_DISABLE=0
    elif [ "$TEST" = "ud" ]; then
        LIBNCCL_PATH="/opt/uccl_rdma/nccl/build/lib/libnccl.so"
        PLUGIN_PATH="/opt/uccl_rdma/efa/libnccl-net.so"
        # COPY_CHANNELS=8
        COPY_CHANNELS=4
        CHANNELS_NET_PEER=1
        CHUNK_SIZE=524288
        BUFFSIZE=1048576 # NCCL default
        PXN_DISABLE=0
    fi    
elif [ "$BIN" = "alltoall_perf" ]; then
    if [ "$TEST" = "srd" ]; then
        LIBNCCL_PATH="/opt/uccl_rdma_zc/nccl/build/lib/libnccl.so"
        PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"
        # This config yeilds the best avg bandwidth.
        COPY_CHANNELS=-1 # -1 means 32
        CHANNELS_NET_PEER=-1 # -1 means 2
        CHUNK_SIZE=524288
        BUFFSIZE=8388608
        PXN_DISABLE=0
    elif [ "$TEST" = "ud" ]; then
        # Need to set kSplitSendRecvEngine=true and recompile the plugin.
        LIBNCCL_PATH="/opt/uccl_rdma/nccl/build/lib/libnccl.so"
        PLUGIN_PATH="/opt/uccl_rdma/efa/libnccl-net.so"
        COPY_CHANNELS=8 # yeilds the best avg bandwidth
        CHANNELS_NET_PEER=4
        CHUNK_SIZE=262144
        BUFFSIZE=1048576
        PXN_DISABLE=0
    fi
fi

echo "Running ${BIN} with ${TEST} on nodes ${NODES} with ${COPY_CHANNELS} channels, ${CHANNELS_NET_PEER} channels per net peer, ${CHUNK_SIZE} chunk size, ${BUFFSIZE} buffer size, PXN_DISABLE=${PXN_DISABLE}" 

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
    -x NCCL_MAX_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${COPY_CHANNELS} \
    -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_PXN_DISABLE=${PXN_DISABLE} \
    -x GLOG_logtostderr=0 \
    -x NCCL_TOPO_FILE=/opt/uccl_rdma/efa/p4d-24xl-topo.xml \
    -x UCCL_ENGINE_QUIET=1 \
    -x NCCL_ALGO=Tree \
    ${BIN_PATH} \
    -b 1K -e 1G -f 2 -w 5 -n 10 -c 1 -g 1 -t 8

