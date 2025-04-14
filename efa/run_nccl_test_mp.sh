# !/bin/bash

source ../shared.sh

# Run nccl-tests with multiple processes

# Usage: ./run_nccl_test_mp.sh [srd|ud] [Total Processes/Ranks/GPUs] [Benchtype, 0: allgather, 1: multi-allreduce]

UCCL_HOME="/opt/zhongjie/uccl_rdma"
LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"

# Visible GPUs to application.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Disable NVLink.
NV_LINK_DISABLE=1
MULTI_GROUP=0
NIC=ens32
# Processes/Ranks/GPUs per node.
PROCS_PER_NODE=8

TEST=${1:-srd}
NUM_PROCS=${2:-32}
PROG_NAME=${3:-0}

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
if [ "$PROG_NAME" -eq 0 ]; then
    PROG_NAME="all_gather_perf"
elif [ "$PROG_NAME" -eq 1 ]; then
    PROG_NAME="all_reduce_perf"
    MULTI_GROUP=0x7
else
    echo "Invalid program name: ${PROG_NAME}"
    exit 1
fi

CHANNELS=4
CHANNELS_NET_PEER=4

# UCCL optimal parameters.
# yangzhou: for allreduce with nvlink, we need to use larger buffer to catch up with NCCL with larger buffers, and avoid outliers. 
# 131072 262144 524288 1048576
CHUNK_SIZE=131072
BUFFSIZE=1048576

if [ "$TEST" = "srd" ]; then
    # SRD optimal parameters.
    CHUNK_SIZE=524288
    BUFFSIZE=8388608
fi

NODES=$(get_nodes "../nodes.txt")
echo "Running test: ${TEST}, ${PROG_NAME}, ${NUM_PROCS} processes, NIC ${NIC}, uccl_quite ${UCCL_QUITE}, ${NODES}, ${CHANNELS} channels."

if [ "$TEST" = "srd" ]; then

    # Clear existing files for all ranks
    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"output_rank_$rank.log" # Truncate or create empty file
    done

    # PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"
    PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N ${PROCS_PER_NODE} --hostfile hostname \
    --tag-output --merge-stderr-to-stdout \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ${NIC} \
    -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
    -x NCCL_PROTO=Simple \
    -x NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
    -x NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
    -x NCCL_NET_DISABLE=0 \
    -x NCCL_MAX_NCHANNELS=${CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${CHANNELS} \
    -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    -x NCCL_TESTS_SPLIT_MASK=${MULTI_GROUP} \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
    -b 1K -e 1G -f 2 -c 1 -w 50 -n 50 -t 1 -g 1 \
    2>&1 | while read -r line; do
    # Extract rank from the format [1,2]
    if [[ "$line" =~ ^\[[0-9]+,([0-9]+)\](.+) ]]; then
        RANK=${BASH_REMATCH[1]}                   # Extract second number as rank
        CONTENT=${BASH_REMATCH[2]}                # Extract the remaining content
        echo "Rank $RANK: $CONTENT"               # Print to terminal
        echo "$CONTENT" >>"output_rank_$RANK.log" # Append to rank-specific file
    else
        echo "$line" # Print untagged output to the terminal
    fi

    # -x CUDA_VISIBLE_DEVICES="0,1" \
    # -x NCCL_ALGO=Ring \
    # -x NCCL_SOCKET_NTHREADS=4 \
    # -x NCCL_NSOCKS_PERTHREAD=2 \
    # -x NCCL_MAX_NCHANNELS=8 \
    # -x NCCL_MIN_NCHANNELS=8 \
    done

elif [ "$TEST" = "ud" ]; then

    # Clear existing files for all ranks
    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"output_rank_$rank.log" # Truncate or create empty file
    done

    PLUGIN_PATH="${UCCL_HOME}/efa/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N ${PROCS_PER_NODE} --hostfile hostname \
    --tag-output --merge-stderr-to-stdout \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    --mca btl_tcp_if_include ${NIC} \
    -x NCCL_NET_PLUGIN=${PLUGIN_PATH} \
    -x LD_PRELOAD="${LIBNCCL_PATH}" \
    -x NCCL_PROTO=Simple \
    -x NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
    -x NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
    -x NCCL_NET_DISABLE=0 \
    -x GLOG_logtostderr=0 \
    -x CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    -x NCCL_TESTS_SPLIT_MASK=${MULTI_GROUP} \
    -x NCCL_MAX_NCHANNELS=${CHANNELS} \
    -x NCCL_MIN_NCHANNELS=${CHANNELS}  \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    -x CUDA_MODULE_LOADING=EAGER \
    -x NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml \
    -x NCCL_PXN_DISABLE=1 \
    -x UCCL_ENGINE_QUIET=1 \
    ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
    -b 1K -e 1G -f 2 -c 0 -w 50 -n 50 -t 1 -g 1 \
    2>&1 | while read -r line; do
    # Extract rank from the format [1,2]
    if [[ "$line" =~ ^\[[0-9]+,([0-9]+)\](.+) ]]; then
        RANK=${BASH_REMATCH[1]}                   # Extract second number as rank
        CONTENT=${BASH_REMATCH[2]}                # Extract the remaining content
        echo "Rank $RANK: $CONTENT"               # Print to terminal
        echo "$CONTENT" >>"output_rank_$RANK.log" # Append to rank-specific file
    else
        echo "$line" # Print untagged output to the terminal
    fi

        # gdb -ex run --args \
        # -x NCCL_ALGO=Ring \
        # -x NCCL_IB_CUDA_SUPPORT=1 \
        # -x NCCL_SOCKET_NTHREADS=4 \
        # -x NCCL_NSOCKS_PERTHREAD=2 \
        # -x NCCL_MAX_NCHANNELS=8 \
        # -x NCCL_MIN_NCHANNELS=8 \
        # -x NCCL_BUFFSIZE=8388608 \
        # -x NCCL_BUFFSIZE=1048576 \

        # -x NCCL_DEBUG=INFO \
        # -x NCCL_DEBUG_SUBSYS=INIT \
        # -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    done
else
    echo "Invalid test: ${TEST}"
    exit 1
fi
