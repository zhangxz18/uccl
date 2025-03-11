# !/bin/bash

source ../shared.sh

# Usage: ./run_nccl_test.sh [srd|ud] [num of processes] [uccl quite] [ens32] [eqds]

TEST=${1:-srd}
UCCL_HOME="/opt/uccl_rdma"
LIBNCCL_PATH="${UCCL_HOME}/nccl/build/lib/libnccl.so"
# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
PROG_NAME=alltoall_perf
NUM_PROCS=${2:-4}
UCCL_QUITE=${3:-1}
NIC=${4:-ens32}
EQDS=${5:-eqds}
NODES=$(get_nodes "../nodes.txt")
GPU=4

echo "Running test: ${TEST}, ${PROG_NAME}, ${NUM_PROCS} processes, NIC ${NIC}, uccl_quite ${UCCL_QUITE}, ${NODES}"

if [ "$TEST" = "srd" ]; then

    # PLUGIN_PATH="${UCCL_HOME}/nccl/ext-net/google-fastsocket/libnccl-net.so"
    PLUGIN_PATH="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so"

    mpirun --bind-to none -np ${NUM_PROCS} -N 1 --host ${NODES} \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        -x NCCL_P2P_DISABLE=1 \
        -x NCCL_SHM_DISABLE=1 \
        -x NCCL_NET_DISABLE=0 \
        -x NCCL_MAX_NCHANNELS=2 \
        -x NCCL_MIN_NCHANNELS=2 \
        -x NCCL_P2P_NET_CHUNKSIZE=524288 \
        -x NCCL_BUFFSIZE=8388608 \
        ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
        -b 1K -e 1G -f 2 -c 1 -w 100 -n 100 -t ${GPU} -g 1

        # -x CUDA_VISIBLE_DEVICES="0,1" \
        # -x NCCL_ALGO=Ring \
        # -x NCCL_SOCKET_NTHREADS=4 \
        # -x NCCL_NSOCKS_PERTHREAD=2 \
        # -x NCCL_MAX_NCHANNELS=8 \
        # -x NCCL_MIN_NCHANNELS=8 \

elif [ "$TEST" = "ud" ]; then

    # Clear existing files for all ranks
    for ((rank = 0; rank < NUM_PROCS; rank++)); do
        >"output_rank_$rank.log" # Truncate or create empty file
    done

    if [ "$EQDS" = "eqds" ] ; then
        PLUGIN_PATH="/opt/zhongjie/uccl_rdma/efa/libnccl-net.so"
    else
        PLUGIN_PATH="${UCCL_HOME}/efa/libnccl-net.so"
    fi

    mpirun --bind-to none -np ${NUM_PROCS} -N 1 --host ${NODES} \
        --tag-output --merge-stderr-to-stdout \
        --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
        --mca orte_base_help_aggregate 0 \
        --mca btl_tcp_if_include ${NIC} \
        -x LD_PRELOAD="${LIBNCCL_PATH} ${PLUGIN_PATH}" \
        -x NCCL_DEBUG=INFO \
        -x NCCL_P2P_DISABLE=1 \
        -x NCCL_SHM_DISABLE=1 \
        -x NCCL_NET_DISABLE=0 \
        -x GLOG_logtostderr=0\
        -x CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
        -x NCCL_MAX_NCHANNELS=2 \
        -x NCCL_MIN_NCHANNELS=2 \
        -x NCCL_NET_GDR_LEVEL=SYS \
        -x NCCL_P2P_NET_CHUNKSIZE=524288 \
        -x NCCL_BUFFSIZE=8388608 \
        -x CUDA_MODULE_LOADING=EAGER \
        -x NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml \
        -x UCCL_ENGINE_QUIET=${UCCL_QUITE} \
        ${UCCL_HOME}/nccl-tests/build/${PROG_NAME} \
        -b 8M -e 256M -f 2 -c 0 -w 50 -n 100 -t ${GPU} -g 1 \
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
    done
else
    echo "Invalid test: ${TEST}"
    exit 1
fi
