# /usr/bin/bash

# Usage ./run_nccl_test.sh [UCCL] [# of Nodes] [# of GPUs per node] 

UCCL=${1:-1}
NUM_PROCS=${2:-2}
NUM_GPUS_PER_NODE=${3:-8}

ROOT="/home/aleria/uccl_rdma"

# IP of Nodes.
NODES="87.120.213.6,87.120.213.5"
# Names of HCAs.
HCA_NAMES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
# Name of Control NIC.
CTRL_NIC="ens10f0np0"
# Path of NCCL
NCCL_PATH="${ROOT}/nccl/build/lib/libnccl.so"
# Path of UCCL
PLUGIN_PATH="${ROOT}/rdma/libnccl-net.so"

# Number of chunnels.
NUM_CHUNNEL=4
# Chunk size.
# 131072, 262144, 524288
P2P_NET_CHUNKSIZE=524288
# Buffer size.
BUFFSIZE=8388608
# Number of chunnels per NET peer.
CHANNELS_NET_PEER=-1
# Algorithm
# TREE, RING
ALGO=-1

# Multi-QP for NCCL.
NUM_QPS_PER_CONNECTION=4
SPLIT_DATA_ON_QPS=1

# all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf
# hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf
PROG_NAME=all_reduce_perf

if [ "$UCCL" -ne 1 ]; then
    PLUGIN_PATH=""
fi

echo "Running test: ${PROG_NAME}, $([ "${UCCL}" -eq 1 ] && echo "UCCL" || echo "NCCL"), ${NUM_PROCS} nodes, ${NUM_GPUS_PER_NODE} GPUs per node, $((NUM_PROCS * NUM_GPUS_PER_NODE)) GPUs in total."

echo "Details: ${NUM_CHUNNEL} channels, P2P_NET_CHUNKSIZE: ${P2P_NET_CHUNKSIZE}, BUFFSIZE: ${BUFFSIZE}, CHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER}, ALGORITHM=${ALGO}"

mpirun --bind-to none -np ${NUM_PROCS} -N 1 \
    --host ${NODES} \
    --mca btl_tcp_if_include ${CTRL_NIC} \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    -x NCCL_P2P_DISABLE=1 \
    -x NCCL_SHM_DISABLE=1 \
    -x NCCL_NET_DISABLE=0 \
    -x NCCL_ALGO=${ALGO} \
    -x NCCL_MAX_NCHANNELS=${NUM_CHUNNEL} \
    -x NCCL_MIN_NCHANNELS=${NUM_CHUNNEL} \
    -x NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
    -x NCCL_P2P_NET_CHUNKSIZE=${P2P_NET_CHUNKSIZE} \
    -x NCCL_BUFFSIZE=${BUFFSIZE} \
    -x NCCL_IB_QPS_PER_CONNECTION=${NUM_QPS_PER_CONNECTION} -x NCCL_IB_SPLIT_DATA_ON_QPS=${SPLIT_DATA_ON_QPS} \
    -x NCCL_IB_HCA=${HCA_NAMES} \
    -x LD_PRELOAD="${NCCL_PATH} ${PLUGIN_PATH}" \
    ${ROOT}/nccl-tests/build/${PROG_NAME} \
    -f 2 \
    --minbytes 1K --maxbytes 1G \
    --warmup_iters 50 --iters 100 \
    -n 1 -t ${NUM_GPUS_PER_NODE}
    
    # --map-by ppr:1:node:PE=32 \
    # --bind-to core \
    # -b 1K -e 1G -f 2 -g 1 -w 100 -n 100 -t 1
	# -C 1 -a 3
    # -x NCCL_P2P_DISABLE=1 \
	# -x NCCL_SHM_DISABLE=1 \
	# -x NCCL_ALGO=TREE \
    # -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
    # -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \
    # mpirun --bind-to none -np 2 -N 1 --host 87.120.213.6,87.120.213.7 \
    # -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    # -x NCCL_BUFFSIZE=8388608 \
    # /home/aleria/uccl_rdma/nccl-tests/build/all_reduce_perf\
    # /home/aleria/uccl_rdma/nccl-tests/build/alltoall_perf\
