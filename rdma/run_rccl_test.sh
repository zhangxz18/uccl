# !/bin/bash

# sinfo
# squeue --me

# salloc -N 4 -n 4 -p mi2104x -t 00:30:00

# mi2508x has better PCIe switch connection to achieve 200G between GPUs and NICs.
#   GPU 0,1 <-> mlx5_0
#   GPU 6,7 <-> mlx5_2
# salloc -N 4 -n 4 -p mi2508x -t 00:30:00

NODEFILE=node.txt
scontrol show hostnames $SLURM_JOB_NODELIST >$NODEFILE

ROOT=/home1/yangzhou/uccl_rdma

TEST=${1:-uccl}

if [ "$TEST" = "rccl" ]; then
    echo "Running RCCL test"
    plugin_path=""
elif [ "$TEST" = "uccl" ]; then
    echo "Running UCCL test"
    plugin_path="${ROOT}/rdma/librccl-net-uccl.so"
else
    echo "Unsupport benchmark type."
    exit 1
fi

mpirun --bind-to none -np 2 -N 1 --hostfile $NODEFILE --map-by ppr:1:node \
    -x LD_LIBRARY_PATH=${ROOT}/rccl/build/release:/work1/yzhou/yangzhou/anaconda3/lib:/opt/rocm-6.3.1/lib:${LD_LIBRARY_PATH} \
    -x NCCL_NET_PLUGIN=${plugin_path} \
    -x NCCL_DEBUG=INFO \
    -x NCCL_P2P_DISABLE=1 \
    -x NCCL_SHM_DISABLE=1 \
    -x NCCL_NET_DISABLE=0 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_IB_QPS_PER_CONNECTION=1 \
    -x HIP_VISIBLE_DEVICES=6 \
    -x NCCL_IB_HCA="mlx5_2:1" \
    ${ROOT}/rccl-tests/build/all_reduce_perf \
    -b 1K -e 1G -f 2 -w 5 -n 10 -c 1 -g 1 -t 1 |&
    tee alltoall_debug.log

# -x NCCL_DEBUG=INFO \

# On mi2104x
# -x NCCL_NET_GDR_LEVEL=SYS \
# -x HIP_VISIBLE_DEVICES=0 \
# -x NCCL_IB_HCA="mlx5_0:1" \

# On mi2508x
# -x HIP_VISIBLE_DEVICES=0,1,6,7 \
# -x NCCL_IB_HCA="mlx5_0:1,mlx5_2:1" \

# Setting to 4 will significantly degrade alltoall perf with 32 channels.
# -x NCCL_IB_QPS_PER_CONNECTION=1 \

# Default has 4 channels and 1 channel per net peer.
# -x NCCL_MAX_NCHANNELS=32 \
# -x NCCL_MIN_NCHANNELS=32 \
# -x NCCL_NCHANNELS_PER_NET_PEER=1 \

# -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
# -x RCCL_MSCCL_FORCE_ENABLE=1 \
# -x RCCL_MSCCLPP_ENABLE=1 \
# -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
# -x NCCL_PROTO=Simple \
# -x NCCL_P2P_NET_CHUNKSIZE=524288 \
# -x NCCL_BUFFSIZE=8388608 \
# all_reduce_perf, alltoall_perf, sendrecv_perf
