# zhongjie: I found some configurations are overwritten by examples/run_pretrain.sh
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-10.42.24.1} # 011
# export MASTER_ADDR=${MASTER_ADDR:-10.42.22.1} # 051
export NNODES=${NNODES:-4}
export COMM_BACKEND=${COMM_BACKEND:-uccl}  # rccl or uccl
export NVLINK_OFF=${NVLINK_OFF:-1}
export VLOG=${VLOG:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
# export LD_PRELOAD=~/source/uccl_zxz/thirdparty/rccl/build_new/librccl.so

# Common settings
export NCCL_IB_HCA="rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1"
export NCCL_P2P_DISABLE=${NVLINK_OFF}
export NCCL_SHM_DISABLE=${NVLINK_OFF}
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_BUFFSIZE=8388608
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-4}
# export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_SOCKET_IFNAME="cni0"
export NCCL_PROTO=Simple
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TOPO_DUMP_FILE="${NCCL_TOPO_DUMP_FILE:-}"

# export PATH=/root/anaconda3/bin:${PATH}
# ---------------- UCCL-specific ----------------
if [ "$COMM_BACKEND" = "uccl" ]; then
    export UCCL_HOME=${UCCL_HOME:-/root/source/uccl}
    export CONDA_LIB_HOME=/root/anaconda3/lib
    export GLOG_v=${VLOG}
    export LD_LIBRARY_PATH="${CONDA_LIB_HOME}:${LD_LIBRARY_PATH}"
    export NCCL_NET_PLUGIN="${UCCL_HOME}/rdma/librccl-net-uccl.so"
    export UCCL_NUM_ENGINES=4
    export UCCL_PORT_ENTROPY=32
    export UCCL_CHUNK_SIZE_KB=32
    export UCCL_RCMODE=1
    echo "UCCL is used"
else
    # ---------------- RCCL-specific ----------------
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
    export NCCL_IB_SPLIT_DATA_ON_QPS=0
    export NCCL_IB_PCI_RELAXED_ORDERING=1
    echo "RCCL is used"
fi

# MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=12335 NNODES=${NNODES} NODE_RANK=${NODE_RANK} HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPUS_PER_NODE=8 EXP=examples/megatron/exp_pretrain.yaml DATA_PATH=~/data/ HF_TOKEN=hf_PNQQiaKtqVUrwTgXmHCzXvbnceXGIcBrY bash examples/run_pretrain.sh > ~/source/uccl_test_result/${COMM_BACKEND}_channel${NCCL_MIN_NCHANNELS}_qp${NCCL_IB_QPS_PER_CONNECTION}_rank${NODE_RANK}.ansi 2>&1

MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=12338 NNODES=${NNODES} NODE_RANK=${NODE_RANK} HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPUS_PER_NODE=4 EXP=/root/source/uccl_zxz/primus_test/exp_pretrain.yaml DATA_PATH=~/data/ HF_TOKEN=hf_PNQQiaKtqVUrwTgXmHCzXvbnceXGIcBrY bash examples/run_pretrain.sh
