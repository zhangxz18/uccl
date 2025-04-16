# !/bin/bash

set -e
set -x

TEST=${1:-srd}

UCCL_HOME="/opt/uccl_rdma"

NV_LINK_DISABLE=1
CHANNELS=4          # 8 for GPU scatter-gather copy
CHANNELS_NET_PEER=4 # 2/4/6/8 is okay, but 1 doesn't work
CHUNK_SIZE=524288   # best for UCCL
BUFFSIZE=8388608    # for UCCL to run allreduce
UCCL_QUITE=1

if [ "$TEST" = "srd" ]; then
    LD_PRELOAD="/opt/uccl_rdma_zc/nccl/build/lib/libnccl.so" \
        NCCL_NET_PLUGIN="/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu/libnccl-net.so" \
        NCCL_DEBUG=INFO \
        NCCL_PROTO=Simple \
        NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        NCCL_NET_DISABLE=0 \
        NCCL_MAX_NCHANNELS=${CHANNELS} \
        NCCL_MIN_NCHANNELS=${CHANNELS} \
        CUDA_VISIBLE_DEVICES=0,2,4,6 \
        NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        NCCL_BUFFSIZE=${BUFFSIZE} \
        torchrun --nproc_per_node=4 ddp2.py --batch_size 128 --epochs 10

elif [ "$TEST" = "ud" ]; then
    LD_PRELOAD="${UCCL_HOME}/nccl/build/lib/libnccl.so" \
        NCCL_NET_PLUGIN="${UCCL_HOME}/efa/libnccl-net.so" \
        NCCL_DEBUG=INFO \
        NCCL_PROTO=Simple \
        NCCL_P2P_DISABLE=${NV_LINK_DISABLE} \
        NCCL_SHM_DISABLE=${NV_LINK_DISABLE} \
        NCCL_NET_DISABLE=0 \
        NCCL_MAX_NCHANNELS=${CHANNELS} \
        NCCL_MIN_NCHANNELS=${CHANNELS} \
        CUDA_VISIBLE_DEVICES=0,2,4,6 \
        NCCL_NCHANNELS_PER_NET_PEER=${CHANNELS_NET_PEER} \
        NCCL_P2P_NET_CHUNKSIZE=${CHUNK_SIZE} \
        NCCL_BUFFSIZE=${BUFFSIZE} \
        NCCL_NET_GDR_LEVEL=SYS \
        CUDA_MODULE_LOADING=EAGER \
        NCCL_TOPO_FILE=${UCCL_HOME}/efa/p4d-24xl-topo.xml \
        NCCL_PXN_DISABLE=1 \
        UCCL_ENGINE_QUIET=${UCCL_QUITE} \
        GLOG_logtostderr=0 \
        torchrun --nproc_per_node=4 ddp2.py --batch_size 128 --epochs 10
        # python -m torch.distributed.run --nproc_per_node=4 ddp.py --batch_size 128 --epochs 10
        # gdb --args python -m torch.distributed.run --nproc_per_node=4 ddp.py --batch_size 128 --epochs 10
fi

        # NCCL_ALGO=Ring \