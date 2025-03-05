# !/bin/bash

sudo /usr/local/cuda/bin/nsys profile --trace=cuda,nvtx,osrt --output=cuda_concurrent_report \
    --gpu-metrics-device=0 /opt/uccl_rdma/aws_efa/cuda_concurrent