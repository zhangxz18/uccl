# UCCL-RDMA

RDMA support for UCCL.

1. UCCL supports two network fabrics: RoCE, Infiniband.
2. UCCL supports two modes: Unreliable Connection (UC) and Reliable Connection (RC).

## Install dependencies
```
sudo apt-get install libibverbs-dev -y
```

## Configuration
### transport_config.h:

1. Network
```
ROCE_NET:               True (RoCE) or false (Infiniband)

SINGLE_IP:              The IP address of control NIC. Set to empty string if each NIC has its own IP address. UCCL will detect them atomically.
```

2. NIC
```
NUM_DEVICES:            The number of physical NICs.

IB_DEVICE_NAME_PREFIX:  The prefix of the device name.

DEVNAME_SUFFIX_LIST:    The suffix of the device name.

LINK_BANDWIDTH:         The bandwidth of each NIC (Bytes per second).
```
### run_nccl_test.sh:
```
ROOT:                   The root directory of the workspace.
NODES:                  IP list of all nodes.
CTRL_NIC:               The name of control NIC.
NCCL_PATH:              The path of libnccl.so
PLUGIN_PATH:            The path of libnccl-net.so

Usage: ./run_nccl_test.sh [NCCL/UCCL: 0/1, default:1] [# of Nodes, default:2] [# of GPUs per node, default:8] [allreduce/alltoall: 0/1]
```

## Build
```
make -j`nproc`
```
