# UCCL-RDMA

RDMA support for UCCL.

## Install dependencies
```
sudo apt-get install libibverbs-dev -y
```

## Configuration
### For transport_config.h:
1. Comment out CLOUDLAB_DEV

2. set USE_ROCE to true or false (Infiniband)

3. set NUM_DEVICES

4. set DEVNAME_SUFFIX_LIST, which is the suffix of the device name. For example, we have two NICs called mlx5_0, mlx5_1, then DEVNAME_SUFFIX_LIST should be {0, 1}

5. set kLinkBandwidth to the actual bandwidth of each NIC.

6. If each NIC has already been configured with its own IP address, nothing needs to be done. Otherwise, set the SINGLE_IP to the used IP address (IPv4 only for now).

### For run_nccl_test.sh:
1. Modify ROOT to the root directory of the workspace.
2. Modify NODES to the actual IPs of all nodes.
3. Modify CTRL_NIC to the actual name of control NIC.

```
Usage: ./run_nccl_test.sh [NCCL/UCCL: 0/1, default:1] [# of Nodes, default:2] [# of GPUs per node, default:8] [allreduce/alltoall: 0/1]
```
## Build
```
make -j`nproc`
```
