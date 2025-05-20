<div align="center">

# UCCL

<p align="center">
    <a href="#about"><b>About</b></a> | 
    <a href="#getting-started"><b>Getting Started</b></a> | 
    <a href="#development-guide"><b>Development Guide</b></a> | 
    <a href="#acknowledgement"><b>Acknowledgement</b></a>
</p>

</div>

## About 

UCCL is an efficient collective communication library for GPUs. 

Existing network transports under NCCL (i.e., kernel TCP and RDMA) leverage one or few network paths to stream huge data volumes, thus prone to congestion happening in datacenter networks. Instead, UCCL employs packet spraying in software to leverage abundant network paths to avoid "single-path-of-congestion". With this design, UCCL provides the following benefits: 
* Faster collectives by leveraging multi-path
* Widely available in the public cloud by leveraging legacy NICs and Ethernet fabric
* Evolvable transport designs including multi-path load balancing and congestion control
* Open-source research platform for ML collectives

On two AWS `g4dn.8xlarge` instances with 1x50G NICs and 1xT4 GPUs under the same cluster placement group, UCCL outperforms NCCL by up to **3.7x** for AllReduce: 

![UCCL allreduce performance](./doc/images/allreduce_perf.png)

On four AWS `p4d.24xlarge` instances with 4x100G NICs and 8xA100 GPUs, UCCL outperforms NCCL by up to **3.3x** for AlltoAll: 

![UCCL allreduce performance](./doc/images/alltoall_perf.png)

Free free to checkout our full [technical report](https://arxiv.org/pdf/2504.17307).

## UCCL Dev Agenda

- [ ] Dynamic membership with GPU servers joining and exiting
- [ ] GPU-initiated network P2P that support all NIC vendors including Nvidia, AWS EFA, and Broadcom, to support MoE all-to-all workload and KV cache transfers in PD disaggregation. 
- [ ] NCCL re-architecturing
    - [ ] Scalable and effcient CPU proxy
    - [ ] Low-cost async collectives with ordering guarantee
    - [ ] Device kernel in vendor-agnostic Triton


## Getting Started

Let's first close the UCCL repo and init submodules. 
```
git clone https://github.com/uccl-project/uccl.git --recursive
export UCCL_HOME=$(pwd)/uccl
```

Then install some common dependencies: 
```
sudo apt update
sudo apt install clang llvm libelf-dev libpcap-dev build-essential libc6-dev-i386 linux-tools-$(uname -r) libgoogle-glog-dev libgtest-dev byobu net-tools iperf iperf3 libgtest-dev cmake m4 libopenmpi-dev libibverbs-dev libpci-dev -y

# Install and activate Anaconda (you can choose any recent versions)
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh -b
source ~/anaconda3/bin/activate
source ~/.bashrc
conda init

# Install dependency into default base env
conda install paramiko -y
```

Next you can dive into individual folders for various supports: 
* [`efa/`](./efa/README.md): AWS EFA NIC (currently support p4d.24xlarge)
* [`afxdp/`](./afxdp/README.md): Non-RDMA NICs (currently support AWS ENA NICs and IBM VirtIO NICs)
* [`rdma_cuda/`](./rdma_cuda/README.md): Nvidia/Mellanox GPUs + RDMA NICs (both IB and RoCE)
* [`rdma_hip/`](./rdma_hip/README.md): AMD GPUs + RDMA NICs (both IB and RoCE)

## Documentation

Please refer to [doc/README.md](./doc/README.md) for full documentation.

## Acknowledgement

UCCL is being actively developed at [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). We welcome open-source developers. 

## Contact
Feel free to raise issues or contact us if you have any questions or suggestions. You can reach us at: Yang Zhou (yangzhou.rpc@gmail.com), Zhongjie Chen (chenzhjthu@gmail.com), and Ziming Mao (ziming.mao@berkeley.edu). 
