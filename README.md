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

On two AWS `g4dn.8xlarge` instances with 50G NICs and T4 GPUs under the cluster placement group, UCCL outperforms NCCL by up to **3.7x** for AllReduce: 

![UCCL Performance Report](./allreduce_perf.png)

## Support


## Development Guide

Please refer to [./doc/README_dev.md](doc/README_dev.md) for development setup and testing.

## Acknowledgement

UCCL is being actively developed at [UC Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). We welcome contributions from open-source developers. 
