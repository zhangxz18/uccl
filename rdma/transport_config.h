#pragma once
#include <cstdint>
#include <thread>

static const uint32_t NUM_QUEUES = 1;
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t MAX_UNACKED_PKTS = 512;
static const uint32_t MAX_TIMING_WHEEL_PKTS = 1024;
// CC parameters.
static const uint32_t kPortEntropy = 32;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 5;

static const char* DEV_RDMA_DEFAULT = "mlx5_2";
static const uint32_t RDMA_MTU = 4096;
static const char* DEV_DEFAULT = "enp65s0f0np0";
static const double kLinkBandwidth = 100.0 * 1e9 / 8;
