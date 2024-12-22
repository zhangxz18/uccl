#pragma once
#include <cstdint>
#include <thread>
#include <string>

static const uint32_t NUM_ENGINES = 1;
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t MAX_UNACKED_PKTS = 512;
static const uint32_t MAX_TIMING_WHEEL_PKTS = 1024;
// CC parameters.
static const uint32_t kPortEntropy = 128;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 5;

// IB interface.
static std::string H100_IP("");
static const uint32_t MAX_IB_DEVICES = 32;
static const char *IB_DEVICE_NAME_PREFIX = "mlx5_";
static const uint8_t GID_INDEX_LIST[MAX_IB_DEVICES] = {
    2,
    3,
};
static const uint8_t IB_PORT_NUM = 1;
static const uint8_t NUM_DEVICES = 2;

// SgeSize = MTU << kSgeSizeShift
static const uint32_t kSgeSizeShift = 6;
static const uint32_t kSignalInterval = 256;
static const uint32_t kCQMODCount = 16;
static const uint32_t kCQMODPeriod = 10;
static const uint32_t kMaxSge = 32;
static const uint32_t kMaxNetReq = 32;
static const uint32_t kMaxRecv = 8;
static const uint32_t kMaxReq = kMaxNetReq * kMaxRecv;

// static const uint32_t RDMA_MTU = 4096;
static const uint32_t RDMA_MTU = 1024;
// 100Gbps link.
static const double kLinkBandwidth = 100.0 * 1e9 / 8;
