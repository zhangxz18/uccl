#pragma once
#include <cstdint>
#include <thread>
#include <string>

static const uint32_t NUM_ENGINES = 4;
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
// If USE_ROCE is set, RoCE will be used instead of IB.
static const bool USE_ROCE = true;
// If SINGLE_IP is set, all devices will use the same IP.
static std::string SINGLE_IP("");
static const uint32_t MAX_IB_DEVICES = 32;
static const char *IB_DEVICE_NAME_PREFIX = "mlx5_";
static const uint8_t GID_INDEX_LIST[MAX_IB_DEVICES] = {
    2,
    3,
};
static const uint8_t IB_PORT_NUM = 1;
static const uint8_t NUM_DEVICES = 2;

static const uint32_t kChunkSize = 256 << 10;
static const uint32_t kSignalInterval = 256;
static const uint32_t kSyncClockIntervalNS = 100000;
static const uint32_t kCQMODCount = 16;
static const uint32_t kCQMODPeriod = 10;
static const uint32_t kMaxSge = 1;
static const uint32_t kMaxNetReq = 32;
static const uint32_t kMaxRecv = 8;
static const uint32_t kMaxReq = kMaxNetReq * kMaxRecv;
static const uint32_t kMaxRetr = 16;
static const uint32_t kMaxBatchPost = 32;

// For debugging and testing.
// Use RDMA RC instead of UC.
static const bool kTestRC = true;
static const uint32_t kTestRCEntropy = 8;
// Disable hardware timestamp.
static const bool kTestNoHWTimestamp = false;
// Bypass the timing wheel.
static const bool kTestNoTimingWheel = false;
// Use constant(maximum) rate for transmission.
static const bool kTestConstantRate = false;
// Test lossy network.
static const double kTestLossRate = 0.0;

// 400Gbps link.
static const double kLinkBandwidth = 400.0 * 1e9 / 8;
