#pragma once
#include <cstdint>
#include <thread>

#include "util.h"

#define USE_MULTIPATH
#define PATH_SELECTION
#define REXMIT_SET_PATH
#define THREADED_MEMCPY
// #define EMULATE_ZC
// #define USE_TCP
// #define ENABLE_CSUM
// #define RTT_STATS

enum class CCType {
    kTimely,
    kTimelyPP,
    kCubic,
    kCubicPP,
};
static constexpr CCType kCCType = CCType::kCubicPP;

#define P4D

/// Interface configuration.
#ifdef P4D
static const uint8_t NUM_DEVICES = 4;
static const uint8_t GID_INDEX_LIST[NUM_DEVICES] = {0, 1, 2, 3};
static const std::string EFA_DEVICE_NAME_LIST[NUM_DEVICES] = {
    "rdmap16s27", "rdmap32s27", "rdmap144s27", "rdmap160s27"};
static const std::string ENA_DEVICE_NAME_LIST[NUM_DEVICES] = {
    "ens32", "ens65", "ens130", "ens163"};
static const double kLinkBandwidth = 100.0 * 1e9 / 8;  // 100Gbps
#endif
static const uint8_t IB_PORT_NUM = 1;
static const uint32_t EFA_MTU = 9000;
static const uint32_t EFA_MAX_PAYLOAD = 8928;
static const uint32_t UD_ADDITION = 40;
/// Interface configuration.

static const uint32_t NUM_ENGINES = 1;      // # of engines per device.
static const uint32_t kNumPktPerChunk = 4;  // # of 9KB packets per chunk.
static const uint32_t kMaxPath = 256 - 1;   // We need to reserve one for
                                            // the control QP.
static const uint32_t kMaxUnackedPktsPP = 1u;

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
static const uint64_t NUM_FRAMES = 65536;  // # of frames.
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;

// CC parameters.
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 10;
static const uint32_t kMaxTwPkts = 1024;
static const double kMaxBwPP = 5.0 * 1e9 / 8;
static const uint32_t kSwitchPathThres = 1u;
static const uint32_t kMaxUnackedPktsPerEngine = kMaxUnackedPktsPP * kMaxPath;
static const uint32_t kMaxPathHistoryPerEngine = 4096;

static_assert(kMaxPath <= 4096, "kMaxPath too large");
static_assert(NUM_QUEUES <= 16, "NUM_QUEUES too large");
static_assert(kMaxUnackedPktsPerEngine <= kMaxPathHistoryPerEngine,
              "kMaxUnackedPktsPerEngine too large");
static_assert(is_power_of_two(kMaxPathHistoryPerEngine),
              "kMaxPathHistoryPerEngine must be power of 2");
