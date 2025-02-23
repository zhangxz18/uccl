#pragma once
#include <cstdint>
#include <thread>

#include "util.h"

#define USE_MULTIPATH
#define PATH_SELECTION
#define REXMIT_SET_PATH
// #define RTT_STATS
// #define USE_SRD

enum class CCType {
    kTimely,
    kTimelyPP,
    kCubic,
    kCubicPP,
};
static constexpr CCType kCCType = CCType::kCubicPP;

#define P4D
// #define G6E

/// Interface configuration.
#ifdef P4D
static const uint8_t NUM_DEVICES = 1;
// static const uint8_t NUM_DEVICES = 4;
static const uint8_t EFA_GID_IDX = 0;
static const std::string EFA_DEVICE_NAME_LIST[] = {
    "rdmap16s27", "rdmap32s27", "rdmap144s27", "rdmap160s27"};
static const std::string ENA_DEVICE_NAME_LIST[] = {"ens32", "ens65", "ens130",
                                                   "ens163"};
static const double kLinkBandwidth = 100.0 * 1e9 / 8;  // 100Gbps
#elif defined(G6E)
static const uint8_t NUM_DEVICES = 4;
static const uint8_t EFA_GID_IDX = 0;
static const std::string EFA_DEVICE_NAME_LIST[] = {"rdmap155s0", "rdmap156s0",
                                                   "rdmap188s0", "rdmap189s0"};
static const std::string ENA_DEVICE_NAME_LIST[] = {"enp135s0", "enp136s0",
                                                   "enp170s0", "enp171s0"};
static const double kLinkBandwidth = 100.0 * 1e9 / 8;  // 100Gbps
#endif
static const uint8_t EFA_PORT_NUM = 1;  // The port of EFA device to use.
static const uint32_t EFA_MTU = 9000;  // Max frame on fabric, includng headers.
static const uint32_t EFA_MAX_PAYLOAD = 8928;  // this excludes EFA_UD_ADDITION.
static const uint32_t EFA_MAX_QPS = 256;       // Max QPs per EFA device.
#ifdef USE_SRD
static const uint32_t EFA_UD_ADDITION = 0;  // Auto-added by EFA during recv.
#else
static const uint32_t EFA_UD_ADDITION = 40;  // Auto-added by EFA during recv.
#endif
/// Interface configuration.

static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
static const uint16_t BASE_PORT = 10000;
static const uint64_t NUM_FRAMES = 65536;  // # of frames.
static const uint32_t RECV_BATCH_SIZE = 32;
static const uint32_t SEND_BATCH_SIZE = 32;
static const uint32_t QKEY = 0x12345;
static const uint32_t SQ_PSN = 0x12345;

// libibverbs configuration.
static const uint32_t kNumEnginesPerDev = 2;  // # of engines per EFA device.
static const uint32_t kNumEngines = NUM_DEVICES * kNumEnginesPerDev;
static const uint32_t kMaxSendWr = 1024;
static const uint32_t kMaxRecvWr = 128;
static const uint32_t kMaxSendRecvWrForCtrl = 1024;
static const uint32_t kMaxCqeTotal = 16384;
static const uint32_t kMaxPollBatch = 32;
static const uint32_t kMaxRecvWrDeficit = 32;
static const uint32_t kMaxChainedWr = 32;
const static uint32_t kMaxUnconsumedRxMsgbufs = NUM_FRAMES / 4;

// Path configuration.
static const uint32_t kMaxDstQP = 8;      // # of paths/QPs for data per src qp.
static const uint32_t kMaxDstQPCtrl = 8;  // # of paths/QPs for control.
static_assert(kMaxDstQP + kMaxDstQPCtrl <= EFA_MAX_QPS);
static const uint32_t kMaxSrcQP = 16;
static const uint32_t kMaxSrcQPCtrl = 16;
static_assert(kMaxSrcQP + kMaxSrcQPCtrl <= EFA_MAX_QPS);
static constexpr uint32_t kMaxSrcDstQP = std::max(kMaxSrcQP, kMaxDstQP);
static constexpr uint32_t kMaxSrcDstQPCtrl =
    std::max(kMaxSrcQPCtrl, kMaxDstQPCtrl);
static const uint32_t kMaxPath = kMaxDstQP * kMaxSrcQP;
static const uint32_t kMaxPathCtrl = kMaxDstQPCtrl * kMaxSrcQPCtrl;
static_assert(kMaxPath == kMaxPathCtrl);  // To make path_id calculation simple.

// CC parameters.
static const uint32_t kMaxUnackedPktsPP = 1u;
static const uint32_t kMaxUnackedPktsPerEngine = kMaxUnackedPktsPP * kMaxPath;
static const std::size_t kSackBitmapSize = 1024;
static const std::size_t kFastRexmitDupAckThres = 30;
static const double kMaxBwPP = 5.0 * 1e9 / 8;
static const uint32_t kSwitchPathThres = 1u;
static const uint32_t kMaxPktsInTimingWheel = 1024;
static const uint32_t kMaxPathHistoryPerEngine = 4096;

static_assert(kMaxPath <= 4096, "kMaxPath too large");
static_assert(kMaxUnackedPktsPerEngine <= kMaxPathHistoryPerEngine,
              "kMaxUnackedPktsPerEngine too large");
static_assert(is_power_of_two(kMaxPathHistoryPerEngine),
              "kMaxPathHistoryPerEngine must be power of 2");
