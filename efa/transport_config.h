#pragma once

#include <unistd.h>

#include <cstdint>
#include <string>
#include <thread>

#define P4D

// #define STATS

/// Interface configuration.
#ifdef P4D
static const uint8_t NUM_DEVICES = 4;
static const uint8_t GID_INDEX_LIST[NUM_DEVICES] = {0, 1, 2, 3};
static const std::string EFA_DEVICE_NAME_LIST[NUM_DEVICES] = {
    "rdmap16s27", "rdmap32s27", "rdmap144s27", "rdmap160s27"};
static const std::string ENA_DEVICE_NAME_LIST[NUM_DEVICES] = {
    "ens32", "ens65", "ens130", "ens163"};
static const double kLinkBandwidth = 100.0 * 1e9 / 8; // 100Gbps
#endif
static const uint8_t IB_PORT_NUM = 1;
static const uint32_t EFA_MTU = 9000;
static const uint32_t EFA_MAX_PAYLOAD = 8928;
static const uint32_t UD_ADDITION = 40;
/// Interface configuration.

// # of engines per device.
static const uint32_t NUM_ENGINES = 4;
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
// PortEntropy/Path/QP per engine. The total number is NUM_ENGINES *
// kPortEntropy.
static const uint32_t kPortEntropy = 256;

// Per-path cwnd or global cwnd.
static const bool kPPCwnd = false;

// Recv buffer size smaller than kRCSize will be handled by RC directly.
static const uint32_t kRCSize = 65536;
// # of Tx work handled in one loop.
static const uint32_t kMaxTxWork = 1;
// # of Rx work handled in one loop.
static const uint32_t kMaxRxWork = 8;
// Chunk size for each WQE.
static const uint32_t kChunkSize = 32 << 10;
// CQ size for UC SQ and RQ.
static const int kCQSize = 16384;
// Interval for posting a signal WQE.
// static const uint32_t kSignalInterval = kCQSize >> 1;
static const uint32_t kSignalInterval = 1;
// Interval for syncing the clock with NIC.
static const uint32_t kSyncClockIntervalNS = 100000;
// Maximum number of CQEs to retrieve in one loop.
static const uint32_t kMaxBatchCQ = 16;
// CQ moderation count.
static const uint32_t kCQMODCount = 32;
// CQ moderation period in microsecond.
static const uint32_t kCQMODPeriod = 100;
// Maximum size of inline data.
static const uint32_t kMaxInline = 0;
// Maximum number of SGEs in one WQE.
static const uint32_t kMaxSge = 2;
// Maximum number of outstanding receive messages in one recv request.
static const uint32_t kMaxRecv = 8;
// Maximum number of outstanding receive requests in one engine.
static const uint32_t kMaxReq = 32 * kMaxRecv;
// Maximum number of WQEs in RQ.
static const uint32_t kMaxRQ = kMaxReq;
// Maximum number of WQEs in Retr RQ.
static const uint32_t kMaxRetr = 16;
// Maximum number of outstanding retransmission chunks on all QPs.
static const uint32_t kMaxInflightRetrChunks = 8;
static_assert(kMaxInflightRetrChunks <= kMaxRetr,
              "kMaxInflightRetrChunks <= kMaxRetr");
// Maximum number of chunks can be transmitted from timing wheel in one loop.
static const uint32_t kMaxBurstTW = 24;
// Posting recv WQEs every kPostRQThreshold.
static const uint32_t kPostRQThreshold = kMaxBatchCQ;
// When CQEs from one QP reach kMAXCumWQE, send immediate ack.
// 1 means always send immediate ack.
static constexpr uint32_t kMAXCumWQE = 4;
// When the cumulative bytes reach kMAXCumBytes, send immediate ack.
static constexpr uint32_t kMAXCumBytes = kMAXCumWQE * kChunkSize;

// Sack bitmap size in bits.
static const std::size_t kSackBitmapSize = 64 << 1;
// kFastRexmitDupAckThres equals to 1 means all duplicate acks are caused by
// packet loss. This is true for flow-level ECMP, which is the common case. When
// the network supports adaptive routing, duplicate acks may be caused by
// adaptive routing. In this case, kFastRexmitDupAckThres should be set to a
// value greater than 0.
static const std::size_t kFastRexmitDupAckThres = 1;

// Maximum number of Retransmission Timeout (RTO) before aborting the flow.
static const uint32_t kRTOAbortThreshold = 50;

// Constant/Dynamic RTO.
static const bool kConstRTO = false;
// kConstRTO == true: Constant retransmission timeout in microseconds.
static const double kRTOUSec = 1000; // 1ms
// kConstRTO == false: Minimum retransmission timeout in microseconds.
static const double kMinRTOUsec = 1000; // 1ms
static const uint32_t kRTORTT = 5;      // RTO = kRTORTT RTTs

// Slow timer (periodic processing) interval in microseconds.
static const size_t kSlowTimerIntervalUs = 1000; // 1ms

/// Debugging and testing.
// Disable hardware timestamp.
static const bool kTestNoHWTimestamp = false;
// Bypass the timing wheel.
static const bool kTestNoTimingWheel = false;
// Use constant(maximum) rate for transmission.
static const bool kTestConstantRate = false;
// Test lossy network.
static const bool kTestLoss = false;
static const double kTestLossRate = 0.0;
// Disable RTO.
static const bool kTestNoRTO = false;
// Always use the same engine for each flow.
static const bool kBindEngine = false;
/// Debugging and testing.
