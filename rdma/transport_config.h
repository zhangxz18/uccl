#pragma once

#include <unistd.h>

#include <cstdint>
#include <thread>
#include <string>

#define CLOUDLAB_DEV

// #define STATS

/// Interface configuration.
static const char *IB_DEVICE_NAME_PREFIX = "mlx5_";
#ifdef CLOUDLAB_DEV
static const bool USE_ROCE = true;
// If SINGLE_IP is set, all devices will use the same IP.
static std::string SINGLE_IP("");
static const uint8_t NUM_DEVICES = 2;
static const uint8_t GID_INDEX_LIST[NUM_DEVICES] = {2,3};
#else
static const bool USE_ROCE = false;
// If SINGLE_IP is set, all devices will use the same IP.
static std::string SINGLE_IP("");
static const uint8_t NUM_DEVICES = 8;
static const uint8_t GID_INDEX_LIST[NUM_DEVICES] = {0, 1, 2, 3, 4, 5, 6, 7};
#endif
static const uint8_t IB_PORT_NUM = 1;
#ifdef CLOUDLAB_DEV
static const double kLinkBandwidth = 25.0 * 1e9 / 8; // 25Gbps
#else
static const double kLinkBandwidth = 400.0 * 1e9 / 8; // 400Gbps
#endif
/// Interface configuration.

// # of engines per device.
static const uint32_t NUM_ENGINES = 4;
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
// PortEntropy/Path/QP per engine. The total number is NUM_ENGINES * kPortEntropy.
static const uint32_t kPortEntropy = 256;

// Use RC rather than UC.
static const bool kUSERC = false;
constexpr static int kTotalQP = kUSERC ? kPortEntropy : kPortEntropy + 2;
// Per-path cwnd or global cwnd.
static const bool kPPCwnd = false;
// Recv buffer size smaller than kRCSize will be handled by RC directly.
static const uint32_t kRCSize = 65536;

// Limit the bytes of consecutive cached QP uses.
static constexpr uint32_t kMAXConsecutiveSameChoiceBytes = 16384;
// Message size threshold for allowing using cached QP.
static constexpr uint32_t kMAXUseCacheQPSize = 8192;
// Message size threshold for bypassing the timing wheel.
static constexpr uint32_t kBypassTimingWheelThres = 9000;

// Chunk size for each WQE.
static const uint32_t kChunkSize = 32 << 10;
// Limit the per-flow outstanding bytes on each engine.
static const uint32_t kMaxOutstandingBytes = 16 * kChunkSize;
// # of Tx work handled in one loop.
static const uint32_t kMaxTxWork = 4;
// Maximum number of Tx bytes to be transmitted in one loop.
static const uint32_t kMaxTxBytesThres = 32 * kChunkSize;
// # of Rx work handled in one loop.
static const uint32_t kMaxRxWork = 8;
// Completion queue (CQ) size.
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
static const uint32_t kMaxInline = 512;
// Maximum number of SGEs in one WQE.
static const uint32_t kMaxSge = 1;
// Maximum number of outstanding receive messages in one recv request.
static const uint32_t kMaxRecv = 8;
// Maximum number of outstanding receive requests in one engine.
static const uint32_t kMaxReq = 32 * kMaxRecv;
// Maximum number of WQEs in SRQ (Shared Receive Queue).
static const uint32_t kMaxSRQ = 64 * kMaxReq;
// Maximum number of WQEs in Retr RQ.
static const uint32_t kMaxRetr = 16;
// Maximum number of outstanding retransmission chunks on all QPs.
static const uint32_t kMaxInflightRetrChunks = 8;
static_assert(kMaxInflightRetrChunks <= kMaxRetr, "kMaxInflightRetrChunks <= kMaxRetr");
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
// kFastRexmitDupAckThres equals to 1 means all duplicate acks are caused by packet loss.
// This is true for flow-level ECMP, which is the common case.
// When the network supports adaptive routing, duplicate acks may be caused by adaptive routing.
// In this case, kFastRexmitDupAckThres should be set to a value greater than 0.
static const std::size_t kFastRexmitDupAckThres = 1;

// Maximum number of Retransmission Timeout (RTO) before aborting the flow.
static const uint32_t kRTOAbortThreshold = 50;

// Constant/Dynamic RTO.
static const bool kConstRTO = false;
// kConstRTO == true: Constant retransmission timeout in microseconds.
static const double kRTOUSec = 1000; // 1ms
// kConstRTO == false: Minimum retransmission timeout in microseconds.
static const double kMinRTOUsec = 1000; // 1ms
static const uint32_t kRTORTT = 5;     // RTO = kRTORTT RTTs

// Slow timer (periodic processing) interval in microseconds.
static const size_t kSlowTimerIntervalUs = 1000;  // 1ms

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
