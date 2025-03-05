#pragma once

#include <unistd.h>

#include <cstdint>
#include <string>
#include <thread>

#define CLOUDLAB_DEV

// #define STATS

/// Interface configuration.
static constexpr uint32_t MAX_PEER = 256;
// Maximum number of flows (one-way) on each engine.
static constexpr uint32_t MAX_FLOW = 256;
static const char *IB_DEVICE_NAME_PREFIX = "mlx5_";
#ifdef CLOUDLAB_DEV
static constexpr bool USE_ROCE = true;
// If SINGLE_IP is set, all devices will use the same IP.
static std::string SINGLE_IP("");
static constexpr uint8_t NUM_DEVICES = 2;
static constexpr uint8_t DEVNAME_SUFFIX_LIST[NUM_DEVICES] = {2, 3};
#else
static constexpr bool USE_ROCE = false;
// If SINGLE_IP is set, all devices will use the same IP.
static std::string SINGLE_IP("");
static constexpr uint8_t NUM_DEVICES = 8;
static constexpr uint8_t DEVNAME_SUFFIX_LIST[NUM_DEVICES] = {0, 1, 2, 3,
                                                             4, 5, 6, 7};
#endif
static constexpr uint8_t IB_PORT_NUM = 1;
#ifdef CLOUDLAB_DEV
static constexpr double kLinkBandwidth = 25.0 * 1e9 / 8; // 25Gbps
#else
static constexpr double kLinkBandwidth = 400.0 * 1e9 / 8; // 400Gbps
#endif
/// Interface configuration.

// # of engines per device.
static constexpr uint32_t NUM_ENGINES = 4;
// Starting from 1/4 of the CPUs to avoid conflicting with nccl proxy service.
static uint32_t NUM_CPUS = std::thread::hardware_concurrency();
static uint32_t ENGINE_CPU_START = NUM_CPUS / 4;
// Minimum post receive size in NCCL.
static constexpr uint32_t NCCL_MIN_POST_RECV = 65536;
// PortEntropy/Path/QP per engine. The total number is NUM_ENGINES *
// kPortEntropy.
static constexpr uint32_t kPortEntropy = 64;
// Chunk size for each WQE.
static constexpr uint32_t kChunkSize = 32 << 10;
// Note that load-based policy shoud >= ENGINE_POLICY_LOAD.
enum engine_lb_policy {
    // Bind each flow to one engine.
    ENGINE_POLICY_BIND,
    // Round-robin among engines.
    ENGINE_POLICY_RR,
    // Choose obliviously.
    ENGINE_POLICY_OBLIVIOUS,
    // Load balancing based on the load of each engine.
    ENGINE_POLICY_LOAD,
    // Variant of ENGINE_POLICY_LOAD, which uses power of two.
    ENGINE_POLICY_LOAD_POT,
};
static constexpr enum engine_lb_policy kEngineLBPolicy = ENGINE_POLICY_RR;

// Congestion control algorithm.
enum SenderCCA {
    SENDER_CCA_NONE,
    // Timely [SIGCOMM'15]
    SENDER_CCA_TIMELY,
};
enum ReceiverCCA {
    RECEIVER_CCA_NONE,
    // EQDS [NSDI'22]
    RECEIVER_CCA_EQDS,
};
static constexpr enum SenderCCA kSenderCCA = SENDER_CCA_TIMELY;
static constexpr enum ReceiverCCA kReceiverCCA = RECEIVER_CCA_EQDS;
static_assert(kSenderCCA != SENDER_CCA_NONE ||
                  kReceiverCCA != RECEIVER_CCA_NONE,
              "At least one of the sender and receiver must have a congestion "
              "control algorithm.");

static const uint32_t PACER_CPU_START = 3 * NUM_CPUS / 4;

// Use RC rather than UC.
static constexpr bool kUSERC = true;
constexpr static int kTotalQP = kPortEntropy + 1 /* Credit QP */ +
                                2 * (kUSERC ? 0 : 1) /* Ctrl QP, Retr QP */;
// Per-path cwnd or global cwnd.
static constexpr bool kPPCwnd = false;
// Recv buffer size smaller than kRCSize will be handled by RC directly.
static constexpr uint32_t kRCSize = 8192;

// Limit the bytes of consecutive cached QP uses.
static constexpr uint32_t kMAXConsecutiveSameChoiceBytes = 16384;
// Message size threshold for allowing using cached QP.
static constexpr uint32_t kMAXUseCacheQPSize = 8192;
// Message size threshold for bypassing the timing wheel.
static constexpr uint32_t kBypassTimingWheelThres = 9000;

// Limit the per-flow outstanding bytes on each engine.
static constexpr uint32_t kMaxOutstandingBytesPerFlow = 9 * kChunkSize;
// Limit the outstanding bytes on each engine.
static constexpr uint32_t kMaxOutstandingBytesEngine = 24 * kChunkSize;
// # of Tx work handled in one loop.
static constexpr uint32_t kMaxTxWork = 2;
// Maximum number of Tx bytes to be transmitted in one loop.
static constexpr uint32_t kMaxTxBytesThres = 32 * kChunkSize;
// # of Rx work handled in one loop.
static constexpr uint32_t kMaxRxWork = 8;
// Completion queue (CQ) size.
static constexpr int kCQSize = 16384;
// Interval for posting a signal WQE.
// static constexpr uint32_t kSignalInterval = kCQSize >> 1;
static constexpr uint32_t kSignalInterval = 1;
// Interval for syncing the clock with NIC.
static constexpr uint32_t kSyncClockIntervalNS = 100000;
// Maximum number of CQEs to retrieve in one loop.
static constexpr uint32_t kMaxBatchCQ = 16;
// CQ moderation count.
static constexpr uint32_t kCQMODCount = 32;
// CQ moderation period in microsecond.
static constexpr uint32_t kCQMODPeriod = 100;
// Maximum size of inline data.
static constexpr uint32_t kMaxInline = 512;
// Maximum number of SGEs in one WQE.
static constexpr uint32_t kMaxSge = 1;
// Maximum number of outstanding receive messages in one recv request.
static constexpr uint32_t kMaxRecv = 1;
// Maximum number of outstanding receive requests in one engine.
static constexpr uint32_t kMaxReq = 128;
// Maximum number of WQEs in SRQ (Shared Receive Queue).
static constexpr uint32_t kMaxSRQ = 64 * kMaxReq * 2;
// Maximum number of WQEs in Retr RQ.
static constexpr uint32_t kMaxRetr = 64;
// Maximum number of outstanding retransmission chunks on all QPs.
static constexpr uint32_t kMaxInflightRetrChunks = 32;
static_assert(kMaxInflightRetrChunks <= kMaxRetr,
              "kMaxInflightRetrChunks <= kMaxRetr");
// Maximum number of chunks can be transmitted from timing wheel in one loop.
static constexpr uint32_t kMaxBurstTW = 8;
// Posting recv WQEs every kPostRQThreshold.
static constexpr uint32_t kPostRQThreshold = kMaxBatchCQ;
// When CQEs from one QP reach kMAXCumWQE, send immediate ack.
// 1 means always send immediate ack.
static constexpr uint32_t kMAXCumWQE = 4;
// When the cumulative bytes reach kMAXCumBytes, send immediate ack.
static constexpr uint32_t kMAXCumBytes = kMAXCumWQE * kChunkSize;
// Before reaching it, the receiver will not consider that it has encountered
// OOO, and thus there is no immediate ack. This is to tolerate the OOO caused
// by the sender's qp scheduling.
static constexpr uint32_t kMAXRXOOO = 8;

// Sack bitmap size in bits.
static constexpr std::size_t kSackBitmapSize = 64 << 1;
// kFastRexmitDupAckThres equals to 1 means all duplicate acks are caused by
// packet loss. This is true for flow-level ECMP, which is the common case. When
// the network supports adaptive routing, duplicate acks may be caused by
// adaptive routing. In this case, kFastRexmitDupAckThres should be set to a
// value greater than 0.
static constexpr std::size_t kFastRexmitDupAckThres = 16;

// Maximum number of Retransmission Timeout (RTO) before aborting the flow.
static constexpr uint32_t kRTOAbortThreshold = 50;

// Constant/Dynamic RTO.
static constexpr bool kConstRTO = false;
// kConstRTO == true: Constant retransmission timeout in microseconds.
static constexpr double kRTOUSec = 1000; // 1ms
// kConstRTO == false: Minimum retransmission timeout in microseconds.
static constexpr double kMinRTOUsec = 1000; // 1ms
static constexpr uint32_t kRTORTT = 5;      // RTO = kRTORTT RTTs

// Slow timer (periodic processing) interval in microseconds.
static constexpr size_t kSlowTimerIntervalUs = 1000; // 1ms

/// Debugging and testing.
// Disable hardware timestamp.
static constexpr bool kTestNoHWTimestamp = false;
// Bypass the timing wheel.
static constexpr bool kTestNoTimingWheel = false;
// Use constant(maximum) rate for transmission.
static constexpr bool kTestConstantRate = false;
// Test lossy network.
static constexpr bool kTestLoss = false;
static constexpr double kTestLossRate = 0.0;
// Disable RTO.
static constexpr bool kTestNoRTO = false;
/// Debugging and testing.
