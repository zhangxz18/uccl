#pragma once

#include <cstdint>

#include "transport_config.h"
#include "util_endian.h"

/**
 * Uccl Packet Header just after UDP header.
 */
struct __attribute__((packed)) UcclPktHdr {
    static constexpr uint16_t kMagic = 0x4e53;
    be16_t magic;  // Magic value tagged after initialization for the flow.
    uint16_t engine_id : 4;  // remote UcclEngine ID to process this packet.
    uint16_t path_id : 12;   // path_id of this dst port.
    enum class UcclFlags : uint8_t {
        kData = 0b0,              // Data packet.
        kAck = 0b10,              // ACK packet.
        kRssProbe = 0b100,        // RSS probing packet.
        kRssProbeRsp = 0b1000,    // RSS probing rsp packet.
        kDataRttProbe = 0b10000,  // RTT probing packet.
        kAckRttProbe = 0b100000,  // RTT probing packet.
    };
    UcclFlags net_flags;  // Network flags.
    uint8_t msg_flags;    // Field to reflect the `FrameDesc' flags.
    be16_t frame_len;     // Length of the frame.
    be64_t flow_id;       // Flow ID to denote the connection.
    be32_t seqno;  // Sequence number to denote the packet counter in the flow.
    be32_t ackno;  // Sequence number to denote the packet counter in the flow.
    uint64_t timestamp1;  // Filled by sender with calibration for output queue
    uint64_t timestamp2;  // Filled by recver eBPF
};
struct __attribute__((packed)) UcclSackHdr {
    uint64_t timestamp3;  // Filled by recer with calibration for output queue
    uint64_t timestamp4;  // Filled by sender eBPF
    be64_t sack_bitmap[kSackBitmapSize /
                       swift::Pcb::kSackBitmapBucketSize];  // Bitmap of the
                                                            // SACKs received.
    be16_t sack_bitmap_count;  // Length of the SACK bitmap [0-256].
};
static const size_t kUcclPktHdrLen = sizeof(UcclPktHdr);
static const size_t kUcclPktdataLen = EFA_MAX_PAYLOAD - kUcclPktHdrLen;
static const size_t kUcclSackHdrLen = sizeof(UcclSackHdr);
static_assert(EFA_GRH_SIZE + kUcclPktHdrLen + kUcclSackHdrLen <
                  PktHdrBuffPool::kPktHdrSize,
              "uccl pkt hdr and sack hdr too large");

#ifdef USE_TCP
static const size_t kNetHdrLen =
    sizeof(ethhdr) + sizeof(iphdr) + sizeof(tcphdr);
#else
static const size_t kNetHdrLen =
    sizeof(ethhdr) + sizeof(iphdr) + sizeof(udphdr);
#endif

inline UcclPktHdr::UcclFlags operator|(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) |
                                 static_cast<UcclFlagsType>(rhs));
}

inline UcclPktHdr::UcclFlags operator&(UcclPktHdr::UcclFlags lhs,
                                       UcclPktHdr::UcclFlags rhs) {
    using UcclFlagsType = std::underlying_type<UcclPktHdr::UcclFlags>::type;
    return UcclPktHdr::UcclFlags(static_cast<UcclFlagsType>(lhs) &
                                 static_cast<UcclFlagsType>(rhs));
}
