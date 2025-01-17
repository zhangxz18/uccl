#include "transport.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <endian.h>
#include <utility>

#include <infiniband/verbs.h>

#include "transport_cc.h"
#include "transport_config.h"
#include "util_rdma.h"
#include "util_timer.h"
#include "util_list.h"

namespace uccl {

static uint64_t fff = 0;

void UcclFlow::post_fifo(struct FlowRequest *req, void **data, size_t *size, int n, struct ibv_mr **mr)
{
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    struct RemFifo *rem_fifo = rdma_ctx_->recv_comm_.base.fifo;
    int slot = rem_fifo->fifo_tail % kMaxReq;
    auto elems = rem_fifo->elems[slot];
    auto qp = rdma_ctx_->fifo_qp_;

    auto recv_comm_ = &rdma_ctx_->recv_comm_;

    // Reset received bytes and fin requests.
    req->recv.fin_msg = 0;
    req->recv.received_bytes = rem_fifo->received_bytes[slot];
    memset(req->recv.received_bytes, 0, sizeof(uint32_t) * kMaxRecv);
    req->recv.elems = elems;
    
    for (int i = 0; i < n; i++) {
        elems[i].addr = reinterpret_cast<uint64_t>(data[i]);
        elems[i].rkey = mr[i]->rkey;
        elems[i].nmsgs = n;
        // For sender to check if the receiver is ready.
        elems[i].idx = rem_fifo->fifo_tail + 1;
        elems[i].size = size[i];
        // For sender to encode the request id in the immediate data.
        elems[i].rid = rdma_ctx_->get_request_id(req, &recv_comm_->base);

        VLOG(5) << "Post Recv: addr: " << elems[i].addr << ", rkey: " << elems[i].rkey << ", size: " << elems[i].size;
    }
    
    // Figure out the remote address to write.
    wr.wr.rdma.remote_addr = recv_comm_->base.remote_ctx.fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
    wr.wr.rdma.rkey = recv_comm_->base.remote_ctx.fifo_key;

    struct ibv_sge sge;
    sge.lkey = rdma_ctx_->fifo_mr_->lkey;
    sge.addr = (uint64_t)elems;
    sge.length = n * sizeof(struct FifoItem);

    wr.sg_list = &sge;
    wr.num_sge = 1;

    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_INLINE;

    // Occasionally post a request with the IBV_SEND_SIGNALED flag.
    if (slot == 0) {
        wr.send_flags |= IBV_SEND_SIGNALED;
        if (!rdma_ctx_->fifo_cq_polling_) {
            engine_->add_fifo_cq_polling(this);
            rdma_ctx_->fifo_cq_polling_ = true;
        }
    }
    struct ibv_send_wr* bad_wr;
    if (ibv_post_send(qp, &wr, &bad_wr)) {
        LOG(ERROR) << "Failed to post send";
    }
    rem_fifo->fifo_tail++;
}

void UcclFlow::flush_rx_buf(Channel::Msg &rx_work)
{
    auto data = rx_work.rx.data;
    auto size = rx_work.rx.size;
    auto mr = rx_work.rx.mr;
    auto n = rx_work.rx.n;
    auto poll_ctx = rx_work.poll_ctx;

    auto recv_comm_ = &rdma_ctx_->recv_comm_;

    if (unlikely(n > kMaxRecv)) {
        LOG(ERROR) << "Number of buffers exceeds the limit.";
        return;
    }

    // Only flush once using the last non-zero receive
    int last = -1;
    for (int i = 0; i < n; i++) if (size[i]) last = i;
    if (last == -1) return;

    auto req = rdma_ctx_->get_request(&recv_comm_->base);
    if (unlikely(!req)) {
        LOG(ERROR) << "Number of outstanding requests exceeds the limit.";
        return;
    }

    req->type = FlowRequest::FLUSH;
    req->nmsgs = n;
    req->poll_ctx = poll_ctx;
    if constexpr (kTestRC)
        req->events = 1;

    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    
    wr.wr_id = rdma_ctx_->get_request_id(req, &recv_comm_->base);

    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(data[last]);
    wr.wr.rdma.rkey = mr[last]->rkey;
    wr.sg_list = &rdma_ctx_->gpu_flush_sge_;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    DCHECK(ibv_post_send(rdma_ctx_->gpu_flush_qp_, &wr, &bad_wr) == 0);

    VLOG(5) << rdma_ctx_->gpu_flush_qp_->qp_num << " Post flush: addr: " << wr.wr.rdma.remote_addr << ", rkey: " << wr.wr.rdma.rkey;
}

void UcclFlow::app_supply_rx_buf(Channel::Msg &rx_work)
{
    auto data = rx_work.rx.data;
    auto size = rx_work.rx.size;
    auto mr = rx_work.rx.mr;
    auto n = rx_work.rx.n;
    auto poll_ctx = rx_work.poll_ctx;
    auto ureq = rx_work.ureq;

    auto recv_comm_ = &rdma_ctx_->recv_comm_;

    if (unlikely(n > kMaxRecv)) {
        LOG(ERROR) << "Number of buffers exceeds the limit.";
        return;
    }

    auto req = rdma_ctx_->get_request(&recv_comm_->base);
    if (unlikely(!req)) {
        LOG(ERROR) << "Number of outstanding requests exceeds the limit.";
        return;
    }

    req->type = FlowRequest::RECV;
    req->nmsgs = n;
    req->poll_ctx = poll_ctx;
    req->ureq = ureq;

    if constexpr (kTestRC) {
        // Post recv work requests for consuming immediate data.
        struct ibv_recv_wr wr, *bad_wr;
        memset(&wr, 0, sizeof(wr));

        req->events = kTestRCEntropy;

        for (int i = 0; i < kTestRCEntropy; i++) {
            wr.wr_id = rdma_ctx_->get_request_id(req, &recv_comm_->base);
            wr.sg_list = nullptr;
            wr.num_sge = 0;
            wr.next = nullptr;
            DCHECK(ibv_post_srq_recv(rdma_ctx_->srq_, &wr, &bad_wr) == 0);
        }
    }

    // Push buffer information to FIFO queue and notify the remote peer.
    post_fifo(req, data, size, n, mr);
}

void UcclFlow::post_single_message(struct FlowRequest *req, struct FifoItem &slot, uint32_t mid)
{
    auto *sent_offset = &req->send.sent_offset;
    auto size = req->send.size;
    auto data = (uint64_t)req->send.data;
    auto lkey = req->send.lkey;
    auto rkey = slot.rkey;
    auto remote_addr = slot.addr;
    uint64_t wr_addr;
    int mismatch = 0;
    uint32_t nchunk = 0;

    if (req->ureq != nullptr) {
       mismatch = 1;
       nchunk = (size + kChunkSize - 1) / kChunkSize;
    }

    while (*sent_offset < size) {
        if constexpr (kTestNoTimingWheel) {
            req->events++;
            DCHECK(rdma_ctx_->wr_ex_pool_->alloc_buff(&wr_addr) == 0);
            struct wr_ex *wr_ex = reinterpret_cast<struct wr_ex *>(wr_addr);
            auto wr = &wr_ex->wr;

            wr_ex->sge.addr = data + *sent_offset;
            wr_ex->sge.lkey = lkey;
            wr_ex->sge.length = std::min(size - *sent_offset, (int)kChunkSize);

            auto chunk_size = wr_ex->sge.length;

            wr->wr.rdma.remote_addr = remote_addr + *sent_offset;
            wr->wr.rdma.rkey = rkey;

            IMMData imm_data(0);

            imm_data.SetNCHUNK(0);
            if (mismatch && (*sent_offset + chunk_size == size)) {
                imm_data.SetNCHUNK(nchunk);
            }
            imm_data.SetRID(slot.rid);
            imm_data.SetMID(mid);
            
            // Select QP.
            auto qpidx = rdma_ctx_->select_qpidx_pow2();
            auto qpw = &rdma_ctx_->uc_qps_[qpidx];

            if (qpw->signal_cnt_++ % kSignalInterval == 0)
                wr->send_flags = IBV_SEND_SIGNALED;
            
            imm_data.SetCSN(qpw->pcb.get_snd_nxt().to_uint32());

            wr->imm_data = htonl(imm_data.GetImmData());

            struct ibv_send_wr *bad_wr;
            DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

            *sent_offset += chunk_size;
            // Track this chunk.
            qpw->txtracking.track_chunk(req, imm_data.GetCSN(), wr_ex, rdtsc());

            continue;
        }

        req->events++;
        // Prepare SGE.
        DCHECK(rdma_ctx_->wr_ex_pool_->alloc_buff(&wr_addr) == 0);
        struct wr_ex *wr_ex = reinterpret_cast<struct wr_ex *>(wr_addr);
        auto wr = &wr_ex->wr;
        wr_ex->sge.addr = data + *sent_offset;
        wr_ex->sge.lkey = lkey;
        wr_ex->sge.length = std::min(size - *sent_offset, (int)kChunkSize);
        auto chunk_size = wr_ex->sge.length;

        // wr->sg_list/num_sge/next/opcode are already set.

        wr->wr.rdma.remote_addr = remote_addr + *sent_offset;
        wr->wr.rdma.rkey = rkey;

        VLOG(5) << "remote_addr: " << wr->wr.rdma.remote_addr << ", rkey: " << wr->wr.rdma.rkey;

        IMMData imm_data(0);

        imm_data.SetNCHUNK(0);
        if (mismatch && (*sent_offset + chunk_size == size)) {
            imm_data.SetNCHUNK(nchunk);
        }
        imm_data.SetRID(slot.rid);
        imm_data.SetMID(mid);
        
        // Select QP.
        auto qpidx = rdma_ctx_->select_qpidx_pow2();
        auto qpw = &rdma_ctx_->uc_qps_[qpidx];
        // There is no need to signal every WQE since we don't handle TX completions.
        // But we still need occasionally post a request with the IBV_SEND_SIGNALED flag.
        if (qpw->signal_cnt_++ % kSignalInterval == 0)
            wr_ex->wr.send_flags = IBV_SEND_SIGNALED;

        imm_data.SetCSN(qpw->pcb.get_snd_nxt().to_uint32());

        wr->imm_data = htonl(imm_data.GetImmData());

        wr_ex->qpidx = qpidx;

        *sent_offset += chunk_size;
        // Queue the SGE on the timing wheel.
        {
            auto wheel = &rdma_ctx_->wheel_;
            uint32_t hdr_overhead;
            if (likely(chunk_size == kChunkSize && rdma_ctx_->mtu_bytes_ == 4096)) {
                hdr_overhead = USE_ROCE ? MAX_CHUNK_IB_4096_HDR_OVERHEAD : MAX_CHUNK_ROCE_IPV4_4096_HDR_OVERHEAD;
            } else {
                auto num_mtu = (chunk_size + rdma_ctx_->mtu_bytes_) / rdma_ctx_->mtu_bytes_;
                hdr_overhead = num_mtu * (USE_ROCE ? ROCE_IPV4_HDR_OVERHEAD : IB_HDR_OVERHEAD);
            }
            if (wheel->queue_on_timing_wheel(qpw->pcb.timely.rate_, rdtsc(), 
                wr_ex, chunk_size + hdr_overhead, qpw->in_wheel_cnt_ == 0)) {
                    qpw->in_wheel_cnt_++;
                    // For future tracking.
                    wr_ex->req = req;
                    VLOG(5) << "Queue " << chunk_size << " bytes on QP#" << qpidx;
            }
            else {
                // Transmit this chunk directly.
                struct ibv_send_wr *bad_wr;
                DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);

                // Track this chunk.
                qpw->txtracking.track_chunk(req, imm_data.GetCSN(), wr_ex, rdtsc());

                VLOG(5) << "Directly send " << chunk_size << " bytes to QP#" << qpidx;
            }
        }

        VLOG(5) << "Sending: csn: " << imm_data.GetCSN() << ", rid: " << slot.rid << ", mid: " << mid << " with QP#" << qpidx;
    }

}

void UcclFlow::test_rc_post_multi_messages(int slot)
{
    auto send_comm_ = &rdma_ctx_->send_comm_;
    auto reqs = send_comm_->fifo_reqs[slot];
    auto rem_fifo = send_comm_->base.fifo;
    auto slots = rem_fifo->elems[slot];
    auto nmsgs = slots[0].nmsgs;

    struct ibv_send_wr wrs[kMaxRecv + 1];
    struct ibv_send_wr *last_wr = &wrs[nmsgs - 1];
    struct ibv_sge size_sges;
    struct ibv_sge sges[kMaxRecv];
    uint32_t send_offset[kMaxRecv];

    auto align = 128;

    uint64_t wr_id = 0ULL;

    for (int i = 0; i < nmsgs; i++) {
        auto req = reqs[i];
        auto *data = req->send.data;
        auto size = req->send.size;
        auto wr = &wrs[i];
        auto sge = &sges[i];
        
        sge->lkey = req->send.lkey;
        send_offset[i] = 0;

        // Multi-messages
        send_comm_->base.fifo->sizes[slot][i] = size;

        wr->sg_list = sge;
        wr->num_sge = 1;

        wr->next = &wrs[i + 1];
        wr->opcode = IBV_WR_RDMA_WRITE;
        wr->send_flags = 0;
        wr->wr.rdma.remote_addr = slots[i].addr;
        wr->wr.rdma.rkey = slots[i].rkey;

        wr_id += rdma_ctx_->get_request_id(req, &send_comm_->base) << (i * 8);

        VLOG(5) << "Post wr_id: " << rdma_ctx_->get_request_id(req, &send_comm_->base);
    }

    if (nmsgs > 1) {
        last_wr++;

        size_sges.addr = reinterpret_cast<uint64_t>(send_comm_->base.fifo->sizes[slot]);
        size_sges.lkey = rdma_ctx_->fifo_mr_->lkey;
        size_sges.length = nmsgs * sizeof(uint32_t);

        last_wr->sg_list = &size_sges;
        last_wr->num_sge = 1;
        last_wr->imm_data = 0;
        
        // Figure out the remote address to write.
        last_wr->wr.rdma.remote_addr = send_comm_->base.remote_ctx.fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
        last_wr->wr.rdma.rkey = send_comm_->base.remote_ctx.fifo_key;

    } else {
        last_wr->imm_data = reqs[0]->send.size;
    }

    last_wr->next = nullptr;
    last_wr->send_flags = IBV_SEND_SIGNALED;
    last_wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;

    // Store the request id for future acknowledgment.
    last_wr->wr_id = wr_id;
    VLOG(5) << "Post merged wr_id: " << wr_id;

    for (int i = 0; i < kTestRCEntropy; i++) {
        auto qpidx = i;
        auto qpw = &rdma_ctx_->uc_qps_[qpidx];

        for (int m = 0; m < nmsgs; m++) {
            auto chunk_size = DIVUP(DIVUP(reqs[m]->send.size, kTestRCEntropy), align) * align;
            auto remaining_bytes = reqs[m]->send.size - send_offset[m];
            auto length = std::min(remaining_bytes, chunk_size);

            sges[m].addr = reinterpret_cast<uint64_t>(reqs[m]->send.data) + send_offset[m];
            sges[m].length = length;

            wrs[m].wr.rdma.remote_addr = slots[m].addr + send_offset[m];
            
            send_offset[m] += length;

            VLOG(5) << "Sending " << length << " bytes to QP#" << qpidx;
        }

        struct ibv_send_wr *bad_wr;
        DCHECK(ibv_post_send(qpw->qp, wrs, &bad_wr) == 0);
    }
}

void UcclFlow::post_multi_messages(int slot)
{
    auto send_comm_ = &rdma_ctx_->send_comm_;
    auto reqs = send_comm_->fifo_reqs[slot];
    auto slots = send_comm_->base.fifo->elems[slot];
    auto nmsgs = slots[0].nmsgs;

    for (int i = 0; i < nmsgs; i++) post_single_message(reqs[i], slots[i], i);
}

void UcclFlow::retransmit_chunk(struct UCQPWrapper *qpw, struct wr_ex *wr_ex)
{
    struct ibv_send_wr barrier_wr, retr_wr, *bad_wr;
    // Step1: Send a barrier WQE through the original lossy QP.
    barrier_wr.sg_list = nullptr;
    barrier_wr.num_sge = 0;
    barrier_wr.next = nullptr;
    barrier_wr.opcode = IBV_WR_SEND_WITH_IMM;
    barrier_wr.send_flags = IBV_SEND_INLINE;
    // Occasionally post a request with the IBV_SEND_SIGNALED flag.
    if (qpw->signal_cnt_++ % kSignalInterval == 0)
        barrier_wr.send_flags |= IBV_SEND_SIGNALED;
    barrier_wr.imm_data = wr_ex->wr.imm_data;

    DCHECK(ibv_post_send(qpw->qp, &barrier_wr, &bad_wr) == 0);

    // Step2: Use SEND/RECV for retransmission with Retr QP.
    retr_wr = wr_ex->wr;
    struct ibv_sge retr_sge[2];

    uint64_t retr_hdr;
    DCHECK(rdma_ctx_->retr_hdr_pool_->alloc_buff(&retr_hdr) == 0);
    struct retr_chunk_hdr *hdr = reinterpret_cast<struct retr_chunk_hdr *>(retr_hdr);
    hdr->qidx = (uint32_t)(qpw - rdma_ctx_->uc_qps_);
    hdr->remote_addr = wr_ex->wr.wr.rdma.remote_addr;

    retr_sge[0].addr = retr_hdr;
    retr_sge[0].length = sizeof(struct retr_chunk_hdr);
    retr_sge[0].lkey = rdma_ctx_->retr_hdr_pool_->get_lkey();
    
    retr_sge[1] = wr_ex->sge;
    
    retr_wr.wr_id = retr_hdr;
    retr_wr.sg_list = retr_sge;
    retr_wr.num_sge = 2;
    retr_wr.opcode = IBV_WR_SEND_WITH_IMM;
    retr_wr.send_flags = IBV_SEND_SIGNALED;

    DCHECK(ibv_post_send(rdma_ctx_->retr_qp_, &retr_wr, &bad_wr) == 0);
    rdma_ctx_->inflight_retr_chunks_++;

    VLOG(5) << "successfully retransmit chunk for QP#" << (qpw - rdma_ctx_->uc_qps_) 
        << ", remote_addr: " << wr_ex->wr.wr.rdma.remote_addr << ", chunk_size: " << wr_ex->sge.length << ", csn: " << IMMData(ntohl(wr_ex->wr.imm_data)).GetCSN();
}

void UcclFlow::rx_ack(uint64_t pkt_addr)
{
    auto cq_ex = rdma_ctx_->ctrl_cq_ex_;
    
    auto t6 = rdtsc();
    auto *ucclsackh = reinterpret_cast<UcclSackHdr *>(pkt_addr);
    
    auto qpidx = ucclsackh->qpidx.value();
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];
    auto ackno = ucclsackh->ackno.value();

    bool update_sackbitmap = false;

    if (UINT_CSN::uintcsn_seqno_lt(ackno, qpw->pcb.snd_una)) {
        VLOG(5) << "Received old ACK " << ackno << " from QP#" << qpidx << " by Ctrl QP";
    } else if (UINT_CSN::uintcsn_seqno_gt(ackno, qpw->pcb.snd_nxt)) {
        VLOG(5) << "Received ACK for untransmitted data " << "ackno: " << ackno << ", snd_nxt: " 
            << qpw->pcb.snd_nxt.to_uint32() << " from QP#" << qpidx << " by Ctrl QP";
    } else if (UINT_CSN::uintcsn_seqno_eq(ackno, qpw->pcb.snd_una)) {
        VLOG(5) << "Received duplicate ACK " << ackno << " from QP#" << qpidx << " by Ctrl QP";
        update_sackbitmap = true;
        
        qpw->pcb.duplicate_acks++;
        qpw->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();
        
        if (qpw->pcb.duplicate_acks < kFastRexmitDupAckThres) {
            // We have not reached the threshold yet, so we do not do retransmission.
        } else if (qpw->pcb.duplicate_acks == kFastRexmitDupAckThres) {
            // Fast retransmit.
            fast_retransmit(qpw);
        } else {
            // We have already done the fast retransmit, so we are now
            // in the fast recovery phase.
            auto sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
            // We check the SACK bitmap to see if there are more undelivered
            // chunks. In fast recovery mode we get after a fast
            // retransmit, we will retransmit all missing chunks that we
            // find from the SACK bitmap, when enumerating the SACK bitmap
            // for up to sack_bitmap_count ACKs.
            uint32_t index = 0;
            while (sack_bitmap_count && index < kSackBitmapSize && !qpw->txtracking.empty()) {
                auto bucket_idx = index / swift::Pcb::kSackBitmapBucketSize;
                auto sack_bitmap = ucclsackh->sack_bitmap[bucket_idx].value();
                
                auto cursor = index % swift::Pcb::kSackBitmapBucketSize;

                if ((sack_bitmap & (1ULL << cursor)) == 0) {
                    // We found a hole.
                    auto seqno = qpw->pcb.snd_una + index;
                    auto chunk = qpw->txtracking.get_unacked_chunk_from_idx(index);
                    if (seqno == chunk.csn) {
                        auto wr_ex = chunk.wr_ex;
                        retransmit_chunk(qpw, wr_ex);
                    }
                    qpw->pcb.rto_reset();
                } else {
                    sack_bitmap_count--;
                }
                index++;
            }
        }

    } else {
        VLOG(5) << "Received valid ACK " << ackno << " from QP#" << qpidx << " by Ctrl QP";

        update_sackbitmap = true;
        auto num_acked_chunks = UINT_CSN(ackno) - qpw->pcb.snd_una;

        auto t1 = qpw->txtracking.ack_chunks(num_acked_chunks.to_uint32());
        auto remote_queueing_tsc = us_to_cycles(be64toh(ucclsackh->remote_queueing.value()), freq_ghz);
        uint64_t t5;
        if constexpr (kTestNoHWTimestamp)
            t5 = t6;
        else
            t5 = engine_->convert_nic_to_host(t6, ibv_wc_read_completion_ts(cq_ex));

        /// TODO: Congestion control
        auto endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
        auto fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;

        VLOG(5) << "Total: " << to_usec(t6 - t1, freq_ghz) << 
            ", Endpoint delay: " << to_usec(endpoint_delay_tsc, freq_ghz) << 
            ", Fabric delay: " << to_usec(fabric_delay_tsc, freq_ghz);
        
        qpw->pcb.update_rate(rdtsc(), fabric_delay_tsc);

        VLOG(5) << "CC rate: " << qpw->pcb.timely.get_rate_gbps() << " Gbps";

        qpw->pcb.snd_una = ackno;
        qpw->pcb.duplicate_acks = 0;
        qpw->pcb.snd_ooo_acks = 0;
        qpw->pcb.rto_rexmits_consectutive = 0;
        qpw->pcb.rto_maybe_reset();
    }
    
    // For duplicate ACKs and valid ACKs, we may need to update the SACK bitmap at the sender side.
    if (update_sackbitmap) {
        for (int i = 0; i < kSackBitmapSize / swift::Pcb::kSackBitmapBucketSize; i++)
            qpw->pcb.tx_sack_bitmap[i] = ucclsackh->sack_bitmap[i].value();
        qpw->pcb.tx_sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
        qpw->pcb.base_csn = ackno + 1;
    }
}

int UcclFlow::sender_poll_retr_cq(void)
{
    auto cq_ex = rdma_ctx_->retr_cq_ex_;
    int cq_budget = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

    while (1) {
        if (cq_ex->status == IBV_WC_SUCCESS) {
            auto wr_id = cq_ex->wr_id;
            rdma_ctx_->retr_hdr_pool_->free_buff(wr_id);
            rdma_ctx_->inflight_retr_chunks_--;
        } else {
            LOG(ERROR) << "Retr CQ state error: " << cq_ex->status;
        }

        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);

    return cq_budget;
}

int UcclFlow::receiver_poll_retr_cq(void)
{
    auto cq_ex = rdma_ctx_->retr_cq_ex_;
    struct ibv_sge sges[kMaxBatchCQ];
    LIST_HEAD(ack_list);
    int cq_budget = 0;
    int num_post_recv = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

    while (1) {
        if (cq_ex->status == IBV_WC_SUCCESS) {
            rx_retr_chunk(&ack_list);
            num_post_recv++;
        } else {
            LOG(ERROR) << "Retr CQ state error: " << cq_ex->status;
        }

        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);

    // Send coalescing ACKs.
    int num_ack = 0;
    struct list_head *pos, *n;
    uint64_t chunk_addr;
    DCHECK(rdma_ctx_->ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
    list_for_each_safe(pos, n, &ack_list) {
        auto ack_item = list_entry(pos, struct ack_item, ack_link);
        auto qpidx = ack_item->qpidx;
        craft_ack(qpidx, chunk_addr, num_ack++);
        list_del(pos);
    }
    flush_acks(num_ack, chunk_addr);
    if (num_ack == 0)
        rdma_ctx_->ctrl_chunk_pool_->free_buff(chunk_addr);

    // Populate recv work requests for consuming retransmission chunks.
    if (num_post_recv) {
        int i;
        for (i = 0; i < num_post_recv; i++) {
            uint64_t chunk_addr;
            if (rdma_ctx_->retr_chunk_pool_->alloc_buff(&chunk_addr)) {
                VLOG(5) << "Failed to allocate retransmission chunk buffer";
            }
            sges[i].addr = chunk_addr;
            sges[i].length = RetrChunkBuffPool::kRetrChunkSize;
            sges[i].lkey = rdma_ctx_->retr_chunk_pool_->get_lkey();
            retr_wrs_[i].sg_list = &sges[i];
            retr_wrs_[i].num_sge = 1;
            retr_wrs_[i].wr_id = chunk_addr;
        }
        retr_wrs_[i - 1].next = nullptr;
        struct ibv_recv_wr *bad_wr;
        DCHECK(ibv_post_recv(rdma_ctx_->retr_qp_, &retr_wrs_[0], &bad_wr) == 0);
        VLOG(5) << "Posted " << i << " recv requests for Retr QP";
        
        // Restore
        retr_wrs_[i - 1].next = (i == kMaxBatchCQ) ? nullptr : &retr_wrs_[i];
    }
    return cq_budget;
}

int UcclFlow::sender_poll_ctrl_cq(void)
{
    int work = 0;
    while (1) {

        struct ibv_poll_cq_attr poll_cq_attr = {};
        auto cq_ex = rdma_ctx_->ctrl_cq_ex_;
        if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;

        int cq_budget = 0;

        while (1) {
            if (cq_ex->status == IBV_WC_SUCCESS) {
                // Completion for receiving ACKs.
                DCHECK(ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV);
                auto imm_data = ntohl(ibv_wc_read_imm_data(cq_ex));
                auto num_ack = imm_data;
                VLOG(5) << "Receive " << num_ack << " ACKs in one CtrlChunk, Chunk addr: " << cq_ex->wr_id;
                auto chunk_addr = cq_ex->wr_id;
                for (int i = 0; i < num_ack; i++) {
                    auto pkt_addr = chunk_addr + i * CtrlChunkBuffPool::kPktSize;
                    rx_ack(pkt_addr);
                }
                post_ctrl_rq_cnt_++;
                rdma_ctx_->ctrl_chunk_pool_->free_buff(chunk_addr);
            } else {
                LOG(ERROR) << "Ctrl CQ state error: " << cq_ex->status;
            }

            if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
        }

        ibv_end_poll(cq_ex);

        work += cq_budget;
    }
    
    return work;
}

int UcclFlow::receiver_poll_ctrl_cq(void) 
{   
    int work = 0;
    while (1) {

        struct ibv_poll_cq_attr poll_cq_attr = {};
        auto cq_ex = rdma_ctx_->ctrl_cq_ex_;
        if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;

        int cq_budget = 0;

        while (1) {
            
            if (cq_ex->status == IBV_WC_SUCCESS) {
                // Completion for sending ACKs.
                DCHECK(ibv_wc_read_opcode(cq_ex) == IBV_WC_SEND);
                auto chunk_addr = cq_ex->wr_id;
                rdma_ctx_->ctrl_chunk_pool_->free_buff(chunk_addr);
            } else {
                LOG(ERROR) << "Ctrl CQ state error: " << cq_ex->status;
            }

            if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
        }

        ibv_end_poll(cq_ex);

        work += cq_budget;
    }

    return work;
}

void UcclFlow::burst_timing_wheel(void)
{
    auto wheel = &rdma_ctx_->wheel_;
    struct ibv_send_wr *bad_wr;

    wheel->reap(rdtsc());

    auto num_chunks = std::min(kMaxBatchPost, (uint32_t)wheel->ready_queue_.size());
    
    for (auto i = 0; i < num_chunks; i++) {
        struct wr_ex *wr_ex = reinterpret_cast<struct wr_ex *>(wheel->ready_queue_.front().sslot_);
        auto qpw = &rdma_ctx_->uc_qps_[wr_ex->qpidx];

        DCHECK(ibv_post_send(qpw->qp, &wr_ex->wr, &bad_wr) == 0);

        VLOG(5) << "Burst send: csn: " << wr_ex->wr.imm_data << " with QP#" << wr_ex->qpidx;

        // Track this chunk.
        IMMData imm_data(ntohl(wr_ex->wr.imm_data));
        qpw->txtracking.track_chunk(wr_ex->req, imm_data.GetCSN(), wr_ex, rdtsc());

        qpw->in_wheel_cnt_--;

        wheel->ready_queue_.pop_front();
    }
}

void UcclFlow::try_update_csn(struct UCQPWrapper *qpw)
{
    while (!qpw->rxtracking.ready_csn_.empty() && 
        static_cast<uint32_t>(*qpw->rxtracking.ready_csn_.begin()) == qpw->pcb.rcv_nxt.to_uint32()) {
        auto csn = *qpw->rxtracking.ready_csn_.begin();
        qpw->rxtracking.ready_csn_.erase(qpw->rxtracking.ready_csn_.begin());
        
        // Data is already DMAed to the application buffer.
        // Nothing more to do.

        qpw->pcb.advance_rcv_nxt();
        VLOG(5) << "try_update_csn:" << " rcv_nxt: " << qpw->pcb.rcv_nxt.to_uint32() << " from QP#" << qpw - rdma_ctx_->uc_qps_;
        qpw->pcb.sack_bitmap_shift_left_one();
        qpw->pcb.barrier_bitmap_shift_left_one();
    }
}

void UcclFlow::rx_barrier(struct list_head *ack_list)
{
    VLOG(5) << "rx_barrier";
    auto cq_ex = rdma_ctx_->cq_ex_;

    DCHECK(rdma_ctx_->is_send_ == false);

    auto recv_comm = &rdma_ctx_->recv_comm_;
    auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));
    auto qp_num = ibv_wc_read_qp_num(cq_ex);
    auto qpidx = rdma_ctx_->qpn2idx_[qp_num];
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];

    auto nchunks = imm_data.GetNCHUNK();
    auto csn = imm_data.GetCSN();
    auto rid = imm_data.GetRID();
    auto mid = imm_data.GetMID();

    VLOG(5) << "Receive barrier: (csn, rid, mid): " << csn << ", " << rid << ", " << mid << " from QP#" << qpidx;

    // Compare CSN with the expected CSN.
    auto ecsn = qpw->pcb.rcv_nxt;

    auto distance = UINT_CSN(csn) - ecsn;

    if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
        VLOG(5) << "Barrier too far ahead. Dropping as we can't handle SACK. "
                    << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
        // Try to remove the pending retransmission chunk if exists.
        auto pending_retr_chunk = qpw->pcb.pending_retr_chunks.find(distance.to_uint32() + qpw->pcb.shift_count);
        if (pending_retr_chunk != qpw->pcb.pending_retr_chunks.end()) {
            auto chunk_addr = pending_retr_chunk->second.chunk_addr;
            rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
            qpw->pcb.pending_retr_chunks.erase(pending_retr_chunk);
            VLOG(5) << "Remove pending retransmission chunk for QP#" << qpidx;
        }
        return;
    }

    if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn))
    {
        // Original chunk is already received. This barrier is invalid.
        // Try to remove the pending retransmission chunk if exists.
        auto pending_retr_chunk = qpw->pcb.pending_retr_chunks.find(distance.to_uint32() + qpw->pcb.shift_count);
        if (pending_retr_chunk != qpw->pcb.pending_retr_chunks.end()) {
            auto chunk_addr = pending_retr_chunk->second.chunk_addr;
            rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
            qpw->pcb.pending_retr_chunks.erase(pending_retr_chunk);
            VLOG(5) << "Remove pending retransmission chunk for QP#" << qpidx;
        }
        return;
    }

    auto bitmap_bucket_idx = distance.to_uint32() / swift::Pcb::kSackBitmapBucketSize;
    auto cursor = distance.to_uint32() % swift::Pcb::kSackBitmapBucketSize;
    auto sack_bitmap = &qpw->pcb.sack_bitmap[bitmap_bucket_idx];
    auto barrier_bitmap = &qpw->pcb.barrier_bitmap[bitmap_bucket_idx];

    if ((*sack_bitmap & (1ULL << cursor))) {
        // Original chunk is already received. This barrier is invalid.
        // Try to remove the pending retransmission chunk if exists.
        auto pending_retr_chunk = qpw->pcb.pending_retr_chunks.find(distance.to_uint32() + qpw->pcb.shift_count);
        if (pending_retr_chunk != qpw->pcb.pending_retr_chunks.end()) {
            auto chunk_addr = pending_retr_chunk->second.chunk_addr;
            rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
            qpw->pcb.pending_retr_chunks.erase(pending_retr_chunk);
            VLOG(5) << "Remove pending retransmission chunk for QP#" << qpidx;
        }
        return;
    }

    if ((*barrier_bitmap & (1ULL << cursor))) {
        // Duplicate barrier. This barrier is invalid.
        VLOG(5) << "Received duplicate barrier " << csn << " from QP#" << qpidx;
        return;
    }

    // This barrier is valid.
    qpw->pcb.barrier_bitmap_bit_set(distance.to_uint32());

    // Handle pending retransmission chunk waiting for this barrier (if exists).
    auto pending_retr_chunk = qpw->pcb.pending_retr_chunks.find(distance.to_uint32() + qpw->pcb.shift_count);
    if (pending_retr_chunk == qpw->pcb.pending_retr_chunks.end()) {
        // No pending retransmission chunk.
        VLOG(5) << "Barrier arrived without pending retransmission chunk for QP#" << qpidx;
        return;
    }

    VLOG(5) << "Barrier found a pending retransmission chunk for QP#" << qpidx;

    // We found a pending retransmission chunk.
    imm_data = IMMData(pending_retr_chunk->second.imm_data);
    auto chunk_addr = pending_retr_chunk->second.chunk_addr;
    auto chunk_len = pending_retr_chunk->second.chunk_len;
    auto remote_addr = pending_retr_chunk->second.remote_addr;

    // Accept this retransmission chunk.
    memcpy(reinterpret_cast<void *>(remote_addr), reinterpret_cast<void *>(chunk_addr + sizeof(struct retr_chunk_hdr)), chunk_len);

    qpw->rxtracking.ready_csn_.insert(csn);

    /// FIXME: Should we update the timestamp here?

    qpw->pcb.sack_bitmap_bit_set(distance.to_uint32());

    // Locate request by rid
    auto req = rdma_ctx_->get_request_by_id(rid, &recv_comm->base);
    auto *msg_size = &req->recv.elems[mid].size;
    uint32_t *received_bytes = req->recv.received_bytes;
    received_bytes[mid] += chunk_len;

    if (nchunks) {
        // Tx size < Rx size, adjust the meesage size using the information carried by the last chunk.
        auto actual_size = kChunkSize * (nchunks - 1) + chunk_len;
        *msg_size = actual_size;
        req->ureq->data_len[mid] = actual_size;
    }

    if (*msg_size == received_bytes[mid]) req->recv.fin_msg++;
    if (req->recv.fin_msg == req->nmsgs) { // This request (may contain multiple messages) is complete.
        VLOG(3) << "Request complete (" << req->nmsgs << " messages)";
        auto poll_ctx = req->poll_ctx;
        // Wakeup app thread.
        {
            std::lock_guard<std::mutex> lock(poll_ctx->mu);
            poll_ctx->done = true;
            poll_ctx->cv.notify_one();
        }
        // Free the request.
        rdma_ctx_->free_request(req);
    }

    try_update_csn(qpw);

    /// FIXME: Should we send ACK immediately here?
    if (list_empty(&qpw->ack.ack_link))
        list_add_tail(&qpw->ack.ack_link, ack_list);

    rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);

    qpw->pcb.pending_retr_chunks.erase(pending_retr_chunk);

}

void UcclFlow::rx_retr_chunk(struct list_head *ack_list)
{
    VLOG(5) << "rx_retr_chunk";
    auto cq_ex = rdma_ctx_->retr_cq_ex_;

    auto now = rdtsc();

    DCHECK(rdma_ctx_->is_send_ == false);

    auto recv_comm = &rdma_ctx_->recv_comm_;
    auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));

    auto nchunks = imm_data.GetNCHUNK();
    auto csn = imm_data.GetCSN();
    auto rid = imm_data.GetRID();
    auto mid = imm_data.GetMID();

    VLOG(5) << "Received retransmission chunk: (csn, rid, mid): " << csn << ", " << rid << ", " << mid << " from Retr QP";

    auto chunk_addr = cq_ex->wr_id;
    auto chunk_len = ibv_wc_read_byte_len(cq_ex) - sizeof(struct retr_chunk_hdr);

    struct retr_chunk_hdr *hdr = reinterpret_cast<struct retr_chunk_hdr *>(chunk_addr);
    
    auto qpw = &rdma_ctx_->uc_qps_[hdr->qidx];
    auto pcb = &qpw->pcb;

    // Compare CSN with the expected CSN.
    auto ecsn = pcb->rcv_nxt;
    auto distance = UINT_CSN(csn) - ecsn;

    if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
        VLOG(5) << "Chunk too far ahead. Dropping as we can't handle SACK. "
                    << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
        rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
        return;
    }

    if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
        // Original chunk is already received.
        rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
        VLOG(5) << "Original chunk is already received. Dropping retransmission chunk for QP#" << hdr->qidx;
        return;
    }
    
    auto bitmap_bucket_idx = distance.to_uint32() / swift::Pcb::kSackBitmapBucketSize;
    auto cursor = distance.to_uint32() % swift::Pcb::kSackBitmapBucketSize;
    auto sack_bitmap = &pcb->sack_bitmap[bitmap_bucket_idx];
    
    if ((*sack_bitmap & (1ULL << cursor))) {
        // Original chunk is already received.
        rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
        VLOG(5) << "Original chunk is already received. Dropping retransmission chunk for QP#" << hdr->qidx;
        return;
    }
    
    // Check barrier bitmap first. We can only receive this chunk 
    // when a corresponding barrier has arrived.
    auto barrier_btimap = &pcb->barrier_bitmap[bitmap_bucket_idx];

    if ((*barrier_btimap & (1ULL << cursor)) == 0) {
        // The corresponding barrier has not arrived yet.
        // Store this retransmission chunk.
        pcb->pending_retr_chunks[distance.to_uint32() + pcb->shift_count] = 
        {hdr->remote_addr, chunk_addr, (uint32_t)chunk_len, imm_data.GetImmData()};
        VLOG(5) << "Wait for the corresponding barrier for QP#" << hdr->qidx;
    } else {
        VLOG(5) << "This retransmission chunk is accepted!!!";
        // Accept this retransmission chunk.
        memcpy(reinterpret_cast<void *>(hdr->remote_addr), reinterpret_cast<void *>(chunk_addr), chunk_len);

        qpw->rxtracking.ready_csn_.insert(csn);

        /// FIXME: Should we update the timestamp here?

        pcb->sack_bitmap_bit_set(distance.to_uint32());

        // Locate request by rid
        auto req = rdma_ctx_->get_request_by_id(rid, &recv_comm->base);
        auto *msg_size = &req->recv.elems[mid].size;
        uint32_t *received_bytes = req->recv.received_bytes;
        received_bytes[mid] += chunk_len;

        if (nchunks) {
            // Tx size < Rx size, adjust the meesage size using the information carried by the last chunk.
            auto nchunk = imm_data.GetNCHUNK();
            auto actual_size = kChunkSize * (nchunks - 1) + chunk_len;
            *msg_size = actual_size;
            req->ureq->data_len[mid] = actual_size;
        }

        if (*msg_size == received_bytes[mid]) req->recv.fin_msg++;
        if (req->recv.fin_msg == req->nmsgs) { // This request (may contain multiple messages) is complete.
            VLOG(3) << "Request complete (" << req->nmsgs << " messages)";
            auto poll_ctx = req->poll_ctx;
            // Wakeup app thread.
            {
                std::lock_guard<std::mutex> lock(poll_ctx->mu);
                poll_ctx->done = true;
                poll_ctx->cv.notify_one();
            }
            // Free the request.
            rdma_ctx_->free_request(req);
        }

        try_update_csn(qpw);

        /// FIXME: Should we send ACK immediately here?
        if (list_empty(&qpw->ack.ack_link))
            list_add_tail(&qpw->ack.ack_link, ack_list);

        rdma_ctx_->retr_chunk_pool_->free_buff(chunk_addr);
        return;
    }
}

void UcclFlow::rx_chunk(struct list_head *ack_list)
{
    VLOG(5) << "rx_chunk";
    auto cq_ex = rdma_ctx_->cq_ex_;

    auto now = rdtsc();
    
    DCHECK(rdma_ctx_->is_send_ == false);
    
    auto recv_comm = &rdma_ctx_->recv_comm_;
    auto byte_len = ibv_wc_read_byte_len(cq_ex);
    auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));
    auto qp_num = ibv_wc_read_qp_num(cq_ex);
    auto qpidx = rdma_ctx_->qpn2idx_[qp_num];
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];

    auto nchunks = imm_data.GetNCHUNK();
    auto csn = imm_data.GetCSN();
    auto rid = imm_data.GetRID();
    auto mid = imm_data.GetMID();

    VLOG(5) << "Received chunk: (byte_len, csn, rid, mid): " << byte_len << ", " << csn << ", " << rid << ", " << mid << " from QP#" << qpidx;

    // Compare CSN with the expected CSN.
    auto ecsn = qpw->pcb.rcv_nxt;

    auto distance = UINT_CSN(csn) - ecsn;
    
    // It's impossible to receive a chunk with a CSN less than the expected CSN here.
    // That is:
    // 0 --- csn --- ecsn --- MAX
    // or
    // 0 --- ecsn ------ csn --- MAX

    if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
        VLOG(4) << "Chunk too far ahead. Dropping as we can't handle SACK. "
                    << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
        return;
    }

    if ((UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn))) {
        VLOG(4) << "Chunk lag behind. Dropping as we can't handle SACK. "
                    << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    }

    /* 
     * No need for the following code to check as we can only accept a retransmission chunk when 
     * the barrier after this chunk has arrived.
    */
    
    // auto bitmap_bucket_idx = distance / swift::Pcb::kSackBitmapBucketSize;
    // auto cursor = distance % swift::Pcb::kSackBitmapBucketSize;
    // auto sack_bitmap = &qpw->pcb.sack_bitmap[bitmap_bucket_idx];
    // DCHECK(!(*sack_bitmap & (1ULL << cursor)));

    qpw->rxtracking.ready_csn_.insert(csn);

    // Always use the latest timestamp.
    if constexpr (kTestNoHWTimestamp)
        qpw->pcb.t_remote_nic_rx = now;
    else
        qpw->pcb.t_remote_nic_rx = ibv_wc_read_completion_ts(cq_ex);

    qpw->pcb.sack_bitmap_bit_set(distance.to_uint32());

    // Locate request by rid
    auto req = rdma_ctx_->get_request_by_id(rid, &recv_comm->base);
    auto *msg_size = &req->recv.elems[mid].size;
    uint32_t *received_bytes = req->recv.received_bytes;
    received_bytes[mid] += byte_len;

    if (nchunks) {
        // Tx size < Rx size, adjust the meesage size using the information carried by the last chunk.
        auto actual_size = kChunkSize * (nchunks - 1) + byte_len;
        *msg_size = actual_size;
        req->ureq->data_len[mid] = actual_size;
    }

    if (*msg_size == received_bytes[mid]) req->recv.fin_msg++;
    if (req->recv.fin_msg == req->nmsgs) { // This request (may contain multiple messages) is complete.
        VLOG(3) << "Request complete (" << req->nmsgs << " messages)";
        auto poll_ctx = req->poll_ctx;
        // Wakeup app thread.
        {
            std::lock_guard<std::mutex> lock(poll_ctx->mu);
            poll_ctx->done = true;
            poll_ctx->cv.notify_one();
        }
        // Free the request.
        rdma_ctx_->free_request(req);
    }

    try_update_csn(qpw);

    if (distance.to_uint32())
        qpw->rxtracking.encounter_ooo();
    
    qpw->rxtracking.cumulate_wqe();
    qpw->rxtracking.cumulate_bytes(byte_len);

    if (list_empty(&qpw->ack.ack_link))
        list_add_tail(&qpw->ack.ack_link, ack_list);

    // Send ACK if needed.
    if (qpw->rxtracking.need_imm_ack()) {
        uint64_t chunk_addr;
        DCHECK(rdma_ctx_->ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
        craft_ack(qpidx, chunk_addr, 0);
        flush_acks(1, chunk_addr);

        qpw->rxtracking.clear_imm_ack();
        list_del(&qpw->ack.ack_link);
    }
}

void UcclFlow::flush_acks(int num_ack, uint64_t chunk_addr)
{
    if (num_ack == 0) return;

    struct ibv_sge sge = {
        .addr = chunk_addr,
        .length = CtrlChunkBuffPool::kPktSize * num_ack,
        .lkey = rdma_ctx_->ctrl_chunk_pool_->get_lkey(),
    };

    tx_ack_wr_.sg_list = &sge;
    
    // For future free.
    tx_ack_wr_.wr_id = chunk_addr;
    
    // Tell sender how many acks are in this wqe.
    tx_ack_wr_.imm_data = htonl(num_ack);

    struct ibv_send_wr *bad_wr;
    DCHECK(ibv_post_send(rdma_ctx_->ctrl_qp_, &tx_ack_wr_, &bad_wr) == 0);
}

void UcclFlow::craft_ack(int qpidx, uint64_t chunk_addr, int num_sge)
{
    uint64_t pkt_addr = chunk_addr + CtrlChunkBuffPool::kPktSize * num_sge;
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];
    auto *ucclsackh = reinterpret_cast<UcclSackHdr* >(pkt_addr);

    ucclsackh->ackno = be32_t(qpw->pcb.ackno().to_uint32());
    ucclsackh->qpidx = be16_t(qpidx);

    auto t4 = rdtsc();
    uint64_t t2;
    if constexpr (kTestNoHWTimestamp)
        t2 = qpw->pcb.t_remote_nic_rx;
    else
        t2 = engine_->convert_nic_to_host(t4, rdma_ctx_->uc_qps_[qpidx].pcb.t_remote_nic_rx);

    ucclsackh->remote_queueing = be64_t(to_usec(t4 - t2, freq_ghz));

    for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                               sizeof(UcclSackHdr::sack_bitmap[0]);
         ++i) {
        ucclsackh->sack_bitmap[i] = be64_t(qpw->pcb.sack_bitmap[i]);
    }
    ucclsackh->sack_bitmap_count = be16_t(qpw->pcb.sack_bitmap_count);

    VLOG(5) << "craft_ack: seqno: " << qpw->pcb.seqno().to_uint32() << ", ackno: " << qpw->pcb.ackno().to_uint32()  << " to QP#" << qpidx;
}

void UcclFlow::test_rc_poll_cq(void)
{
    auto cq_ex = rdma_ctx_->cq_ex_;
    int cq_budget = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return;

    while (1) {

        DCHECK(cq_ex->status == IBV_WC_SUCCESS);

        auto opcode = ibv_wc_read_opcode(cq_ex);

        if (opcode != IBV_WC_RDMA_READ) {
            auto qp_num = ibv_wc_read_qp_num(cq_ex);
            auto qpidx = rdma_ctx_->qpn2idx_[qp_num];
            VLOG(5) << "Event from UC QP#" << qpidx << ", wr_id: " << cq_ex->wr_id;
        }

        auto send = opcode == IBV_WC_RDMA_WRITE;
        struct NetCommBase *comm_base = send ? &rdma_ctx_->send_comm_.base : &rdma_ctx_->recv_comm_.base;
        
        auto wr_id = cq_ex->wr_id;
        auto req0 = rdma_ctx_->get_request_by_id(wr_id & 0xff, comm_base);
        auto fin = true;

        if (opcode == IBV_WC_RDMA_WRITE) {
            // For sender, one FlowRequest for one message.
            for (int i = 0; i < req0->nmsgs; i++) {
                auto req = rdma_ctx_->get_request_by_id((wr_id >> (i * 8)) & 0xff, comm_base);
                if (--req->events)
                    fin = false;
                else if (i) {
                    // Wakeup app thread.
                    {
                        std::lock_guard<std::mutex> lock(req->poll_ctx->mu);
                        req->poll_ctx->done = true;
                        req->poll_ctx->cv.notify_one();
                    }
                    rdma_ctx_->free_request(req);
                }
                VLOG(5) << req->events << " events left for request " << rdma_ctx_->get_request_id(req, comm_base);
            }
        } else {
            DCHECK(opcode == IBV_WC_RECV_RDMA_WITH_IMM || opcode == IBV_WC_RDMA_READ);
            // For receiver, only one FlowRequest for multiple messages.
            if (--req0->events)
                fin = false;
        }
        
        if (fin) {
            VLOG(3) << "Request complete (" << req0->nmsgs << " messages)";
            // Wakeup app thread.
            {
                std::lock_guard<std::mutex> lock(req0->poll_ctx->mu);
                req0->poll_ctx->done = true;
                req0->poll_ctx->cv.notify_one();
            }
            // Free the first request.
            rdma_ctx_->free_request(req0);
        }

        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);
}

int UcclFlow::sender_poll_uc_cq(void)
{
    auto cq_ex = rdma_ctx_->cq_ex_;
    int cq_budget = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;
    
    while (1) {
        if (cq_ex->status != IBV_WC_SUCCESS)
            LOG(ERROR) << "UC CQ state error: " << cq_ex->status << " from QP:" << ibv_wc_read_qp_num(cq_ex);
        
        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);

    return cq_budget;
}

void UcclFlow::receiver_check_ctrl_rq(void)
{
    // Do nothing.
}

void UcclFlow::sender_check_ctrl_rq(bool force)
{
    // Populate recv work requests for consuming control packets.
    while (post_ctrl_rq_cnt_ >= kPostRQThreshold) {
        struct ibv_recv_wr *bad_wr;
        for (int i = 0; i < kPostRQThreshold; i++) {
            uint64_t chunk_addr;
            DCHECK(rdma_ctx_->ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
            rx_ack_sges_[i].addr = chunk_addr;
            rx_ack_wrs_[i].wr_id = chunk_addr;
        }
        DCHECK(ibv_post_recv(rdma_ctx_->ctrl_qp_, &rx_ack_wrs_[0], &bad_wr) == 0);
        VLOG(5) << "Posted " << post_ctrl_rq_cnt_ << " recv requests for Ctrl QP";
        post_ctrl_rq_cnt_ -= kPostRQThreshold;
    }

    if (force && post_ctrl_rq_cnt_) {
        struct ibv_recv_wr *bad_wr;
        for (int i = 0; i < post_ctrl_rq_cnt_; i++) {
            uint64_t chunk_addr;
            DCHECK(rdma_ctx_->ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
            rx_ack_sges_[i].addr = chunk_addr;
            rx_ack_wrs_[i].wr_id = chunk_addr;
        }
        rx_ack_wrs_[post_ctrl_rq_cnt_ - 1].next = nullptr;
        DCHECK(ibv_post_recv(rdma_ctx_->ctrl_qp_, &rx_ack_wrs_[0], &bad_wr) == 0);
        VLOG(5) << "Posted " << post_ctrl_rq_cnt_ << " recv requests for Ctrl QP";
        rx_ack_wrs_[post_ctrl_rq_cnt_ - 1].next = &rx_ack_wrs_[post_ctrl_rq_cnt_];
        post_ctrl_rq_cnt_ = 0;
    }
}

void UcclFlow::check_srq(bool force)
{
    while (post_srq_cnt_ >= kPostRQThreshold) {
        struct ibv_recv_wr *bad_wr;
        DCHECK(ibv_post_srq_recv(rdma_ctx_->srq_, &imm_wrs_[0], &bad_wr) == 0);
        VLOG(5) << "Posted " << post_srq_cnt_ << " recv requests for SRQ";
        post_srq_cnt_ -= kPostRQThreshold;
    }

    if (force && post_srq_cnt_) {
        struct ibv_recv_wr *bad_wr;
        imm_wrs_[post_srq_cnt_ - 1].next = nullptr;
        DCHECK(ibv_post_srq_recv(rdma_ctx_->srq_, &imm_wrs_[0], &bad_wr) == 0);
        VLOG(5) << "Posted " << post_srq_cnt_ << " recv requests for SRQ";
        imm_wrs_[post_srq_cnt_ - 1].next = &imm_wrs_[post_srq_cnt_];
        post_srq_cnt_ = 0;
    }
}

int UcclFlow::receiver_poll_uc_cq(void)
{
    auto cq_ex = rdma_ctx_->cq_ex_;
    LIST_HEAD(ack_list);
    int cq_budget = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;
    
    while (1) {
        if (cq_ex->status == IBV_WC_SUCCESS) {
            auto opcode = ibv_wc_read_opcode(cq_ex);
            if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
                if constexpr (!kTestLoss)
                    rx_chunk(&ack_list);
                else {
                    DCHECK(kTestLossRate > 0);
                    auto drop_period = (uint32_t)(1 / kTestLossRate);
                    static uint32_t drop_cnt = 0;
                    if (drop_cnt++ % drop_period == 0) {
                        VLOG(5) << "Drop a chunk";
                    } else {
                        rx_chunk(&ack_list);
                    }
                }
                post_srq_cnt_++;
            } else if (opcode == IBV_WC_RECV) {
                DCHECK(ibv_wc_read_byte_len(cq_ex) == 0);
                rx_barrier(&ack_list);
                post_srq_cnt_++;
            } else if (opcode == IBV_WC_RDMA_READ) {
                auto req = rdma_ctx_->get_request_by_id(cq_ex->wr_id, 
                    &rdma_ctx_->recv_comm_.base);
                auto poll_ctx = req->poll_ctx;
                // Wakeup app thread.
                {
                    std::lock_guard<std::mutex> lock(poll_ctx->mu);
                    poll_ctx->done = true;
                    poll_ctx->cv.notify_one();
                }
                rdma_ctx_->free_request(req);
                VLOG(3) << "Flush operation complete";
            }
        } else {
            LOG(ERROR) << "UC CQ state error: " << cq_ex->status << " from QP:" << ibv_wc_read_qp_num(cq_ex);
        }
        
        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);
    
    // Send coalescing ACKs.
    int num_ack = 0;
    struct list_head *pos, *n;
    uint64_t chunk_addr;
    DCHECK(rdma_ctx_->ctrl_chunk_pool_->alloc_buff(&chunk_addr) == 0);
    list_for_each_safe(pos, n, &ack_list) {
        auto ack_item = list_entry(pos, struct ack_item, ack_link);
        auto qpidx = ack_item->qpidx;
        craft_ack(qpidx, chunk_addr, num_ack++);
        list_del(pos);
    }
    flush_acks(num_ack, chunk_addr);
    if (num_ack == 0)
        rdma_ctx_->ctrl_chunk_pool_->free_buff(chunk_addr);

    return cq_budget;
}

bool UcclFlow::poll_fifo_cq(void)
{
    FlowRequest *req = nullptr;
    auto cq = rdma_ctx_->fifo_cq_;
    struct ibv_wc wcs[kMaxBatchCQ];
    int nb_cqe = ibv_poll_cq(cq, kMaxBatchCQ, wcs);
    if (nb_cqe <= 0) return false;
    
    rdma_ctx_->fifo_cq_polling_ = false;
    return true;
}

bool UcclFlow::tx_messages(Channel::Msg &tx_work) {
    auto data = tx_work.tx.data;
    auto size = tx_work.tx.size;
    auto mr = tx_work.tx.mr;
    auto poll_ctx = tx_work.poll_ctx;
    auto ureq = tx_work.ureq;

    auto send_comm_ = &rdma_ctx_->send_comm_;

    int slot = send_comm_->fifo_head % kMaxReq;
    auto reqs = send_comm_->fifo_reqs[slot];
    auto rem_fifo = send_comm_->base.fifo;
    volatile struct FifoItem *slots = rem_fifo->elems[slot];

    auto idx = send_comm_->fifo_head + 1;
    if (slots[0].idx != idx) {
        VLOG(5) << "Receiver is not ready, pending this tx work.";
        return false;
    }
    
    // Wait until all slots are ready
    auto nmsgs = slots[0].nmsgs;
    for (int i = 1; i < nmsgs; i++) while(slots[i].idx != idx) {}

    VLOG(5) << "Receiver is ready to receive";

    __sync_synchronize();

    for (int i = 0; i < nmsgs; i++) {
        if (reqs[i] != nullptr) continue;
        DCHECK(!(slots[i].size < 0 || slots[i].addr == 0 || slots[i].rkey == 0));
        
        struct FlowRequest *req = rdma_ctx_->get_request(&send_comm_->base);
        DCHECK(req);

        if (size > slots[i].size) {
            // Can't send more than what the receiver can receive.
            size = slots[i].size;
        }
        
        if (ureq) {
            req->ureq = size < slots[i].size ? ureq : nullptr;
            // Adjust expected size.
            ureq->data_len[0] = size;
        }

        req->type = FlowRequest::SEND;
        req->nmsgs = nmsgs;
        req->poll_ctx = poll_ctx;
        req->send.size = size;
        req->send.sent_offset = 0;
        req->send.data = data;
        req->send.lkey = mr->lkey;

        if constexpr (kTestRC)
            req->events = kTestRCEntropy;

        // Track this request.
        reqs[i] = req;

        // If this is a multi-recv, send only when all requests have matched.
        for (int i = 0; i < nmsgs; i++) {
            if (reqs[i] == nullptr) return true;
        }

        if constexpr (kTestRC)
            test_rc_post_multi_messages(slot);
        else
            post_multi_messages(slot);

        memset((void*)slots, 0, sizeof(struct FifoItem));
        memset(reqs, 0, kMaxRecv * sizeof(struct FlowRequest *));

        send_comm_->fifo_head++;
        return true;
    }
    return true;
}

bool UcclFlow::periodic_check() 
{
    if constexpr (kTestNoRTO)
        return true;
    
    for (int i = 0; i < kPortEntropy; i++) {
        auto qpw = &rdma_ctx_->uc_qps_[i];
        qpw->pcb.rto_advance();

        // TODO(ilias): send RST packet, indicating removal of the flow.
        if (qpw->pcb.max_rto_rexmits_consectutive_reached()) {
            VLOG(5) << "Max RTO retransmits reached. Closing flow.";
        }

        if (qpw->pcb.rto_expired()) {
            rto_retransmit(qpw);
        }
    }
    return true;
}

void UcclFlow::__retransmit(struct UCQPWrapper *qpw, bool rto)
{
    /// TODO: We should throttle the volume of retransmission. 
    /// Currently, we hard limit the number of inflight retransmission chunks.
    if (rdma_ctx_->inflight_retr_chunks_ > kMaxInflightRetrChunks || qpw->txtracking.empty())
        return;

    // Case#1: SACK bitmap at the sender side is empty. Retransmit the oldest unacked chunk.
    auto sack_bitmap_count = qpw->pcb.tx_sack_bitmap_count;
    if (!sack_bitmap_count) {
        auto chunk = qpw->txtracking.get_oldest_unacked_chunk();
        auto wr_ex = chunk.wr_ex;
        retransmit_chunk(qpw, wr_ex);
        qpw->pcb.rto_reset();
        if (rto) {
            qpw->pcb.rto_rexmits++;
            qpw->pcb.rto_rexmits_consectutive++;
        } else {
            qpw->pcb.fast_rexmits++;
        }
        return;
    }
    
    // Case#2: Retransmit the unacked chunks according to the SACK bitmap.
    bool done = false;
    auto base_csn = UINT_CSN(qpw->pcb.base_csn);

    uint32_t index = 0;
    while (sack_bitmap_count && index < kSackBitmapSize && !qpw->txtracking.empty()) {
        auto bucket_idx = index / swift::Pcb::kSackBitmapBucketSize;
        auto sack_bitmap = qpw->pcb.tx_sack_bitmap[bucket_idx];
        
        auto cursor = index % swift::Pcb::kSackBitmapBucketSize;

        if ((sack_bitmap & (1ULL << cursor)) == 0) {
            // We found a hole.
            auto seqno = base_csn + index;
            auto chunk = qpw->txtracking.get_unacked_chunk_from_idx(index);
            if (seqno == chunk.csn) {
                auto wr_ex = chunk.wr_ex;
                retransmit_chunk(qpw, wr_ex);
                done = true;
            } else {
                // This bit is stale and its corresponding chunk is already acked.
                // Do nothing.
            }
        } else {
            sack_bitmap_count--;
        }
        index++;
    }

    qpw->pcb.rto_reset();
    if (done) {
        if (rto) {
            qpw->pcb.rto_rexmits++;
            qpw->pcb.rto_rexmits_consectutive++;
        } else {
            qpw->pcb.fast_rexmits++;
        }
    }
}

void UcclRDMAEngine::test_rc_handle_completion(void)
{
    for (auto it = fifo_cq_list_.begin(); it != fifo_cq_list_.end();) {
        auto flow = *it;
        if (flow->poll_fifo_cq()) {
            it = fifo_cq_list_.erase(it);
        } else {
            it++;
        }
    }

    for (auto flow: active_flows_map_) {
        flow.second->test_rc_poll_cq();
    }
}

void UcclRDMAEngine::handle_completion(void) 
{
    int work = 0;
    // First, poll the CQ for Ctrl QPs.
    for (auto flow: active_flows_map_) {
        work += flow.second->poll_ctrl_cq();
    }

    // Second, poll FIFO CQ.
    for (auto it = fifo_cq_list_.begin(); it != fifo_cq_list_.end();) {
        auto flow = *it;
        if (flow->poll_fifo_cq()) {
            it = fifo_cq_list_.erase(it);
        } else {
            it++;
        }
    }

    // Third, poll the CQ for Retr QP.
    for (auto flow: active_flows_map_) {
        work += flow.second->poll_retr_cq();
    }
    
    // Last, poll the CQ for UC QPs.
    for (auto flow: active_flows_map_) {
        work += flow.second->poll_uc_cq();
        flow.second->check_srq(!!work);
        flow.second->check_ctrl_rq(!!work);
    }
}

void UcclRDMAEngine::handle_rx_work(void) 
{
    Channel::Msg rx_work;
    int budget = 0;
    while (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) ==
        1) {
        if (rx_work.opcode == Channel::Msg::kRx) {
            VLOG(3) << "[Engine#" << engine_idx_ << "] " << "kRX";
            active_flows_map_[rx_work.flow_id]->app_supply_rx_buf(rx_work);
        } else {
            DCHECK(rx_work.opcode == Channel::Msg::kFlush);
            VLOG(3) << "[Engine#" << engine_idx_ << "] " << "kFlush";
            active_flows_map_[rx_work.flow_id]->flush_rx_buf(rx_work);
        }
        if (++budget == kMaxRxWork) break;
    }
}

void UcclRDMAEngine::handle_tx_work(void)
{
    // Handle pending tx work first.
    for (auto it = pending_tx_work_.begin(); it != pending_tx_work_.end();) {
        auto tx_work = *it;
        auto flow = active_flows_map_[tx_work.flow_id];
        if (flow == nullptr) {
            LOG(ERROR) << "Flow not found";
            it = pending_tx_work_.erase(it);
            continue;
        }

        if (!flow->tx_messages(tx_work)) {
            // All tx works are processed in order, so if one tx work blocks,
            // the following tx works will also block.
            return;
        } else {
            // Good, the tx work is done.
            it = pending_tx_work_.erase(it);
        }
    }

    // Then, handle new tx work.
    Channel::Msg tx_work;
    if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) ==
        1) {
        // Make data written by the app thread visible to the engine.
        std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);

        VLOG(3) << "[Engine#" << engine_idx_ << "] " << "kTX";

        if (!active_flows_map_[tx_work.flow_id]->tx_messages(tx_work))
            pending_tx_work_.push_back(tx_work);
    }
}

void UcclRDMAEngine::handle_timing_wheel(void)
{
    if constexpr (kTestNoTimingWheel || kTestRC) return;
    for (auto flow: active_flows_map_) {
        flow.second->burst_timing_wheel();
    }
}

void UcclRDMAEngine::run() {

    while (!shutdown_) {
        // Calculate the cycles elapsed since last periodic processing.
        auto now_tsc = rdtsc();
        const auto elapsed_tsc = now_tsc - last_periodic_tsc_;

        if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
            // Perform periodic processing.
            periodic_process();
            last_periodic_tsc_ = now_tsc;
        }

        handle_clock_synchronization();

        handle_rx_work();

        handle_tx_work();
        
        handle_timing_wheel();
        
        if constexpr (kTestRC)
            test_rc_handle_completion();
        else
            handle_completion();

    }
    // std::cout << "Engine " << engine_idx_ << " shutdown" << std::endl;
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclRDMAEngine::periodic_process() {
    // Advance the periodic ticks counter.
    periodic_ticks_++;
    if constexpr (!kTestRC)
        handle_rto();
    process_ctl_reqs();
}

void UcclRDMAEngine::handle_rto() {
    for (auto [flow_id, flow] : active_flows_map_) {
        auto is_active_flow = flow->periodic_check();
        DCHECK(is_active_flow);
    }
}

/// TODO: handle error case
void UcclRDMAEngine::process_ctl_reqs() {
    Channel::CtrlMsg ctrl_work;
    if (jring_sc_dequeue_bulk(channel_->ctrl_cmdq_, &ctrl_work, 1, nullptr) ==
        1) {
        switch (ctrl_work.opcode) {
            case Channel::CtrlMsg::kInstallFlowRDMA:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kInstallFlowRDMA";
                    handle_install_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kSyncFlowRDMA:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kSyncFlowRDMA";
                    handle_sync_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kRegMR:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kRegMR";
                    handle_regmr_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kRegMRDMABUF:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kRegMRDMABUF";
                    handle_regmr_dmabuf_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kDeregMR:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kDeregMR";
                    handle_deregmr_on_engine_rdma(ctrl_work);
                break;
            default:
                break;
        }
    }
}

void UcclRDMAEngine::handle_regmr_dmabuf_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    auto *flow = active_flows_map_[ctrl_work.flow_id];
    auto *poll_ctx = ctrl_work.poll_ctx;
    if (flow == nullptr) {
        LOG(ERROR) << "Flow not found";
        return;
    }

    auto *rdma_ctx = flow->rdma_ctx_;
    DCHECK(rdma_ctx != nullptr);

    auto *mr = ibv_reg_dmabuf_mr(rdma_ctx->pd_, 
        ctrl_work.meta.ToEngine.offset, ctrl_work.meta.ToEngine.len, (uint64_t)ctrl_work.meta.ToEngine.addr, ctrl_work.meta.ToEngine.fd, 
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    DCHECK(mr != nullptr);

    VLOG(5) << "Memory region (DMA-BUF) address: "<< (uint64_t)mr->addr << ", lkey: " << mr->lkey << ", rkey: " << mr->rkey << ", size: " << mr->length;

    // Wakeup app thread.
    {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }

    Channel::CtrlMsg ctrl_work_rsp;
    ctrl_work_rsp.meta.ToEndPoint.mr = mr;
    ctrl_work_rsp.opcode = Channel::CtrlMsg::kCompleteRegMR;
    while (jring_sp_enqueue_bulk(channel_->ctrl_rspq_, &ctrl_work_rsp, 1, nullptr) != 1) {}
}

void UcclRDMAEngine::handle_regmr_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    auto *flow = active_flows_map_[ctrl_work.flow_id];
    auto *poll_ctx = ctrl_work.poll_ctx;
    if (flow == nullptr) {
        LOG(ERROR) << "Flow not found";
        return;
    }
    
    auto *rdma_ctx = flow->rdma_ctx_;
    DCHECK(rdma_ctx != nullptr);

    auto *mr = ibv_reg_mr(rdma_ctx->pd_, ctrl_work.meta.ToEngine.addr, ctrl_work.meta.ToEngine.len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    DCHECK(mr != nullptr);

    VLOG(5) << "Memory region address: "<< (uint64_t)mr->addr << ", lkey: " << mr->lkey << ", rkey: " << mr->rkey << ", size: " << mr->length;

    // Wakeup app thread.
    {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }

    Channel::CtrlMsg ctrl_work_rsp;
    ctrl_work_rsp.meta.ToEndPoint.mr = mr;
    ctrl_work_rsp.opcode = Channel::CtrlMsg::kCompleteRegMR;
    while (jring_sp_enqueue_bulk(channel_->ctrl_rspq_, &ctrl_work_rsp, 1, nullptr) != 1) {}
}

void UcclRDMAEngine::handle_deregmr_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    auto *flow = active_flows_map_[ctrl_work.flow_id];
    auto *poll_ctx = ctrl_work.poll_ctx;
    auto *mr = ctrl_work.meta.ToEngine.mr;
    if (flow == nullptr) {
        LOG(ERROR) << "Flow not found";
        return;
    }
    
    auto *rdma_ctx = flow->rdma_ctx_;

    ibv_dereg_mr(mr);

    // Wakeup app thread.
    {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }
}


void UcclRDMAEngine::handle_sync_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    int ret;
    auto meta = ctrl_work.meta;
    auto *poll_ctx = ctrl_work.poll_ctx;
    
    auto *flow = active_flows_map_[ctrl_work.flow_id];
    if (flow == nullptr) {
        LOG(ERROR) << "Flow not found";
        return;
    }

    auto *rdma_ctx = flow->rdma_ctx_;

    auto comm_base = rdma_ctx->is_send_ ? &rdma_ctx->send_comm_.base : &rdma_ctx->recv_comm_.base;
    
    // Copy fields from Endpoint
    if (meta.ToEngine.fifo) {
        comm_base->remote_ctx.fifo_addr = meta.ToEngine.fifo_addr;
        comm_base->remote_ctx.fifo_key = meta.ToEngine.fifo_key;
    }

    VLOG(5) << "Remote FIFO addr " << comm_base->remote_ctx.fifo_addr
              << " key " << comm_base->remote_ctx.fifo_key;

    if (rdma_ctx->ready_entropy_cnt_ < kPortEntropy) {
        // UC QPs.
        auto qp = rdma_ctx->uc_qps_[rdma_ctx->ready_entropy_cnt_].qp;
        rdma_ctx->uc_qps_[rdma_ctx->ready_entropy_cnt_].remote_psn = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(qp, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTR";
        ret = modify_qp_rts(qp, rdma_ctx, rdma_ctx->uc_qps_[rdma_ctx->ready_entropy_cnt_].local_psn, kTestRC);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTS";
        rdma_ctx->ready_entropy_cnt_++;
    } else if (rdma_ctx->ready_entropy_cnt_ == kPortEntropy) {
        // Ctrl QP.
        rdma_ctx->ctrl_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->ctrl_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTR";
        ret = modify_qp_rts(rdma_ctx->ctrl_qp_, rdma_ctx, rdma_ctx->ctrl_local_psn_, false);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTS";
        rdma_ctx->ready_entropy_cnt_++;
    } else if (rdma_ctx->ready_entropy_cnt_ == kPortEntropy + 1) {
        // Fifo QP.
        rdma_ctx->fifo_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->fifo_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify Fifo QP to RTR";
        ret = modify_qp_rts(rdma_ctx->fifo_qp_, rdma_ctx, rdma_ctx->fifo_local_psn_, true);
        DCHECK(ret == 0) << "Failed to modify Fifo QP to RTS";
        rdma_ctx->ready_entropy_cnt_++;
    } else if (rdma_ctx->ready_entropy_cnt_ == kPortEntropy + 2) {
        // Retr QP.
        rdma_ctx->retr_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->retr_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify Retr QP to RTR";
        ret = modify_qp_rts(rdma_ctx->retr_qp_, rdma_ctx, rdma_ctx->retr_local_psn_, false);
        DCHECK(ret == 0) << "Failed to modify Retr QP to RTS";
        rdma_ctx->ready_entropy_cnt_++;
    } 
    else {
        LOG(ERROR) << "Invalid ready_entropy_cnt_ " << rdma_ctx->ready_entropy_cnt_;
    }

    if (rdma_ctx->ready_entropy_cnt_ == RDMAContext::kTotalQP) {
        
        if (!rdma_ctx->is_send_) {
            // GPU flush QP.
            ret = modify_qp_rtr_gpuflush(rdma_ctx->gpu_flush_qp_, rdma_ctx);
            DCHECK(ret == 0) << "Failed to modify GPU flush QP to RTR";
            ret = modify_qp_rts(rdma_ctx->gpu_flush_qp_, rdma_ctx, 0, true);
            DCHECK(ret == 0) << "Failed to modify GPU flush QP to RTS";
        }
        
        // Wakeup app thread.
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }
}

void UcclRDMAEngine::handle_install_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    int ret;
    auto flow_id = ctrl_work.flow_id;
    auto meta = ctrl_work.meta;
    auto *poll_ctx = ctrl_work.poll_ctx;
    auto dev = ctrl_work.meta.ToEngine.dev;

    auto *flow = new UcclFlow(this, channel_, flow_id, RDMAFactory::CreateContext(dev, meta));

    std::tie(std::ignore, ret) = active_flows_map_.insert({flow_id, flow});
    DCHECK(ret);

    // Wakeup app thread.
    {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }

    auto *rdma_ctx = flow->rdma_ctx_;
    
    Channel::CtrlMsg ctrl_work_rsp[RDMAContext::kTotalQP];
    for (int i = 0; i < kPortEntropy; i++) {
        auto qp = rdma_ctx->uc_qps_[i].qp;
        DCHECK(qp != nullptr);
        ctrl_work_rsp[i].meta.ToEndPoint.local_psn = rdma_ctx->uc_qps_[i].local_psn;
        ctrl_work_rsp[i].meta.ToEndPoint.local_qpn = qp->qp_num;
        ctrl_work_rsp[i].opcode = Channel::CtrlMsg::kCompleteFlowRDMA;

        rdma_ctx->qpn2idx_.insert({qp->qp_num, i});
    }

    ctrl_work_rsp[kPortEntropy].meta.ToEndPoint.local_psn = rdma_ctx->ctrl_local_psn_;
    ctrl_work_rsp[kPortEntropy].meta.ToEndPoint.local_qpn = rdma_ctx->ctrl_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy].meta.ToEndPoint.fifo = false;
    ctrl_work_rsp[kPortEntropy].opcode = Channel::CtrlMsg::kCompleteFlowRDMA;

    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.local_psn = rdma_ctx->fifo_local_psn_;
    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.local_qpn = rdma_ctx->fifo_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.fifo = true;
    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.fifo_key = rdma_ctx->fifo_mr_->rkey;
    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.fifo_addr = reinterpret_cast<uint64_t>(rdma_ctx->fifo_mr_->addr);
    ctrl_work_rsp[kPortEntropy + 1].opcode = Channel::CtrlMsg::kCompleteFlowRDMA;

    ctrl_work_rsp[kPortEntropy + 2].meta.ToEndPoint.local_psn = rdma_ctx->retr_local_psn_;
    ctrl_work_rsp[kPortEntropy + 2].meta.ToEndPoint.local_qpn = rdma_ctx->retr_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy + 2].meta.ToEndPoint.fifo = false;
    ctrl_work_rsp[kPortEntropy + 2].opcode = Channel::CtrlMsg::kCompleteFlowRDMA;

    while (jring_sp_enqueue_bulk(channel_->ctrl_rspq_, ctrl_work_rsp, RDMAContext::kTotalQP, nullptr) != RDMAContext::kTotalQP) {
    }

}

RDMAEndpoint::RDMAEndpoint(const uint8_t *gid_idx_list, int num_devices, int num_engines_per_dev, int engine_cpu_start)
    : num_devices_(num_devices), num_engines_per_dev_(num_engines_per_dev), engine_cpu_start_(engine_cpu_start), stats_thread_([this]() { stats_thread_fn(); }) {
    
    // Initialize all RDMA devices.
    static std::once_flag flag_once;
    std::call_once(flag_once, [&]() {
        for (int i = 0; i < num_devices; i++) {
            RDMAFactory::init_dev(gid_idx_list[i]);
            auto *factory_dev = RDMAFactory::get_factory_dev(i);
            // Copy fields from factory device to endpoint device.
            rdma_dev_list_[i].context = factory_dev->context;
            memcpy(rdma_dev_list_[i].ib_name, factory_dev->ib_name, sizeof(factory_dev->ib_name));
            rdma_dev_list_[i].ib_port_num = factory_dev->ib_port_num;
            rdma_dev_list_[i].gid_idx = factory_dev->gid_idx;
            rdma_dev_list_[i].local_ip_str = factory_dev->local_ip_str;
        }
    });

    CHECK_LE(num_engines_per_dev, NUM_CPUS / 4)
        << "num_engines_per_dev should be less than or equal to the number of CPUs / 4";

    int total_num_engines = num_devices * num_engines_per_dev;

    // Create multiple engines. Each engine has its own thread and channel to let the endpoint communicate with.
    for (int i = 0; i < total_num_engines; i++) channel_vec_[i] = new Channel();

    for (int engine_id = 0, engine_cpu_id = engine_cpu_start;
         engine_id < total_num_engines; engine_id++, engine_cpu_id++) {
        
        auto dev = engine_id / num_engines_per_dev;
        
        engine_vec_.emplace_back(std::make_unique<UcclRDMAEngine>(
            dev, engine_id, channel_vec_[engine_id]));
        
        engine_th_vec_.emplace_back(std::make_unique<std::thread>(
            [engine_ptr = engine_vec_.back().get(), engine_id, engine_cpu_id]() {
                VLOG(5) << "[Engine#" << engine_id << "] "
                          << "running on CPU " << engine_cpu_id;
                pin_thread_to_cpu(engine_cpu_id);
                engine_ptr->run();
            }));
    }

    ctx_pool_ = new SharedPool<PollCtx *, true>(kMaxInflightMsg);
    ctx_pool_buf_ = new uint8_t[kMaxInflightMsg * sizeof(PollCtx)];
    for (int i = 0; i < kMaxInflightMsg; i++) {
        ctx_pool_->push(new (ctx_pool_buf_ + i * sizeof(PollCtx)) PollCtx());
    }

    // Create listening sockets
    for (int i = 0; i < num_devices; i++) {
        listen_fds_[i] = socket(AF_INET, SOCK_STREAM, 0);
        DCHECK(listen_fds_[i] >= 0) << "ERROR: opening socket";
        int flag = 1;
        DCHECK(setsockopt(listen_fds_[i], SOL_SOCKET, SO_REUSEADDR, &flag,
                        sizeof(int)) >= 0)
            << "ERROR: setsockopt SO_REUSEADDR fails";
        struct sockaddr_in serv_addr;
        bzero((char *)&serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(kBootstrapPort + i);
        DCHECK(bind(listen_fds_[i], (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >=
            0)
            << "ERROR: binding";

        DCHECK(!listen(listen_fds_[i], 128)) << "ERROR: listen";
        VLOG(5) << "[Endpoint] server ready, listening on port "
                << kBootstrapPort + i;
    }
}

void UcclRDMAEngine::release()
{
    for (auto flow: active_flows_map_) delete flow.second->rdma_ctx_;
    active_flows_map_.clear();
}

RDMAEndpoint::~RDMAEndpoint() {
    for (auto &engine : engine_vec_) engine->shutdown();
    for (auto &engine_th : engine_th_vec_) engine_th->join();
    for (auto &engine : engine_vec_) engine->release();

    for (int i = 0; i < num_devices_ * num_engines_per_dev_; i++) delete channel_vec_[i];

    delete ctx_pool_;
    delete[] ctx_pool_buf_;

    for (int i = 0; i < num_devices_; i++)
        close(listen_fds_[i]);

    {
        std::lock_guard<std::mutex> lock(fd_map_mu_);
        for (auto &[flow_id, boostrap_fd] : fd_map_) {
            close(boostrap_fd);
        }
    }

    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        shutdown_ = true;
        stats_cv_.notify_all();
    }

    stats_thread_.join();
}

ConnID RDMAEndpoint::uccl_connect(int dev, std::string remote_ip) {
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int bootstrap_fd;

    bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(bootstrap_fd >= 0);

    server = gethostbyname(remote_ip.c_str());
    DCHECK(server);

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(kBootstrapPort + dev);

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {};
    localaddr.sin_family = AF_INET;
    auto local_ip_str = rdma_dev_list_[dev].local_ip_str;
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    VLOG(5) << "[Endpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort + dev;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        VLOG(5) << "[Endpoint] connecting... Make sure the server is up.";
    }

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    auto local_engine_idx = find_least_loaded_engine_idx_and_update();
    CHECK_GE(local_engine_idx, 0);

    FlowID flow_id;
    while (true) {
        int ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        VLOG(5) << "[Endpoint] connect: receive proposed FlowID: " << std::hex
                  << "0x" << flow_id;

        // Check if the flow ID is unique, and return it to the server.
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            unique =
                (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique) break;
    }
    
    install_flow_on_engine_rdma(dev, flow_id, local_engine_idx, bootstrap_fd, true);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RDMAEndpoint::uccl_connect(int dev, int engine_id, std::string remote_ip) {
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int bootstrap_fd;

    bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(bootstrap_fd >= 0);

    server = gethostbyname(remote_ip.c_str());
    DCHECK(server);

    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr,
          server->h_length);
    serv_addr.sin_port = htons(kBootstrapPort + dev);

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {};
    localaddr.sin_family = AF_INET;
    auto local_ip_str = rdma_dev_list_[dev].local_ip_str;
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    VLOG(5) << "[Endpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort + dev;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        VLOG(5) << "[Endpoint] connecting... Make sure the server is up.";
    }

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    auto local_engine_idx = engine_id;
    DCHECK(local_engine_idx < num_engines_per_dev_);
    put_load_on_engine(local_engine_idx);
    CHECK_GE(local_engine_idx, 0);

    FlowID flow_id;
    while (true) {
        int ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        VLOG(5) << "[Endpoint] connect: receive proposed FlowID: " << std::hex
                  << "0x" << flow_id;

        // Check if the flow ID is unique, and return it to the server.
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            unique =
                (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique) break;
    }
    
    install_flow_on_engine_rdma(dev, flow_id, local_engine_idx, bootstrap_fd, true);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RDMAEndpoint::uccl_accept(int dev, std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fds_[dev], (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    VLOG(5) << "[Endpoint] accept from " << remote_ip << ":"
              << cli_addr.sin_port;

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    auto local_engine_idx = find_least_loaded_engine_idx_and_update();
    CHECK_GE(local_engine_idx, 0);

    // Generate unique flow ID for both client and server.
    FlowID flow_id;
    while (true) {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<FlowID> distribution(0, std::numeric_limits<FlowID>::max());
        flow_id = distribution(generator);
        // generate flow_id sequentially for better debugging
        flow_id = fff++;
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            unique =
                (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) {
                // Speculatively insert the flow ID.
                fd_map_[flow_id] = bootstrap_fd;
            } else {
                continue;
            }
        }

        VLOG(5) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
                  << flow_id;

        // Ask client if this is unique
        // Let client use flow_id + 50000
        FlowID cid = flow_id + 50000;
        int ret = send_message(bootstrap_fd, &cid, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        bool unique_from_client;
        ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique_from_client) {
            break;
        } else {
            // Remove the speculatively inserted flow ID.
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            DCHECK(1 == fd_map_.erase(flow_id));
        }
    }

    install_flow_on_engine_rdma(dev, flow_id, local_engine_idx, bootstrap_fd, false);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RDMAEndpoint::uccl_accept(int dev, int engine_id, std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fds_[dev], (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    VLOG(5) << "[Endpoint] accept from " << remote_ip << ":"
              << cli_addr.sin_port;

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    auto local_engine_idx = engine_id;
    DCHECK(local_engine_idx < num_engines_per_dev_);
    put_load_on_engine(local_engine_idx);
    CHECK_GE(local_engine_idx, 0);

    // Generate unique flow ID for both client and server.
    FlowID flow_id;
    while (true) {
        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<FlowID> distribution(0, std::numeric_limits<FlowID>::max());
        flow_id = distribution(generator);
        // generate flow_id sequentially for better debugging
        flow_id = fff++;
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            unique =
                (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) {
                // Speculatively insert the flow ID.
                fd_map_[flow_id] = bootstrap_fd;
            } else {
                continue;
            }
        }

        VLOG(5) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
                  << flow_id;

        // Ask client if this is unique
        // Let client use flow_id + 50000
        FlowID cid = flow_id + 50000;
        int ret = send_message(bootstrap_fd, &cid, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        bool unique_from_client;
        ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique_from_client) {
            break;
        } else {
            // Remove the speculatively inserted flow ID.
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            DCHECK(1 == fd_map_.erase(flow_id));
        }
    }

    install_flow_on_engine_rdma(dev, flow_id, local_engine_idx, bootstrap_fd, false);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

PollCtx *RDMAEndpoint::uccl_send_async(ConnID conn_id, struct Mhandle *mhandle, const void *data,
                                   const size_t size, struct ucclRequest *ureq) 
{
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
    };
    
    msg.tx.data = const_cast<void *>(data);
    msg.tx.size = size;
    msg.tx.mr = mhandle->mr;
    msg.ureq = ureq;
    
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->tx_cmdq_,
                                 &msg, 1, nullptr) != 1);

    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_send_async(ConnID conn_id, struct Mhandle *mhandle, const void *data,
                                   const size_t size) 
{
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
    };
    
    msg.tx.data = const_cast<void *>(data);
    msg.tx.size = size;
    msg.tx.mr = mhandle->mr;
    msg.ureq = nullptr;
    
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->tx_cmdq_,
                                 &msg, 1, nullptr) != 1);

    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_flush(ConnID conn_id, struct Mhandle **mhandles, void **data, int *size, int n)
{
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kFlush,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,  
    };

    for (int i = 0; i < n; i++) {
        msg.flush.data[i] = data[i];
        msg.flush.size[i] = size[i];
        msg.flush.mr[i] = mhandles[i]->mr;
    }
    msg.flush.n = n;

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->rx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_recv_async(ConnID conn_id, struct Mhandle **mhandles, void **data, int *size, int n, struct ucclRequest *ureq) 
{
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
    };

    for (int i = 0; i < n; i++) {
        msg.rx.data[i] = data[i];
        msg.rx.size[i] = size[i];
        msg.rx.mr[i] = mhandles[i]->mr;
    }
    msg.rx.n = n;
    msg.ureq = ureq;

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->rx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_recv_async(ConnID conn_id, struct Mhandle **mhandles, void **data, int *size, int n) 
{
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
    };

    for (int i = 0; i < n; i++) {
        msg.rx.data[i] = data[i];
        msg.rx.size[i] = size[i];
        msg.rx.mr[i] = mhandles[i]->mr;
    }
    msg.rx.n = n;
    msg.ureq = nullptr;

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->rx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

bool RDMAEndpoint::uccl_wait(PollCtx *ctx) {
    {
        std::unique_lock<std::mutex> lock(ctx->mu);
        ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
    }
    fence_and_clean_ctx(ctx);
    return true;
}

bool RDMAEndpoint::uccl_poll(PollCtx *ctx) {
    while (!uccl_poll_once(ctx));
    return true;
}

bool RDMAEndpoint::uccl_poll_once(PollCtx *ctx) {
    if (!ctx->done.load()) return false;
    fence_and_clean_ctx(ctx);
    return true;
}

void RDMAEndpoint::install_flow_on_engine_rdma(int dev, FlowID flow_id,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd, bool is_send) {
    int ret;
    struct XchgMeta meta = { 0 };
    // We use this pointer to fill meta data.
    auto *to_engine_meta = &meta.ToEngine;
    struct RDMAExchangeFormatRemote xchg_meta[RDMAContext::kTotalQP];

    DCHECK(dev < num_devices_);

    auto factory_dev = RDMAFactory::get_factory_dev(dev);

    auto *context = factory_dev->context;

    // Sync GID with remote peer.
    char buf[16];
    memcpy(buf, &factory_dev->gid.raw, 16);
    ret = send_message(bootstrap_fd, buf, 16);
    DCHECK(ret == 16);
    ret = receive_message(bootstrap_fd, &buf, 16);
    DCHECK(ret == 16);
    memcpy(&to_engine_meta->remote_gid.raw, buf, 16);

    // Sync PortAttr with remote peer.
    ret = send_message(bootstrap_fd, &factory_dev->port_attr, sizeof(ibv_port_attr));
    DCHECK(ret == sizeof(ibv_port_attr));
    struct ibv_port_attr remote_port_attr;
    ret = receive_message(bootstrap_fd, &remote_port_attr, sizeof(ibv_port_attr));
    DCHECK(ret == sizeof(ibv_port_attr));
    to_engine_meta->remote_port_attr = remote_port_attr;
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "[Endpoint] Local GID:\t";
        for (int i = 0; i < 16; ++i) {
            oss << ((i == 0)? "" : ":") << static_cast<int>(factory_dev->gid.raw[i]);
        }
        VLOG(6) << oss.str();
    }
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "[Endpoint] Remote GID:\t";
        for (int i = 0; i < 16; ++i) {
            oss << ((i == 0)? "" : ":") << static_cast<int>(to_engine_meta->remote_gid.raw[i]);
        }
        VLOG(6) << oss.str();
    }

    VLOG(5) << "[Endpoint] Sync GID done";

    // Which mtu to use?
    to_engine_meta->mtu = factory_dev->port_attr.active_mtu;
    // Which dev to use?
    to_engine_meta->dev = dev;
    // Flow direction?
    to_engine_meta->is_send = is_send;

    // Install RDMA flow on engine.
    auto *poll_ctx = new PollCtx();
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kInstallFlowRDMA,
        .flow_id = flow_id,
        .meta = meta,
        .poll_ctx = poll_ctx,
    };
    while (jring_mp_enqueue_bulk(channel_vec_[local_engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1)
        ;

    // Wait until the flow has been installed on the engine.
    // This also serves as a barrier to ensure that only this flow 
    // is being installed by the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    VLOG(5) << "[Endpoint] Install flow done" << std::endl;

    // Receive local QPN,PSN and FifoAddr from engine.
    int qidx = 0;
    while (qidx < RDMAContext::kTotalQP) {
        Channel::CtrlMsg rsp_msg;
        while (jring_mc_dequeue_bulk(channel_vec_[local_engine_idx]->ctrl_rspq_, &rsp_msg, 1, nullptr) != 1);
        if (rsp_msg.opcode != Channel::CtrlMsg::Op::kCompleteFlowRDMA) continue;
        xchg_meta[qidx].qpn = rsp_msg.meta.ToEndPoint.local_qpn;
        xchg_meta[qidx].psn = rsp_msg.meta.ToEndPoint.local_psn;
        if (rsp_msg.meta.ToEndPoint.fifo) {
            xchg_meta[qidx].fifo_key = rsp_msg.meta.ToEndPoint.fifo_key;
            xchg_meta[qidx].fifo_addr = rsp_msg.meta.ToEndPoint.fifo_addr;
        }
        qidx++;
    }

    // Sync QPN, PSN and FifoAddr with remote peer.
    for (int i = 0; i < RDMAContext::kTotalQP; i++) {
        ret = send_message(bootstrap_fd, &xchg_meta[i], sizeof(struct RDMAExchangeFormatRemote));
        DCHECK(ret == sizeof(struct RDMAExchangeFormatRemote));
    }
    for (int i = 0; i < RDMAContext::kTotalQP; i++) {
        ret = receive_message(bootstrap_fd, &xchg_meta[i], sizeof(struct RDMAExchangeFormatRemote));
        DCHECK(ret == sizeof(struct RDMAExchangeFormatRemote));
    }

    VLOG(5) << "[Endpoint] Sync QPN and PSN done" << std::endl;
    
    // Send remote QPN, PSN and FifoAddr to engine.
    poll_ctx = new PollCtx();
    for (int i = 0; i < RDMAContext::kTotalQP; i++) {
        meta.ToEngine.remote_qpn = xchg_meta[i].qpn;
        meta.ToEngine.remote_psn = xchg_meta[i].psn;
        if (i == RDMAContext::kFifoIndex) {
            meta.ToEngine.fifo = true;
            meta.ToEngine.fifo_key = xchg_meta[i].fifo_key;
            meta.ToEngine.fifo_addr = xchg_meta[i].fifo_addr;
        }
        Channel::CtrlMsg ctrl_msg = {
            .opcode = Channel::CtrlMsg::Op::kSyncFlowRDMA,
            .flow_id = flow_id,
            .meta = meta,
            .poll_ctx = poll_ctx,
        };
        while (jring_mp_enqueue_bulk(channel_vec_[local_engine_idx]->ctrl_cmdq_,
                                    &ctrl_msg, 1, nullptr) != 1)
        ;
    }

    // Wait until the information has been received on the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    VLOG(5) << "[Endpoint] Sync flow done" << std::endl;

    // sync so to receive flow_id packets.
    net_barrier(bootstrap_fd);
}

inline void RDMAEndpoint::put_load_on_engine(int engine_id)
{
    engine_load_vec_[engine_id]++;
}

inline int RDMAEndpoint::find_least_loaded_engine_idx_and_update() {
    std::lock_guard<std::mutex> lock(engine_load_vec_mu_);
    if (engine_load_vec_.empty()) return -1;  // Handle empty vector case

    auto minElementIter =
        std::min_element(engine_load_vec_.begin(), engine_load_vec_.end());
    *minElementIter += 1;
    return std::distance(engine_load_vec_.begin(), minElementIter);
}

inline void RDMAEndpoint::fence_and_clean_ctx(PollCtx *ctx) {
    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx->clear();
    ctx_pool_->push(ctx);
}

void RDMAEndpoint::stats_thread_fn() {
    if (GetEnvVar("UCCL_ENGINE_QUIET") == "1") return;

    while (!shutdown_) {
        {
            std::unique_lock<std::mutex> lock(stats_mu_);
            bool shutdown = stats_cv_.wait_for(
                lock, std::chrono::seconds(kStatsTimerIntervalSec),
                [this] { return shutdown_.load(); });
            if (shutdown) break;
        }

        if (engine_vec_.empty()) continue;
        std::string s;
        s += "\n\t[Uccl Engine] ";
        for (auto &engine : engine_vec_) {
            s += engine->status_to_string();
        }
        VLOG(4) << s;
    }
}

int RDMAEndpoint::uccl_regmr_dmabuf(ConnID conn_id, void *addr, size_t len, int type, int offset, int fd, struct Mhandle **mhandle)
{
    auto *poll_ctx = new PollCtx();

    struct XchgMeta meta = {};

    meta.ToEngine.addr = addr;
    meta.ToEngine.len = len;
    meta.ToEngine.type = type;
    meta.ToEngine.offset = offset;
    meta.ToEngine.fd = fd;
    
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kRegMRDMABUF,
        .flow_id = conn_id.flow_id,
        .meta = meta,
        .poll_ctx = poll_ctx
    };

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1)
        ;
    // Wait until the flow has been installed on the engine.
    // This also serves as a barrier to ensure that only this flow 
    // is being installed by the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    Channel::CtrlMsg rsp_msg;
    while (1) {
        if (jring_mc_dequeue_bulk(channel_vec_[conn_id.engine_idx]->ctrl_rspq_, &rsp_msg, 1, nullptr) == 1) {
            DCHECK(rsp_msg.opcode == Channel::CtrlMsg::Op::kCompleteRegMR);
            *mhandle = new Mhandle();
            (*mhandle)->mr = rsp_msg.meta.ToEndPoint.mr;
            break;
        }
    }

    return 0;
}

int RDMAEndpoint::uccl_regmr(ConnID conn_id, void *addr, size_t len, int type /*unsed for now*/, struct Mhandle **mhandle)
{
    auto *poll_ctx = new PollCtx();

    struct XchgMeta meta = {};

    meta.ToEngine.addr = addr;
    meta.ToEngine.len = len;
    meta.ToEngine.type = type;
    
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kRegMR,
        .flow_id = conn_id.flow_id,
        .meta = meta,
        .poll_ctx = poll_ctx
    };

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1)
        ;
    // Wait until the flow has been installed on the engine.
    // This also serves as a barrier to ensure that only this flow 
    // is being installed by the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    Channel::CtrlMsg rsp_msg;
    while (1) {
        if (jring_mc_dequeue_bulk(channel_vec_[conn_id.engine_idx]->ctrl_rspq_, &rsp_msg, 1, nullptr) == 1) {
            DCHECK(rsp_msg.opcode == Channel::CtrlMsg::Op::kCompleteRegMR);
            *mhandle = new Mhandle();
            (*mhandle)->mr = rsp_msg.meta.ToEndPoint.mr;
            break;
        }
    }

    return 0;
}

void RDMAEndpoint::uccl_deregmr(ConnID conn_id, struct Mhandle *mhandle)
{
    auto *poll_ctx = new PollCtx();

    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kDeregMR,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx
    };

    ctrl_msg.meta.ToEngine.mr = mhandle->mr;

    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1)
        ;
    // Wait until the information has been received on the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;
}

std::string UcclFlow::to_string()
{
    std::string s; s.clear();

    for (int qpidx = 0; qpidx < kPortEntropy; qpidx++) {
        auto qpw = rdma_ctx_->uc_qps_[qpidx];
        s += "\n\t[UC] QP#" + std::to_string(qpidx);
        s += qpw.pcb.to_string();
    }

    return s;
}

std::string UcclRDMAEngine::status_to_string()
{
    std::string s; s.clear();
    for (auto [flow_id, flow] : active_flows_map_) {
        s += "\nEngine " + std::to_string(engine_idx_) + "Flow 0x%" + std::to_string(flow_id);
        s += flow->to_string();
    }
    return s;
}

}  // namespace uccl
