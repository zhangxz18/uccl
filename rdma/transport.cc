#include "transport.h"
#include "transport_cc.h"
#include "transport_config.h"
#include "util_rdma.h"
#include "util_timer.h"
#include "util_list.h"
#include <cstdlib>
#include <endian.h>
#include <infiniband/verbs.h>
#include <utility>

namespace uccl {

void UcclFlow::rx_messages() {

}

void UcclFlow::post_fifo(struct FlowRequest *req, void **data, size_t *size, int n)
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
        elems[i].rkey = rdma_ctx_->data_mr_->rkey;
        elems[i].nmsgs = n;
        // For sender to check if the receiver is ready.
        elems[i].idx = rem_fifo->fifo_tail + 1;
        elems[i].size = size[i];
        // For sender to encode the request id in the immediate data.
        elems[i].rid = rdma_ctx_->get_request_id(req, &recv_comm_->base);

        LOG(INFO) << "Post Recv: addr: " << elems[i].addr << ", rkey: " << elems[i].rkey << ", size: " << elems[i].size;
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
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr_id = rdma_ctx_->get_request_id(req, &recv_comm_->base);
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

void UcclFlow::app_supply_rx_buf(Channel::Msg &rx_work) {
    auto data = rx_work.rx.data;
    auto size = rx_work.rx.size;
    auto n = rx_work.rx.n;
    auto poll_ctx = rx_work.poll_ctx;

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

    // Push buffer information to FIFO queue and notify the remote peer.
    post_fifo(req, data, size, n);
}

void UcclFlow::rdma_single_send(struct FlowRequest *req, struct FifoItem &slot, uint32_t mid)
{
    auto *sent_offset = &req->send.sent_offset;
    auto *size = &req->send.size;
    auto data = req->send.data;
    auto lkey = req->send.lkey;
    auto rkey = slot.rkey;
    auto remote_addr = slot.addr;
    uint64_t sge_addr;

    while (*size) {
        auto qpidx = rdma_ctx_->select_qpidx();
        auto qpw = &rdma_ctx_->uc_qps_[qpidx];

        // Prepare SGE.
        DCHECK(rdma_ctx_->sge_ex_pool_.alloc_sge(&sge_addr) == 0);
        struct sge_ex *sge_ex = reinterpret_cast<struct sge_ex *>(sge_addr);
        sge_ex->sge.addr = (uintptr_t)data;
        sge_ex->sge.lkey = lkey;
        sge_ex->sge.length = std::min(*size, (int)kChunkSize);
        auto chunk_size = sge_ex->sge.length;
        *size -= chunk_size;
        data = static_cast<char*>(data) + sge_ex->sge.length;

        sge_ex->wr_remote_addr = remote_addr + *sent_offset;
        sge_ex->wr_rkey = rkey;

        // There is no need to signal every WQE since we don't handle TX completions.
        // But we still need occasionally post a request with the IBV_SEND_SIGNALED flag.
        if (signal_cnt_++ % kSignalInterval == 0)
            sge_ex->wr_send_flags = IBV_SEND_SIGNALED;

        IMMData imm_data(0);

        imm_data.SetHint(0);
        imm_data.SetCSN(qpw->pcb.get_snd_nxt().to_uint32());
        imm_data.SetRID(slot.rid);
        imm_data.SetMID(mid);

        sge_ex->wr_imm_data = htonl(imm_data.GetImmData());

        sge_ex->timely = &qpw->pcb.timely;
        sge_ex->req = req;
        sge_ex->csn = qpw->pcb.seqno().to_uint32() - 1;
        sge_ex->qpidx = qpidx;
        sge_ex->last_chunk = *size == 0;

        // Queue the SGE on the timing wheel.
        rdma_ctx_->wheel_.queue_on_timing_wheel(qpw->pcb.timely.rate_, rdtsc(), sge_ex, chunk_size);

        LOG(INFO) << "Sending: csn: " << qpw->pcb.seqno().to_uint32() - 1 << ", rid: " << slot.rid << ", mid: " << mid << "with QP#" << qpw->qp->qp_num;
        LOG(INFO) << "Queue " << chunk_size << " bytes";
        
        *sent_offset += chunk_size;
    }

}

void UcclFlow::rdma_multi_send(int slot)
{
    auto send_comm_ = &rdma_ctx_->send_comm_;
    auto reqs = send_comm_->fifo_reqs[slot];
    auto rem_fifo = send_comm_->base.fifo;
    auto slots = rem_fifo->elems[slot];
    auto nmsgs = slots[0].nmsgs;

    for (int i = 0; i < nmsgs; i++) rdma_single_send(reqs[i], slots[i], i);
}

bool UcclFlow::handle_ctrl_cq_wc(void)
{
    auto cq_ex = rdma_ctx_->ctrl_cq_ex_;
    auto pkt_addr = cq_ex->wr_id;
    auto wc_opcode = ibv_wc_read_opcode(cq_ex);

    bool ret = false;
    
    if (wc_opcode == IBV_WC_SEND) {
        // Sending ACK is done.
    } else if (wc_opcode == IBV_WC_RECV && cq_ex->status == IBV_WC_SUCCESS) {
        // Receiving an ACK.
        auto t6 = rdtsc();
        auto qpidx = ibv_wc_read_imm_data(cq_ex);
        auto qpw = &rdma_ctx_->uc_qps_[qpidx];
        ret = true;
        auto *ucclh = reinterpret_cast<UcclPktHdr *>(pkt_addr);
        auto *ucclsackh = reinterpret_cast<UcclSackHdr *>(pkt_addr + kUcclHdrLen);

        auto seqno = ucclh->seqno.value();
        auto ackno = ucclh->ackno.value();

        if (swift::UINT_20::uint20_seqno_lt(ackno, qpw->pcb.snd_una)) {
            LOG(INFO) << "Received old ACK " << ackno << " from QP#" << rdma_ctx_->uc_qps_[qpidx].qp->qp_num << " by Ctrl QP";
        } else if (swift::UINT_20::uint20_seqno_eq(ackno, qpw->pcb.snd_una)) {
            LOG(INFO) << "Received duplicate ACK " << ackno << " from QP#" << rdma_ctx_->uc_qps_[qpidx].qp->qp_num << " by Ctrl QP";
        } else if (swift::UINT_20::uint20_seqno_gt(ackno, qpw->pcb.snd_nxt)) {
            LOG(INFO) << "Received ACK for untransmitted data " << "ackno: " << ackno << ", snd_nxt: " << qpw->pcb.snd_nxt.to_uint32() << " from QP#" << rdma_ctx_->uc_qps_[qpidx].qp->qp_num << " by Ctrl QP";
        } else {
            LOG(INFO) << "Received valid ACK " << ackno << " from QP#" << rdma_ctx_->uc_qps_[qpidx].qp->qp_num << " by Ctrl QP";
            
            size_t num_acked_chunks = ackno - qpw->pcb.snd_una.to_uint32();

            auto t1 = qpw->txtracking.ack_chunks(num_acked_chunks);
            auto remote_queueing_tsc = us_to_cycles(be64toh(ucclsackh->remote_queueing.value()), freq_ghz);
            auto t5 = engine_->convert_nic_to_host(t6, ibv_wc_read_completion_ts(cq_ex));

            /// TODO: Congestion control
            auto endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
            auto fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;

            LOG(INFO) << "Total: " << to_usec(t6 - t1, freq_ghz) << 
                ", Endpoint delay: " << to_usec(endpoint_delay_tsc, freq_ghz) << 
                ", Fabric delay: " << to_usec(fabric_delay_tsc, freq_ghz);
            
            qpw->pcb.update_rate(rdtsc(), fabric_delay_tsc);

            LOG(INFO) << "CC rate: " << qpw->pcb.timely.get_rate_gbps() << " Gbps";

            qpw->pcb.snd_una = ackno;
            qpw->pcb.duplicate_acks = 0;
            qpw->pcb.snd_ooo_acks = 0;
            qpw->pcb.rto_rexmits_consectutive = 0;
            qpw->pcb.rto_maybe_reset();
        }
    }
    rdma_ctx_->ctrl_pkt_pool_.free_buff(pkt_addr);

    return ret;
}

void UcclFlow::complete_ctrl_cq(void) 
{
    while (1) {
        struct ibv_wc wcs[kMaxBatchCQ];
        int nb_post_recv = 0;

        struct ibv_poll_cq_attr poll_cq_attr = {0};
        auto cq_ex = rdma_ctx_->ctrl_cq_ex_;
        if (ibv_start_poll(cq_ex, &poll_cq_attr)) return;

        int cq_budget = 0;

        while (1) {
            nb_post_recv += handle_ctrl_cq_wc() ? 1 : 0;
            if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
        }

        ibv_end_poll(cq_ex);

        // Populate recv work requests for consuming control packets.
        struct ibv_recv_wr *bad_wr;
        for (int i = 0; i < nb_post_recv; i++) {
            uint64_t pkt_addr;
            if (rdma_ctx_->ctrl_pkt_pool_.alloc_buff(&pkt_addr)) {
                LOG(ERROR) << "Failed to allocate control packet buffer";
                return;
            }
            rx_ack_sges_[i].addr = pkt_addr;
            rx_ack_sges_[i].lkey = rdma_ctx_->ctrl_pkt_pool_.get_lkey();
            rx_ack_sges_[i].length = CtrlPktBuffPool::kPktSize;
            rx_ack_wrs_[i].wr_id = pkt_addr;
            rx_ack_wrs_[i].sg_list = &rx_ack_sges_[i];
            rx_ack_wrs_[i].num_sge = 1;
        }
        rx_ack_wrs_[nb_post_recv - 1].next = nullptr;
        if (nb_post_recv) {
            DCHECK(ibv_post_recv(rdma_ctx_->ctrl_qp_, &rx_ack_wrs_[0], &bad_wr) == 0);
            // Restore
            rx_ack_wrs_[nb_post_recv - 1].next = nb_post_recv == kMaxBatchCQ ? nullptr : &rx_ack_wrs_[nb_post_recv];
            LOG(INFO) << "Posted " << nb_post_recv << " recv requests for Ctrl QP";
        }
    }

}

void UcclFlow::flush_timing_wheel(void)
{
    auto wheel = &rdma_ctx_->wheel_;
    struct sge_ex *exs[32];
    struct ibv_send_wr wr, *bad_wr;

    auto permitted_chunks = wheel->get_num_ready_tx_chunk(32, exs);

    if (!permitted_chunks) return;

    memset(&wr, 0, sizeof(wr));
    
    for (auto i = 0; i < permitted_chunks; i++) {
        auto sge_ex = exs[i];
        auto qpidx = sge_ex->qpidx;
        auto qpw = &rdma_ctx_->uc_qps_[qpidx];
        auto req = sge_ex->req;
        
        wr.sg_list = &sge_ex->sge;
        wr.num_sge = 1;
        wr.next = nullptr;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.wr.rdma.remote_addr = sge_ex->wr_remote_addr;
        // Use last sge_ex's rkey, imm_data.
        wr.wr.rdma.rkey = sge_ex->wr_rkey;
        wr.imm_data = sge_ex->wr_imm_data;
        wr.send_flags = sge_ex->wr_send_flags;

        // Track this merged chunk.
        qpw->txtracking.track_chunk(req, be32toh(wr.imm_data), 
            reinterpret_cast<void*>(wr.sg_list->addr), sge_ex->sge.length, 
                sge_ex->last_chunk, rdtsc());
        
        DCHECK(ibv_post_send(qpw->qp, &wr, &bad_wr) == 0);

        rdma_ctx_->sge_ex_pool_.free_sge(reinterpret_cast<uint64_t>(sge_ex));
    }
}

void UcclFlow::try_update_csn(struct UCQPWrapper *qpw)
{
    while (!ready_csn_.empty() && static_cast<uint32_t>(*ready_csn_.begin()) == qpw->pcb.rcv_nxt.to_uint32()) {
        auto csn = *ready_csn_.begin();
        ready_csn_.erase(ready_csn_.begin());
        
        // Data is already DMAed to the application buffer.
        // Nothing more to do.

        qpw->pcb.advance_rcv_nxt();
        LOG(INFO) << "try_update_csn:" << " rcv_nxt: " << qpw->pcb.rcv_nxt.to_uint32();
        prev_csn_ = csn;
        qpw->pcb.sack_bitmap_shift_left_one();
    }
}

void UcclFlow::uc_tx_complete(void)
{
    // Do nothing here.
}

void UcclFlow::uc_receive_data(struct list_head *ack_list, int *post_recv_qidx_list, int *num_post_recv)
{
    auto cq_ex = rdma_ctx_->cq_ex_;
    
    DCHECK(rdma_ctx_->is_send_ == false);
    
    auto recv_comm = &rdma_ctx_->recv_comm_;
    auto byte_len = ibv_wc_read_byte_len(cq_ex);
    auto imm_data = IMMData(ntohl(ibv_wc_read_imm_data(cq_ex)));
    auto qp_num = ibv_wc_read_qp_num(cq_ex);
    auto qpidx = rdma_ctx_->qpn2idx_[qp_num];
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];

    auto hint = imm_data.GetHint();
    auto csn = imm_data.GetCSN();
    auto rid = imm_data.GetRID();
    auto mid = imm_data.GetMID();

    LOG(INFO) << "Received chunk: (byte_len, csn, rid, mid): " << byte_len << ", " << csn << ", " << rid << ", " << mid;

    // Compare CSN with the expected CSN.
    auto ecsn = qpw->pcb.rcv_nxt.to_uint32();

    // It's impossible to receive a chunk with a CSN less than the expected CSN.
    // For CSN that lags behind, RNIC has already handled it.
    DCHECK(!swift::seqno_lt(csn, ecsn));

    auto distance = csn - ecsn;

    if (distance > kReassemblyMaxSeqnoDistance) {
        LOG(INFO) << "Packet too far ahead. Dropping as we can't handle SACK. "
                    << "csn: " << csn << ", ecsn: " << ecsn;
        return;
    }

    ready_csn_.insert(csn);

    // Always use the latest timestamp.
    qpw->pcb.t_remote_nic_rx = ibv_wc_read_completion_ts(cq_ex);

    qpw->pcb.sack_bitmap_bit_set(distance);

    // Locate request by rid
    auto req = rdma_ctx_->get_request_by_id(rid, &recv_comm->base);
    auto msg_size = req->recv.elems[mid].size;
    uint32_t *received_bytes = req->recv.received_bytes;

    received_bytes[mid] += byte_len;
    if (msg_size == received_bytes[mid]) req->recv.fin_msg++;
    if (req->recv.fin_msg == req->nmsgs) { // This request (may contain multiple messages) is complete.
        LOG(INFO) << "Request complete (" << req->nmsgs << " messages)";
        auto poll_ctx = req->poll_ctx;
        // Wakeup app thread waiting one endpoint.
        {
            std::lock_guard<std::mutex> lock(poll_ctx->mu);
            poll_ctx->done = true;
            poll_ctx->cv.notify_one();
        }
        // Free the request.
        rdma_ctx_->free_request(req);
    }

    try_update_csn(qpw);

    if (distance)
        qpw->rxtracking.encounter_ooo();
    
    qpw->rxtracking.cumulate_wqe();
    qpw->rxtracking.cumulate_bytes(byte_len);

    if (list_empty(&qpw->ack.ack_link))
        list_add_tail(&qpw->ack.ack_link, ack_list);

    // Send ACK if needed.
    if (qpw->rxtracking.need_imm_ack()) {
        send_ack(qpidx, 0);
        flush_acks(1);

        qpw->rxtracking.clear_imm_ack();
        list_del(&qpw->ack.ack_link);
    }

    if (qpw->rxtracking.need_fill() == 0) {
        post_recv_qidx_list[(*num_post_recv)++] = qpidx;
    }
}

void UcclFlow::flush_acks(int size)
{
    if (size == 0) return;
    struct ibv_send_wr *bad_wr;
    tx_ack_wrs_[size - 1].next = nullptr;
    DCHECK(ibv_post_send(rdma_ctx_->ctrl_qp_, &tx_ack_wrs_[0], &bad_wr) == 0);
    // Restore
    tx_ack_wrs_[size - 1].next = size == kMaxBatchCQ ? nullptr : &tx_ack_wrs_[size];
}

void UcclFlow::send_ack(int qpidx, int wr_idx)
{
    auto qpw = &rdma_ctx_->uc_qps_[qpidx];
    uint64_t pkt_addr;
    if (rdma_ctx_->ctrl_pkt_pool_.alloc_buff(&pkt_addr)) {
        LOG(ERROR) << "Failed to allocate control packet buffer";
        return;
    }
    const size_t kControlPayloadBytes = kUcclHdrLen + kUcclSackHdrLen;
    auto *ucclh = reinterpret_cast<UcclPktHdr* >(pkt_addr);
    ucclh->magic = be16_t(UcclPktHdr::kMagic);
    ucclh->net_flags = UcclPktHdr::UcclFlags::kAck;
    ucclh->frame_len = be16_t(kControlPayloadBytes);
    ucclh->seqno = be32_t(qpw->pcb.seqno().to_uint32());
    ucclh->ackno = be32_t(qpw->pcb.ackno().to_uint32());
    ucclh->flow_id= be64_t(flow_id_);

    auto *ucclsackh = reinterpret_cast<UcclSackHdr *>(pkt_addr + kUcclHdrLen);

    auto t4 = rdtsc();
    auto t2 = engine_->convert_nic_to_host(t4, rdma_ctx_->uc_qps_[qpidx].pcb.t_remote_nic_rx);

    ucclsackh->remote_queueing = be64_t(to_usec(t4 - t2, freq_ghz));

    for (size_t i = 0; i < sizeof(UcclSackHdr::sack_bitmap) /
                               sizeof(UcclSackHdr::sack_bitmap[0]);
         ++i) {
        ucclsackh->sack_bitmap[i] = be64_t(qpw->pcb.sack_bitmap[i]);
    }
    ucclsackh->sack_bitmap_count = be16_t(qpw->pcb.sack_bitmap_count);

    tx_ack_sges_[wr_idx].addr = pkt_addr;
    tx_ack_sges_[wr_idx].lkey = rdma_ctx_->ctrl_pkt_pool_.get_lkey();
    tx_ack_sges_[wr_idx].length = kControlPayloadBytes;

    // We use wr_id to store the packet address for future freeing.
    tx_ack_wrs_[wr_idx].wr_id = pkt_addr;
    tx_ack_wrs_[wr_idx].sg_list = &tx_ack_sges_[wr_idx];
    tx_ack_wrs_[wr_idx].num_sge = 1;
    tx_ack_wrs_[wr_idx].opcode = IBV_WR_SEND_WITH_IMM;
    tx_ack_wrs_[wr_idx].imm_data = qpidx;
    tx_ack_wrs_[wr_idx].send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

    LOG(INFO) << "send_ack: seqno: " << qpw->pcb.seqno().to_uint32() << ", ackno: " << qpw->pcb.ackno().to_uint32()  << " to QP#" << qpw->qp->qp_num;
}

void UcclFlow::complete_uc_cq(void)
{
    auto cq_ex = rdma_ctx_->cq_ex_;
    struct ibv_wc wcs[kMaxBatchCQ];
    int post_recv_qidx_list[kMaxBatchCQ];
    LIST_HEAD(ack_list);
    int cq_budget = 0;
    int num_post_recv = 0;

    struct ibv_poll_cq_attr poll_cq_attr = {0};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return;
    
    while (1) {
        DCHECK(cq_ex->status == IBV_WC_SUCCESS);
        if (ibv_wc_read_opcode(cq_ex) == IBV_WC_RECV_RDMA_WITH_IMM) {
            // Receive data.
            uc_receive_data(&ack_list, post_recv_qidx_list, &num_post_recv);
        } else {
            DCHECK(ibv_wc_read_opcode(cq_ex) == IBV_WC_RDMA_WRITE);
            // Handle TX completion.
            uc_tx_complete();
        }
        if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
    }
    ibv_end_poll(cq_ex);
    
    // Send coalescing ACKs.
    {
        int wr_idx = 0;
        struct list_head *pos, *n;
        list_for_each_safe(pos, n, &ack_list) {
            auto ack_item = list_entry(pos, struct ack_item, ack_link);
            auto qpidx = ack_item->qpidx;
            send_ack(qpidx, wr_idx++);
            list_del(pos);
        }
        flush_acks(wr_idx);
    }

    // Populate recv work requests for consuming immediate data.
    for (auto i = 0; i < num_post_recv; i++) {
        auto idx = post_recv_qidx_list[i];
        auto qpw = &rdma_ctx_->uc_qps_[idx];
        imm_wrs_[qpw->rxtracking.fill_count() - 1].next = nullptr;
        auto qp = qpw->qp;
        struct ibv_recv_wr *bad_wr;
        DCHECK(ibv_post_recv(qp, &imm_wrs_[0], &bad_wr) == 0);
        LOG(INFO) << "Posted " << qpw->rxtracking.fill_count() << " recv requests for UC QP#" << qp->qp_num;
        
        // Restore
        {
            imm_wrs_[qpw->rxtracking.fill_count() - 1].next = (qpw->rxtracking.fill_count() == kMaxBatchCQ) ? nullptr : &imm_wrs_[qpw->rxtracking.fill_count()];
            qpw->rxtracking.clear_fill();
        }
    }
}

bool UcclFlow::complete_fifo_cq(void)
{
    FlowRequest *req = nullptr;
    auto cq = rdma_ctx_->fifo_cq_;
    struct ibv_wc wc;
    int nb_cqe = ibv_poll_cq(cq, 1, &wc);
    if (nb_cqe <= 0) return false;
    if (wc.status != IBV_WC_SUCCESS) {
        LOG(ERROR) << "Error in FIFO CQ completion:" << wc.status;
        return false;
    }
    
    rdma_ctx_->fifo_cq_polling_ = false;
    return true;
}

bool UcclFlow::tx_messages(Channel::Msg &tx_work) {
    auto data = tx_work.tx.data;
    auto size = tx_work.tx.size;
    auto poll_ctx = tx_work.poll_ctx;
    auto tx_ready_poll_ctx = tx_work.tx_ready_poll_ctx;

    auto send_comm_ = &rdma_ctx_->send_comm_;

    int slot = send_comm_->fifo_head % kMaxReq;

    auto reqs = send_comm_->fifo_reqs[slot];

    auto rem_fifo = send_comm_->base.fifo;

    volatile struct FifoItem *slots = rem_fifo->elems[slot];

    auto idx = send_comm_->fifo_head + 1;
    if (slots[0].idx != idx) {
        return true;
    }
    
    // Wait until all slots are ready
    auto nmsgs = slots[0].nmsgs;
    for (int i = 1; i < nmsgs; i++) while(slots[i].idx != idx) {}

    LOG(INFO) << "Receiver is ready to receive";

    // Wakeup the application thread waiting for the receiver to be ready.
    {
        std::lock_guard<std::mutex> lock(tx_ready_poll_ctx->mu);
        tx_ready_poll_ctx->done = true;
        tx_ready_poll_ctx->cv.notify_one();
    }

    __sync_synchronize();

    for (int i = 0; i < nmsgs; i++) {
        if (reqs[i] != nullptr) continue;
        if (slots[i].size < 0 || slots[i].addr == 0 || slots[i].rkey == 0) {
            LOG(ERROR) << "Receiver posted incorrect receive info";
            return false;
        }
        // Can't send more than what the receiver can receive.
        if (size > slots[i].size) size = slots[i].size;
        
        struct FlowRequest *req = rdma_ctx_->get_request(&send_comm_->base);
        if (!req) {
            LOG(ERROR) << "Failed to get request";
            return false;
        }

        req->type = FlowRequest::SEND;
        req->nmsgs = nmsgs;
        req->poll_ctx = poll_ctx;
        req->send.size = size;
        req->send.sent_offset = 0;
        req->send.data = data;
        req->send.lkey = rdma_ctx_->data_mr_->lkey;

        // Track this request.
        reqs[i] = req;

        // If this is a multi-recv, send only when all requests have matched.
        for (int i = 0; i < nmsgs; i++) {
            if (reqs[i] == nullptr) return false;
        }

        rdma_multi_send(slot);

        memset((void*)slots, 0, sizeof(struct FifoItem));
        memset(reqs, 0, kMaxRecv * sizeof(struct FlowRequest *));

        send_comm_->fifo_head++;
        return false;
    }
    return false;
}

void UcclFlow::process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                                    uint64_t ts4) {}

bool UcclFlow::periodic_check() {
    return true;
}

void UcclFlow::fast_retransmit() {
}

void UcclFlow::rto_retransmit() {
}

/**
 * @brief Helper function to transmit a number of chunks from the queue
 * of pending TX data.
 */
void UcclFlow::transmit_pending_chunks() {

}

void UcclFlow::deserialize_and_append_to_txtracking() {
}

void UcclFlow::prepare_l2header(uint8_t *pkt_addr) const {
    auto *eh = (ethhdr *)pkt_addr;
    memcpy(eh->h_source, local_l2_addr_, ETH_ALEN);
    memcpy(eh->h_dest, remote_l2_addr_, ETH_ALEN);
    eh->h_proto = htons(ETH_P_IP);
}

void UcclFlow::prepare_l3header(uint8_t *pkt_addr,
                                uint32_t payload_bytes) const {
    auto *ipv4h = (iphdr *)(pkt_addr + sizeof(ethhdr));
    ipv4h->ihl = 5;
    ipv4h->version = 4;
    ipv4h->tos = 0x0;
    ipv4h->id = htons(0x1513);
    ipv4h->frag_off = htons(0);
    ipv4h->ttl = 64;
#ifdef USE_TCP
    ipv4h->protocol = IPPROTO_TCP;
    ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(tcphdr) + payload_bytes);
#else
    ipv4h->protocol = IPPROTO_UDP;
    ipv4h->tot_len = htons(sizeof(iphdr) + sizeof(udphdr) + payload_bytes);
#endif
    ipv4h->saddr = htonl(local_addr_);
    ipv4h->daddr = htonl(remote_addr_);
    ipv4h->check = 0;
    // AWS would block traffic if ipv4 checksum is not calculated.
    ipv4h->check = ipv4_checksum(ipv4h, sizeof(iphdr));
}

void UcclFlow::prepare_l4header(uint8_t *pkt_addr, uint32_t payload_bytes,
                                uint16_t dst_port) const {
#ifdef USE_TCP
    auto *tcph = (tcphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
    memset(tcph, 0, sizeof(tcphdr));
#ifdef USE_MULTIPATH
    tcph->source = htons(BASE_PORT);
    tcph->dest = htons(dst_port);
#else
    tcph->source = htons(BASE_PORT);
    tcph->dest = htons(BASE_PORT);
#endif
    tcph->doff = 5;
    tcph->check = 0;
#ifdef ENABLE_CSUM
    tcph->check = ipv4_udptcp_cksum(IPPROTO_TCP, local_addr_, remote_addr_,
                                    sizeof(tcphdr) + payload_bytes, tcph);
#endif
#else
    auto *udph = (udphdr *)(pkt_addr + sizeof(ethhdr) + sizeof(iphdr));
#ifdef USE_MULTIPATH
    udph->source = htons(BASE_PORT);
    udph->dest = htons(dst_port);
#else
    udph->source = htons(BASE_PORT);
    udph->dest = htons(BASE_PORT);
#endif
    udph->len = htons(sizeof(udphdr) + payload_bytes);
    udph->check = htons(0);
#ifdef ENABLE_CSUM
    udph->check = ipv4_udptcp_cksum(IPPROTO_UDP, local_addr_, remote_addr_,
                                    sizeof(udphdr) + payload_bytes, udph);
#endif
#endif
}

void UcclRDMAEngine::handle_pending_tx_work(void)
{
    for (auto it = pending_tx_work_.begin(); it != pending_tx_work_.end();) {
        auto tx_work = *it;
        auto flow = active_flows_map_[tx_work.flow_id];
        if (flow == nullptr) {
            LOG(ERROR) << "Flow not found";
            it = pending_tx_work_.erase(it);
            continue;
        }

        if (flow->tx_messages(tx_work)) {
            it++;
        } else {
            // Good, the tx work is done.
            it = pending_tx_work_.erase(it);
        }
    }
}

void UcclRDMAEngine::handle_completion(void) 
{
    // First, poll the CQ for Ctrl QPs.
    for (auto flow: active_flows_map_) {
        flow.second->complete_ctrl_cq();
    }

    // Second, poll FIFO CQ.
    for (auto it = fifo_cq_list_.begin(); it != fifo_cq_list_.end();) {
        auto flow = *it;
        if (flow->complete_fifo_cq()) {
            it = fifo_cq_list_.erase(it);
        } else {
            it++;
        }
    }
    
    // Third, poll the CQ for UC QPs.
    for (auto flow: active_flows_map_) {
        flow.second->complete_uc_cq();
    }

}

void UcclRDMAEngine::handle_async_recv(void) 
{
    Channel::Msg rx_work;
    if (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) ==
        1) {
        LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kRX";
        active_flows_map_[rx_work.flow_id]->app_supply_rx_buf(rx_work);
    }
}

void UcclRDMAEngine::handle_async_send(void)
{
    Channel::Msg tx_work;

    if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) ==
        1) {
        // Make data written by the app thread visible to the engine.
        std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);

        LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kTX";

        if (active_flows_map_[tx_work.flow_id]->tx_messages(tx_work))
            pending_tx_work_.push_back(tx_work);
    }
}

void UcclRDMAEngine::drain_send_queues(void)
{
    for (auto flow: active_flows_map_) {
        flow.second->flush_timing_wheel();
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

        sync_clock();

        handle_pending_tx_work();

        handle_async_recv();

        handle_async_send();
        
        handle_completion();

        drain_send_queues();
    
    }
    std::cout << "Engine " << engine_idx_ << " shutdown" << std::endl;
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclRDMAEngine::periodic_process() {
    // Advance the periodic ticks counter.
    periodic_ticks_++;
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
                    LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kInstallFlowRDMA";
                    handle_install_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kSyncFlowRDMA:
                    LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kSyncFlowRDMA";
                    handle_sync_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kRegMR:
                    LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kRegMR";
                    handle_regmr_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kDeregMR:
                    LOG(INFO) << "[Engine#" << engine_idx_ << "] " << "kDeregMR";
                    handle_deregmr_on_engine_rdma(ctrl_work);
                break;
            default:
                break;
        }
    }
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

    if (rdma_ctx->data_mr_) {
        LOG(ERROR) << "Only one MR is allowed";
        return;
    }

    auto *mr = ibv_reg_mr(rdma_ctx->pd_, ctrl_work.meta.ToEngine.addr, ctrl_work.meta.ToEngine.len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    DCHECK(mr != nullptr);

    rdma_ctx->data_mr_ = mr;

    LOG(INFO) << "Memory region address: "<< (uint64_t)mr->addr << ", lkey: " << mr->lkey << ", rkey: " << mr->rkey << ", size: " << mr->length;

    // Wakeup app thread waiting one endpoint.
    {
        std::lock_guard<std::mutex> lock(poll_ctx->mu);
        poll_ctx->done = true;
        poll_ctx->cv.notify_one();
    }
}

void UcclRDMAEngine::handle_deregmr_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{
    auto *flow = active_flows_map_[ctrl_work.flow_id];
    auto *poll_ctx = ctrl_work.poll_ctx;
    if (flow == nullptr) {
        LOG(ERROR) << "Flow not found";
        return;
    }
    
    auto *rdma_ctx = flow->rdma_ctx_;

    if (!rdma_ctx->data_mr_) {
        LOG(ERROR) << "MR not found";
        return;
    }

    ibv_dereg_mr(rdma_ctx->data_mr_);
    rdma_ctx->data_mr_ = nullptr;

    // Wakeup app thread waiting one endpoint.
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

    LOG(INFO) << "Remote FIFO addr " << comm_base->remote_ctx.fifo_addr
              << " key " << comm_base->remote_ctx.fifo_key;

    if (rdma_ctx->sync_cnt_ < kPortEntropy) {
        // UC QPs.
        auto qp = rdma_ctx->uc_qps_[rdma_ctx->sync_cnt_].qp;
        rdma_ctx->uc_qps_[rdma_ctx->sync_cnt_].remote_psn = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(qp, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTR";
        ret = modify_qp_rts(qp, rdma_ctx, rdma_ctx->uc_qps_[rdma_ctx->sync_cnt_].local_psn, false);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else if (rdma_ctx->sync_cnt_ == kPortEntropy) {
        // Ctrl QP.
        rdma_ctx->ctrl_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->ctrl_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTR";
        ret = modify_qp_rts(rdma_ctx->ctrl_qp_, rdma_ctx, rdma_ctx->ctrl_local_psn_, false);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else if (rdma_ctx->sync_cnt_ == kPortEntropy + 1) {
        // Fifo QP.
        rdma_ctx->fifo_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->fifo_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn);
        DCHECK(ret == 0) << "Failed to modify Fifo QP to RTR";
        ret = modify_qp_rts(rdma_ctx->fifo_qp_, rdma_ctx, rdma_ctx->fifo_local_psn_, true);
        DCHECK(ret == 0) << "Failed to modify Fifo QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else {
        LOG(ERROR) << "Invalid sync_cnt_ " << rdma_ctx->sync_cnt_;
    }

    // Wakeup app thread waiting one endpoint.
    if (rdma_ctx->sync_cnt_ == RDMAContext::kTotalQP) {
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

    // Wakeup app thread waiting one endpoint.
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

    LOG(INFO) << engine_vec_.size();
    for (int engine_id = 0, engine_cpu_id = engine_cpu_start;
         engine_id < total_num_engines; engine_id++, engine_cpu_id++) {
        
        auto dev = engine_id / num_engines_per_dev;
        auto local_ip_str = rdma_dev_list_[dev].local_ip_str;
        
        engine_vec_.emplace_back(std::make_unique<UcclRDMAEngine>(
            dev, engine_id, channel_vec_[engine_id], local_ip_str));
        
        engine_th_vec_.emplace_back(std::make_unique<std::thread>(
            [engine_ptr = engine_vec_.back().get(), engine_id, engine_cpu_id]() {
                LOG(INFO) << "[Engine#" << engine_id << "] "
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

    // Create listening socket
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(listen_fd_ >= 0) << "ERROR: opening socket";

    int flag = 1;
    DCHECK(setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &flag,
                      sizeof(int)) >= 0)
        << "ERROR: setsockopt SO_REUSEADDR fails";

    struct sockaddr_in serv_addr;
    bzero((char *)&serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(kBootstrapPort);
    DCHECK(bind(listen_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >=
           0)
        << "ERROR: binding";

    DCHECK(!listen(listen_fd_, 128)) << "ERROR: listen";
    LOG(INFO) << "[Endpoint] server ready, listening on port "
              << kBootstrapPort;
}

RDMAEndpoint::~RDMAEndpoint() {
    for (auto &engine : engine_vec_) engine->shutdown();
    for (auto &engine_th : engine_th_vec_) engine_th->join();
    for (int i = 0; i < num_engines_per_dev_; i++) delete channel_vec_[i];

    delete ctx_pool_;
    delete[] ctx_pool_buf_;

    close(listen_fd_);

    {
        std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
        for (auto &[flow_id, boostrap_id] : bootstrap_fd_map_) {
            close(boostrap_id);
        }
    }

    static std::once_flag flag_once;
    std::call_once(flag_once, [&]() { 
        RDMAFactory::shutdown();
    });

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
    serv_addr.sin_port = htons(kBootstrapPort);

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {0};
    localaddr.sin_family = AF_INET;
    auto local_ip_str = rdma_dev_list_[dev].local_ip_str;
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    LOG(INFO) << "[Endpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        LOG(INFO) << "[Endpoint] connecting... Make sure the server is up.";
        sleep(1);
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
        LOG(INFO) << "[Endpoint] connect: receive proposed FlowID: " << std::hex
                  << "0x" << flow_id;

        // Check if the flow ID is unique, and return it to the server.
        bool unique;
        {
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
            if (unique) bootstrap_fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique) break;
    }
    
    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd, true);

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
    serv_addr.sin_port = htons(kBootstrapPort);

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {0};
    localaddr.sin_family = AF_INET;
    auto local_ip_str = rdma_dev_list_[dev].local_ip_str;
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    LOG(INFO) << "[Endpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        LOG(INFO) << "[Endpoint] connecting... Make sure the server is up.";
        sleep(1);
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
        LOG(INFO) << "[Endpoint] connect: receive proposed FlowID: " << std::hex
                  << "0x" << flow_id;

        // Check if the flow ID is unique, and return it to the server.
        bool unique;
        {
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
            if (unique) bootstrap_fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique) break;
    }
    
    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd, true);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RDMAEndpoint::uccl_accept(int dev, std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    LOG(INFO) << "[Endpoint] accept from " << remote_ip << ":"
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
        flow_id = U64Rand(0, std::numeric_limits<FlowID>::max());
        bool unique;
        {
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
            if (unique) {
                // Speculatively insert the flow ID.
                bootstrap_fd_map_[flow_id] = bootstrap_fd;
            } else {
                continue;
            }
        }

        LOG(INFO) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
                  << flow_id;

        // Ask client if this is unique
        int ret = send_message(bootstrap_fd, &flow_id, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        bool unique_from_client;
        ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique_from_client) {
            break;
        } else {
            // Remove the speculatively inserted flow ID.
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            DCHECK(1 == bootstrap_fd_map_.erase(flow_id));
        }
    }

    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd, false);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RDMAEndpoint::uccl_accept(int dev, int engine_id, std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    LOG(INFO) << "[Endpoint] accept from " << remote_ip << ":"
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
        flow_id = U64Rand(0, std::numeric_limits<FlowID>::max());
        bool unique;
        {
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            unique =
                (bootstrap_fd_map_.find(flow_id) == bootstrap_fd_map_.end());
            if (unique) {
                // Speculatively insert the flow ID.
                bootstrap_fd_map_[flow_id] = bootstrap_fd;
            } else {
                continue;
            }
        }

        LOG(INFO) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
                  << flow_id;

        // Ask client if this is unique
        int ret = send_message(bootstrap_fd, &flow_id, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        bool unique_from_client;
        ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique_from_client) {
            break;
        } else {
            // Remove the speculatively inserted flow ID.
            std::lock_guard<std::mutex> lock(bootstrap_fd_map_mu_);
            DCHECK(1 == bootstrap_fd_map_.erase(flow_id));
        }
    }

    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd, false);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

PollCtx *RDMAEndpoint::uccl_send_async(ConnID conn_id, const void *data,
                                   const size_t size) {
    auto *poll_ctx = ctx_pool_->pop();
    auto *tx_ready_poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
        .tx_ready_poll_ctx = tx_ready_poll_ctx,
    };
    
    msg.tx.data = const_cast<void *>(data);
    msg.tx.size = size;
    
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->tx_cmdq_,
                                 &msg, 1, nullptr) != 1);

    // Wait until tx is ready.
    uccl_poll(tx_ready_poll_ctx);

    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_recv_async(ConnID conn_id, void **data, size_t *size, int n) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx,
    };

    for (int i = 0; i < n; i++) {
        msg.rx.data[i] = data[i];
        msg.rx.size[i] = size[i];
    }
    msg.rx.n = n;

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
                                      const std::string &remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd, bool is_send) {
    int ret;
    struct RDMAExchangeFormatLocal meta = { 0 };
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
        VLOG(1) << oss.str();
    }
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "[Endpoint] Remote GID:\t";
        for (int i = 0; i < 16; ++i) {
            oss << ((i == 0)? "" : ":") << static_cast<int>(to_engine_meta->remote_gid.raw[i]);
        }
        VLOG(1) << oss.str();
    }

    LOG(INFO) << "[Endpoint] Sync GID done";

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
        .remote_ip = htonl(str_to_ip(remote_ip)),
        .meta = meta,
        .poll_ctx = poll_ctx,
    };
    while (jring_mp_enqueue_bulk(channel_vec_[local_engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1)
        ;

    // Wait until the flow has been installed on the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    LOG(INFO) << "[Endpoint] Install flow done" << std::endl;

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

    LOG(INFO) << "[Endpoint] Sync QPN and PSN done" << std::endl;
    
    // Send remote QPN, PSN and FifoAddr to engine.
    poll_ctx = new PollCtx();
    for (int i = 0; i < RDMAContext::kTotalQP; i++) {
        meta.ToEngine.remote_qpn = xchg_meta[i].qpn;
        meta.ToEngine.remote_psn = xchg_meta[i].psn;
        if (i == RDMAContext::kTotalQP - 1) {
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

    LOG(INFO) << "[Endpoint] Sync flow done" << std::endl;

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

        // if (engine_vec_.empty()) continue;
        // std::string s;
        // s += "\n\t[Uccl Engine] ";
        // for (auto &engine : engine_vec_) {
        //     s += engine->status_to_string();
        // }
        // LOG(INFO) << s;
    }
}

bool RDMAEndpoint::uccl_regmr(ConnID conn_id, void *addr, size_t len, int type /*unsed for now*/)
{
    auto *poll_ctx = new PollCtx();

    struct RDMAExchangeFormatLocal meta = { 0 };

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
    // Wait until the information has been received on the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    return true;
}

void RDMAEndpoint::uccl_deregmr(ConnID conn_id)
{
    auto *poll_ctx = new PollCtx();

    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kDeregMR,
        .flow_id = conn_id.flow_id,
        .poll_ctx = poll_ctx
    };

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


}  // namespace uccl