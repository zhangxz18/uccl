#include "transport.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <endian.h>
#include <string>
#include <utility>

#include <infiniband/verbs.h>

#include "transport_config.h"
#include "util.h"
#include "util_rdma.h"
#include "util_timer.h"

namespace uccl {

void UcclFlow::poll_flow_cq(void)
{
    if (!flow_cq_cnt_) return;
    
    auto comm_base = &recv_comm_.base;
    auto cq = comm_base->flow_cq;
    struct ibv_wc wcs[kMaxBatchCQ];
    
    int nb_cqe = ibv_poll_cq(cq, kMaxBatchCQ, wcs);
    for (auto i = 0; i < nb_cqe; i++) {
        auto opcode = wcs[i].opcode;
        if (opcode == IBV_WC_RDMA_WRITE) {
            // RC send completion.
            if (wcs[i].qp_num == comm_base->rc_qp->qp_num) {
                auto poll_ctx = (PollCtx *)wcs[i].wr_id;
                uccl_wakeup(poll_ctx);
            }
        } else if (opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            // RC recv completion.
            auto ureq = (struct ucclRequest *)wcs[i].wr_id;
            ureq->recv.data_len[0] = ntohl(wcs[i].imm_data);
            uccl_wakeup(ureq->poll_ctx);
        } else if (opcode == IBV_WC_RDMA_READ) {
            // GPU flush completion.
            auto *poll_ctx = reinterpret_cast<PollCtx *>(wcs[i].wr_id);
            uccl_wakeup(poll_ctx);
        }  
    }
    flow_cq_cnt_ -= nb_cqe;
}

void UcclFlow::post_flush(struct Mhandle **mhandles, void **data, int *size, int n, PollCtx *poll_ctx, int last)
{
    struct ibv_send_wr wr = {};
    wr.wr_id = (uint64_t)poll_ctx;
    wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(data[last]);
    wr.wr.rdma.rkey = mhandles[last]->mr->rkey;
    wr.sg_list = &recv_comm_.gpu_flush_sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    DCHECK(ibv_post_send(recv_comm_.gpu_flush_qp, &wr, &bad_wr) == 0);
    
    flow_cq_cnt_++;

    VLOG(5) << "Post flush: addr: " << wr.wr.rdma.remote_addr << ", rkey: " << wr.wr.rdma.rkey;
}

void UcclFlow::rc_recv(void *data, int size, struct Mhandle *mhandle, struct ibv_send_wr *wr, struct ibv_sge *sge, struct ucclRequest *ureq)
{
    auto *comm_base = &recv_comm_.base;
    struct RemFifo *rem_fifo = comm_base->fifo;
    int slot = rem_fifo->fifo_tail % kMaxReq;
    auto elems = rem_fifo->elems[slot];
    auto qp = comm_base->fifo_qp;
    
    elems[0].addr = reinterpret_cast<uint64_t>(data);
    elems[0].rkey = mhandle->mr->rkey;
    elems[0].nmsgs = 1;
    // For sender to check if the receiver is ready.
    elems[0].idx = rem_fifo->fifo_tail + 1;
    elems[0].size = size;
    // For sender to know we are using RC.
    elems[0].engine_offset = RDMAEndpoint::RC_MAGIC;

    VLOG(5) << "Post Recv: addr: " << elems[0].addr << ", rkey: " << elems[0].rkey << ", size: " << elems[0].size;
    
    memset(wr, 0, sizeof(*wr));
    // Figure out the remote address to write.
    wr->wr.rdma.remote_addr = comm_base->remote_fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
    wr->wr.rdma.rkey = comm_base->remote_fifo_rkey;

    sge->lkey = comm_base->fifo_mr->lkey;
    sge->addr = (uint64_t)elems;
    sge->length = 1 * sizeof(struct FifoItem);

    wr->sg_list = sge;
    wr->num_sge = 1;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = IBV_SEND_INLINE;

    // Occasionally post a request with the IBV_SEND_SIGNALED flag.
    if (slot == 0) {
        wr->send_flags |= IBV_SEND_SIGNALED;
        flow_cq_cnt_++;
    }

    struct ibv_send_wr *bad_wr;
    DCHECK(ibv_post_send(qp, wr, &bad_wr) == 0);

    // Post a recv buffer for consuming immedate data.
    struct ibv_recv_wr recv_wr = {};
    recv_wr.wr_id = (uint64_t)ureq;
    recv_wr.sg_list = nullptr;
    recv_wr.num_sge = 0;
    recv_wr.next = nullptr;
    struct ibv_recv_wr *bad_recv_wr;
    DCHECK(ibv_post_recv(comm_base->rc_qp, &recv_wr, &bad_recv_wr) == 0);
    flow_cq_cnt_++;

    VLOG(3) << "recv slot" << slot;

    rem_fifo->fifo_tail++;
}

struct FifoItem *UcclFlow::post_fifo(uint32_t engine_idx, void **data, int *size, int n, struct Mhandle **mhandle, 
    struct ibv_send_wr *wr, struct ibv_sge *sge)
{
    auto *comm_base = &recv_comm_.base;
    memset(wr, 0, sizeof(*wr));
    struct RemFifo *rem_fifo = comm_base->fifo;
    int slot = rem_fifo->fifo_tail % kMaxReq;
    auto elems = rem_fifo->elems[slot];
    auto qp = comm_base->fifo_qp;
    
    for (int i = 0; i < n; i++) {
        elems[i].addr = reinterpret_cast<uint64_t>(data[i]);
        elems[i].rkey = mhandle[i]->mr->rkey;
        elems[i].nmsgs = n;
        // For sender to check if the receiver is ready.
        elems[i].idx = rem_fifo->fifo_tail + 1;
        elems[i].size = size[i];
        // For sender to decide the engine.
        elems[i].engine_offset = engine_idx % ep_->num_engines_per_dev_;

        // elems[i].rid is filled by engine. See supply_rx_buff.

        VLOG(5) << "Post Recv: addr: " << elems[i].addr << ", rkey: " << elems[i].rkey << ", size: " << elems[i].size;
    }
    
    // Figure out the remote address to write.
    wr->wr.rdma.remote_addr = comm_base->remote_fifo_addr + slot * kMaxRecv * sizeof(struct FifoItem);
    wr->wr.rdma.rkey = comm_base->remote_fifo_rkey;

    sge->lkey = comm_base->fifo_mr->lkey;
    sge->addr = (uint64_t)elems;
    sge->length = n * sizeof(struct FifoItem);

    wr->sg_list = sge;
    wr->num_sge = 1;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = IBV_SEND_INLINE;

    // Occasionally post a request with the IBV_SEND_SIGNALED flag.
    if (slot == 0) {
        wr->send_flags |= IBV_SEND_SIGNALED;
        flow_cq_cnt_++;
    }

    VLOG(3) << "recv slot" << slot;

    rem_fifo->fifo_tail++;

    return elems;
}

void UcclRDMAEngine::rc_handle_completion(void) 
{
    int work = 0;
    for (auto& it : rdma_ctx_map_) {
        // Update ratio and offset
        it.second->update_clock(ratio_, offset_);

        // Poll the CQ for data path QPs.
        work += it.second->poll_rc_cq();

        // Foce check when there is no work.
        it.second->check_srq(!work);
    }
}

void UcclRDMAEngine::uc_handle_completion(void) 
{
    int work = 0;
    // First, poll the CQ for Ctrl QPs.
    for (auto& it : rdma_ctx_map_) {
        // Update ratio and offset
        it.second->update_clock(ratio_, offset_);
        work += it.second->poll_ctrl_cq();
    }

    for (auto& it : rdma_ctx_map_) {
        // Poll the CQ for Retr QP
        work += it.second->poll_retr_cq();
        // Poll the CQ for data path QPs.
        work += it.second->poll_uc_cq();
        // Foce check when there is no work.
        it.second->check_srq(!work);
        it.second->check_ctrl_rq(!work);
    }
}

void UcclRDMAEngine::handle_rx_work(void)
{
    Channel::Msg rx_work;
    int budget = kMaxRxWork;

    while (!pending_rx_works_.empty() && budget--) {
        // Process pending rx works.
        auto it = pending_rx_works_.front();
        auto rdma_ctx = it.first;
        auto ureq = it.second;
        
        if (rdma_ctx->supply_rx_buff(rx_work.ureq) == 0) {
            pending_rx_works_.pop_front();
        } else {
            return;
        }
    }

    if (budget < 0) return;

    while (budget--) {
        if (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) == 0) break;
        // Make data written by the app thread visible to the engine.
        std::ignore = std::atomic_load_explicit(&rx_work.poll_ctx->fence,
                                                std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);

        auto peer_id = rx_work.peer_id;
        auto it = rdma_ctx_map_.find(peer_id);
        DCHECK(it != rdma_ctx_map_.end());
        auto rdma_ctx = it->second;

        if (rdma_ctx->supply_rx_buff(rx_work.ureq)) {
            pending_rx_works_.push_back(std::make_pair(rdma_ctx, rx_work.ureq));
        }
    }
}

void UcclRDMAEngine::handle_tx_work(void)
{
    Channel::Msg tx_work;
    int budget = kMaxTxWork;
    uint32_t bytes = 0;

    while (budget--) {
        if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) == 0) break;
        // Make data written by the app thread visible to the engine.
        std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);

        auto peer_id = tx_work.peer_id;
        auto it = rdma_ctx_map_.find(peer_id);
        DCHECK(it != rdma_ctx_map_.end());
        auto rdma_ctx = it->second;

        rdma_ctx->tx_messages(tx_work.ureq);

        bytes += tx_work.ureq->send.data_len;

        if (bytes >= kMaxTxBytesThres) break;
    }
}

void UcclRDMAEngine::handle_timing_wheel(void)
{
    if constexpr (kTestNoTimingWheel) return;
    for (auto &it: rdma_ctx_map_) {
        it.second->burst_timing_wheel();
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
        
        handle_completion();

    }
    VLOG(4) << "Engine " << engine_idx_ << " shutdown";
}

/**
 * @brief Method to perform periodic processing. This is called by the
 * main engine cycle (see method `Run`).
 */
void UcclRDMAEngine::periodic_process() {
    // Handle RTOs for all UC QPs.
    if constexpr (!kUSERC)
        handle_rto();
    // Handle control plane requests.
    process_ctl_reqs();
}

void UcclRDMAEngine::handle_rto() {

    if constexpr (kTestNoRTO) return;

    auto expired_qp_vec = rto_tm_.check_expired();

    for (auto data : expired_qp_vec) {
        auto *rdma_ctx = reinterpret_cast<struct RDMAContext *>(data.rdma_ctx);
        auto *qpw = reinterpret_cast<struct UCQPWrapper *>(data.qpw);

        DCHECK(rdma_ctx && qpw);
        
        rdma_ctx->mark_qp_timeout(qpw);

        rdma_ctx->rto_retransmit(qpw);
    }
}

/// TODO: handle error case
void UcclRDMAEngine::process_ctl_reqs() {
    Channel::CtrlMsg ctrl_work;
    while (jring_sc_dequeue_bulk(channel_->ctrl_cmdq_, &ctrl_work, 1, nullptr) ==
        1) {
        switch (ctrl_work.opcode) {
            case Channel::CtrlMsg::kInstallCtx:
                    VLOG(6) << "[Engine#" << engine_idx_ << "] " << "kInstallCtx";
                    handle_install_ctx_on_engine(ctrl_work);
                break;
            default:
                break;
        }
    }
}

void UcclRDMAEngine::handle_install_ctx_on_engine(Channel::CtrlMsg &ctrl_work)
{
    int ret;
    auto meta = ctrl_work.meta;
    auto info = &meta.install_ctx;

    int bootstrap_fd = info->bootstrap_fd;
    auto dev = dev_;

    RDMAContext *rdma_ctx;

    {
        DCHECK(rdma_ctx_map_.find(info->peer_id) == rdma_ctx_map_.end());
        rdma_ctx = RDMAFactory::CreateContext(&rto_tm_, dev, engine_idx_ % NUM_ENGINES, meta);
        std::tie(std::ignore, ret) = rdma_ctx_map_.insert({info->peer_id, rdma_ctx});
        DCHECK(ret);
    }

    // Create a thread to handle the QP setup to avoid blocking the engine.
    std::thread qp_setup_thread([ctrl_work, rdma_ctx, bootstrap_fd, dev]() {
        auto meta = ctrl_work.meta;
        auto info = &meta.install_ctx;
        auto *poll_ctx = ctrl_work.poll_ctx;
        // Send PSN, QPN to remote peer.
        const int size = sizeof(uint32_t) + sizeof(uint32_t);
        char buf[kTotalQP * size];
        for (auto i = 0; i < kPortEntropy; i++) {
            memcpy(buf + i * size, &rdma_ctx->dp_qps_[i].local_psn, sizeof(uint32_t));
            memcpy(buf + i * size + sizeof(uint32_t), &rdma_ctx->dp_qps_[i].qp->qp_num, sizeof(uint32_t));
        }
        if constexpr (!kUSERC) {
            memcpy(buf + kPortEntropy * size, &rdma_ctx->ctrl_local_psn_, sizeof(uint32_t));
            memcpy(buf + kPortEntropy * size + sizeof(uint32_t), &rdma_ctx->ctrl_qp_->qp_num, sizeof(uint32_t));
    
            memcpy(buf + (kPortEntropy+1) * size, &rdma_ctx->retr_local_psn_, sizeof(uint32_t));
            memcpy(buf + (kPortEntropy+1) * size + sizeof(uint32_t), &rdma_ctx->retr_qp_->qp_num, sizeof(uint32_t));
        }

        int ret = send_message(bootstrap_fd, buf, kTotalQP * size);
        DCHECK(ret == kTotalQP * size);

        // Receive PSN, QPN from remote peer.
        ret = receive_message(bootstrap_fd, buf, kTotalQP * size);
        DCHECK(ret == kTotalQP * size);

        // Modify QPs to RTR and RTS.
        for (auto i = 0; i < kPortEntropy; i++) {
            auto remote_psn = *reinterpret_cast<uint32_t*>(buf + i * size);
            auto remote_qpn = *reinterpret_cast<uint32_t*>(buf + i * size + sizeof(uint32_t));
            auto qp = rdma_ctx->dp_qps_[i].qp;

            ret = modify_qp_rtr(qp, dev, &rdma_ctx->remote_ctx_, remote_qpn, remote_psn, 0);
            DCHECK(ret == 0) << "Failed to modify data path QP to RTR";

            ret = modify_qp_rts(qp, rdma_ctx->dp_qps_[i].local_psn, kUSERC);
            DCHECK(ret == 0) << "Failed to modify data path QP to RTS";
        }

        if constexpr (!kUSERC) {
            auto ctrl_rpsn = *reinterpret_cast<uint32_t*>(buf + kPortEntropy * size);
            auto ctrl_rqpn = *reinterpret_cast<uint32_t*>(buf + kPortEntropy * size + sizeof(uint32_t));
            auto ctrl_qp = rdma_ctx->ctrl_qp_;
    
            ret = modify_qp_rtr(ctrl_qp, dev, &rdma_ctx->remote_ctx_, ctrl_rqpn, ctrl_rpsn, 1);
            DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTR";
    
            ret = modify_qp_rts(ctrl_qp, rdma_ctx->ctrl_local_psn_, false);
    
            auto retr_rpsn = *reinterpret_cast<uint32_t*>(buf + (kPortEntropy+1) * size);
            auto retr_rqpn = *reinterpret_cast<uint32_t*>(buf + (kPortEntropy+1) * size + sizeof(uint32_t));
            auto retr_qp = rdma_ctx->retr_qp_;
    
            ret = modify_qp_rtr(retr_qp, dev, &rdma_ctx->remote_ctx_, retr_rqpn, retr_rpsn, 0);
            DCHECK(ret == 0) << "Failed to modify Retr QP to RTR";
    
            ret = modify_qp_rts(retr_qp, rdma_ctx->retr_local_psn_, false);
            DCHECK(ret == 0) << "Failed to modify Retr QP to RTS";
        }

        uccl_wakeup(poll_ctx);
    });

    // Detach the thread to allow it to run independently.
    qp_setup_thread.detach();
}

RDMAEndpoint::RDMAEndpoint(const uint8_t *gid_idx_list, int num_devices, int num_engines_per_dev, int engine_cpu_start)
    : num_devices_(num_devices), num_engines_per_dev_(num_engines_per_dev), engine_cpu_start_(engine_cpu_start),
         stats_thread_([this]() { stats_thread_fn(); }) {
    
    // Initialize all RDMA devices.
    static std::once_flag flag_once;
    std::call_once(flag_once, [&]() {
        for (int i = 0; i < num_devices; i++) {
            RDMAFactory::init_dev(gid_idx_list[i]);
        }
    });

    rdma_ctl_ = rdma_ctl;

    int total_num_engines = num_devices * num_engines_per_dev;

    CHECK_LE(ENGINE_CPU_START + total_num_engines - 1, NUM_CPUS)
        << "The number of engines exceeds the number of CPUs";

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
        create_listen_socket(&test_listen_fds_[i], kTestListenPort + i);
    }
}

void UcclRDMAEngine::release()
{
    for (auto &it : rdma_ctx_map_) {
        delete it.second;
    }
    rdma_ctx_map_.clear();
}

RDMAEndpoint::~RDMAEndpoint() {
    for (auto &engine : engine_vec_) engine->shutdown();
    for (auto &engine_th : engine_th_vec_) engine_th->join();
    for (auto &engine : engine_vec_) engine->release();

    for (int dev = 0; dev < num_devices_; dev++) {
        for (auto &it: active_flows_map_[dev]) {
            auto flow = it.second;
            flow->release();
            delete flow;
        }
        active_flows_map_[dev].clear();

        peer_map_[dev].clear();

        close(test_listen_fds_[dev]);
    }

    for (int i = 0; i < num_devices_ * num_engines_per_dev_; i++) delete channel_vec_[i];

    delete ctx_pool_;
    delete[] ctx_pool_buf_;

    for (auto &[flow_id, boostrap_fd] : fd_map_) {close(boostrap_fd);}
    fd_map_.clear();

    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        shutdown_ = true;
        stats_cv_.notify_all();
    }

    stats_thread_.join();
}

PollCtx *RDMAEndpoint::install_ctx_on_engine(uint32_t engine_idx, union CtrlMeta meta)
{
    auto *cmdq = channel_vec_[engine_idx]->ctrl_cmdq_;
    
    auto *poll_ctx = ctx_pool_->pop();
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kInstallCtx,
        .meta = meta,
        .poll_ctx = poll_ctx,
    };

    while (jring_mp_enqueue_bulk(cmdq, &ctrl_msg, 1, nullptr) != 1) {}

    return poll_ctx;
}

void RDMAEndpoint::safe_install_ctx(int dev, int bootstrap_fd, bool local_lock_first, std::string &remote_ip, int remote_dev, 
    PeerID *peer_id, struct RemoteRDMAContext *remote_ctx)
{
    if (local_lock_first) {
        peer_map_mu_[dev].lock();
        auto it = peer_map_[dev].find({remote_ip, remote_dev});
        if (it == peer_map_[dev].end()) { // This device has not connected to the remote device yet.
            // Tell remote side we have already held the lock.
            send_ready(bootstrap_fd);
            // Wait until remote side holds its lock.
            wait_ready(bootstrap_fd);
            *peer_id = next_peer_id_[dev]++;
            // Install RDMAContexts on all engines.
            install_ctx_on_engines(bootstrap_fd, dev, *peer_id, remote_ctx);
            peer_map_[dev].insert({{remote_ip, remote_dev}, {*peer_id, remote_ctx->remote_gid, remote_ctx->remote_port_attr, 1}});
            // Wait until remote side releases its lock.
            wait_ready(bootstrap_fd);
            // Release the lock and tell remote side we have released the lock.
            peer_map_mu_[dev].unlock();
            send_ready(bootstrap_fd);
        } else {
            // This device has connected to the remote device.
            *peer_id = it->second.peer_id;
            remote_ctx->remote_gid = it->second.remote_gid;
            remote_ctx->remote_port_attr = it->second.remote_port_attr;
            it->second.flow_cnt++;
            // Release the lock and tell remote side we have released the lock.
            peer_map_mu_[dev].unlock();
            send_abort(bootstrap_fd);
        }
    } else {
        bool installed = !wait_sync(bootstrap_fd);
        if (installed) {
            // Remote side tell us that this device has connected to the remote device.
            peer_map_mu_[dev].lock();
            auto it = peer_map_[dev].find({remote_ip, remote_dev});
            DCHECK(it != peer_map_[dev].end());
            *peer_id = it->second.peer_id;
            remote_ctx->remote_gid = it->second.remote_gid;
            remote_ctx->remote_port_attr = it->second.remote_port_attr;
            it->second.flow_cnt++;
            peer_map_mu_[dev].unlock();
        } else {
            // Hold the lock and tell remote side we have already held the lock.
            peer_map_mu_[dev].lock();
            auto it = peer_map_[dev].find({remote_ip, remote_dev});
            DCHECK(it == peer_map_[dev].end());
            send_ready(bootstrap_fd);
            *peer_id = next_peer_id_[dev]++;
            // Install RDMAContexts on all engines.
            install_ctx_on_engines(bootstrap_fd, dev, *peer_id, remote_ctx);
            peer_map_[dev].insert({{remote_ip, remote_dev}, {*peer_id, remote_ctx->remote_gid, remote_ctx->remote_port_attr, 1}});
            // Release the lock and tell remote side we have released the lock.
            peer_map_mu_[dev].unlock();
            send_ready(bootstrap_fd);
            // Wait until remote side releases its lock.
            wait_ready(bootstrap_fd);
        }
    }
}

void RDMAEndpoint::install_ctx_on_engines(int fd, int dev, PeerID peer_id, struct RemoteRDMAContext *remote_ctx)
{
    union CtrlMeta meta = {};
    auto *info = &meta.install_ctx;

    // synchronize GID and PortAttr with remote peer.
    int ret;
    auto factory_dev = RDMAFactory::get_factory_dev(dev);
    
    ret = send_message(fd, &factory_dev->gid.raw, 16);
    DCHECK(ret == 16) << "Failed to send GID";
    ret = receive_message(fd, &info->remote_gid.raw, 16);
    DCHECK(ret == 16) << "Failed to receive GID";

    ret = send_message(fd, &factory_dev->port_attr, sizeof(ibv_port_attr));
    DCHECK(ret == sizeof(ibv_port_attr)) << "Failed to send PortAttr";
    ret = receive_message(fd, &info->remote_port_attr, sizeof(ibv_port_attr));
    DCHECK(ret == sizeof(ibv_port_attr)) << "Failed to receive PortAttr";

    info->bootstrap_fd = fd;
    info->peer_id = peer_id;

    for (int i = 0; i < num_engines_per_dev_; i++) {
        auto engine_idx = find_first_engine_idx_on_dev(dev) + i;
        auto *poll_ctx = install_ctx_on_engine(engine_idx, meta);
        uccl_poll(poll_ctx);
    }

    remote_ctx->remote_gid = info->remote_gid;
    remote_ctx->remote_port_attr = info->remote_port_attr;
}

ConnID RDMAEndpoint::uccl_connect(int dev, int remote_dev, std::string remote_ip, uint16_t remote_port)
{
    struct sockaddr_in serv_addr = {};
    struct hostent *server;
    int ret;
    int bootstrap_fd;
    bool local_lock_first = false;

    bootstrap_fd = socket(AF_INET, SOCK_STREAM, 0);
    DCHECK(bootstrap_fd >= 0) << "uccl_connect: socket()";

    server = gethostbyname(remote_ip.c_str());
    DCHECK(server) << "uccl_connect: gethostbyname()";

    // Force the socket to bind to the local IP address.
    sockaddr_in localaddr = {};
    localaddr.sin_family = AF_INET;
    auto *factory_dev = RDMAFactory::get_factory_dev(dev);
    localaddr.sin_addr.s_addr = str_to_ip(factory_dev->local_ip_str.c_str());
    ret = bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));
    DCHECK(ret == 0) << "uccl_connect: bind()";

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = str_to_ip(remote_ip.c_str());
    serv_addr.sin_port = htons(remote_port);

    VLOG(5) << "[Endpoint] connecting to " << "<" << remote_ip << ", " << remote_dev << ">:" << remote_port;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        VLOG(5) << "[Endpoint] connecting... Make sure the server is up.";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));

    FlowID flow_id;
    while (true) {
        ret = receive_message(bootstrap_fd, &flow_id, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID)) << "uccl_connect: receive_message()";

        // Check if the flow ID is unique, and return it to the server.
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            unique =
                (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) fd_map_[flow_id] = bootstrap_fd;
        }

        ret = send_message(bootstrap_fd, &unique, sizeof(bool));
        DCHECK(ret == sizeof(bool)) << "uccl_connect: send_message()";

        if (unique) {
            // Send our device ID to the server.
            ret = send_message(bootstrap_fd, &dev, sizeof(int));
            DCHECK(ret == sizeof(int)) << "uccl_connect: send_message()";
            break;
        }
    }

    if (str_to_ip(factory_dev->local_ip_str.c_str()) < str_to_ip(remote_ip.c_str())) {
        local_lock_first = true;
    } else if (str_to_ip(factory_dev->local_ip_str.c_str()) == str_to_ip(remote_ip.c_str())) {
        // Handle the intra-node case.
        if (dev < remote_dev) local_lock_first = true;
    }

    PeerID peer_id;
    struct RemoteRDMAContext remote_ctx;

    safe_install_ctx(dev, bootstrap_fd, local_lock_first, remote_ip, remote_dev, &peer_id, &remote_ctx);

    // Install flow on Endpoint.
    auto *flow = new UcclFlow(this, bootstrap_fd, dev, peer_id, flow_id, remote_ctx, remote_ip, remote_dev, true);
    DCHECK(flow);
    {
        active_flows_spin_[dev].Lock();
        std::tie(std::ignore, ret) = active_flows_map_[dev].insert({flow_id, flow});
        active_flows_spin_[dev].Unlock();
        DCHECK(ret);
    }

    return ConnID{.context = flow,
                  .flow_id = flow_id,
                  .peer_id = peer_id,
                  .dev = dev};
}

ConnID RDMAEndpoint::uccl_accept(int dev, int listen_fd, std::string &remote_ip, int *remote_dev)
{
    struct sockaddr_in cli_addr;
    socklen_t clien = sizeof(cli_addr);
    int bootstrap_fd;
    int ret;
    bool local_lock_first = false;

    bootstrap_fd = accept(listen_fd, (struct sockaddr *)&cli_addr, &clien);
    DCHECK(bootstrap_fd >= 0) << "uccl_accept: accept()";
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    VLOG(5) << "[Endpoint] accept from " << remote_ip << ":" << cli_addr.sin_port;

    fcntl(bootstrap_fd, F_SETFL, O_NONBLOCK);
    int flag = 1;
    setsockopt(bootstrap_fd, IPPROTO_TCP, TCP_NODELAY, (void *)&flag,
               sizeof(int));
     
    // Generate unique flow ID for both client and server.
    FlowID flow_id;
    while (true) {
        bool unique;
        {
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            // generate flow_id sequentially for better debugging
            static uint64_t fff = 0;
            flow_id = fff++;
            unique = (fd_map_.find(flow_id) == fd_map_.end());
            if (unique) {
                // Speculatively insert the flow ID.
                fd_map_[flow_id] = bootstrap_fd;
            } else {
                continue;
            }
        }

        VLOG(5) << "[Endpoint] accept: propose FlowID: " << std::hex << "0x"
                  << flow_id;

        // Let client use flow_id + 50000 and ask client if this is unique
        FlowID cid = flow_id + 50000;
        int ret = send_message(bootstrap_fd, &cid, sizeof(FlowID));
        DCHECK(ret == sizeof(FlowID));
        bool unique_from_client;
        ret = receive_message(bootstrap_fd, &unique_from_client, sizeof(bool));
        DCHECK(ret == sizeof(bool));

        if (unique_from_client) {
            // Receive the remote_dev from client.
            ret = receive_message(bootstrap_fd, remote_dev, sizeof(int));
            DCHECK(ret == sizeof(int));
            break;
        } else {
            // Remove the speculatively inserted flow ID.
            std::lock_guard<std::mutex> lock(fd_map_mu_);
            DCHECK(1 == fd_map_.erase(flow_id));
        }
    }

    auto factory_dev = RDMAFactory::get_factory_dev(dev);
    if (str_to_ip(factory_dev->local_ip_str.c_str()) < str_to_ip(remote_ip.c_str())) {
        local_lock_first = true;
    } else if (str_to_ip(factory_dev->local_ip_str.c_str()) == str_to_ip(remote_ip.c_str())) {
        // Handle the intra-node case.
        if (dev < *remote_dev) local_lock_first = true;
    }

    PeerID peer_id;
    struct RemoteRDMAContext remote_ctx;
    
    safe_install_ctx(dev, bootstrap_fd, local_lock_first, remote_ip, *remote_dev, &peer_id, &remote_ctx);

    // Install flow on Endpoint.
    auto *flow = new UcclFlow(this, bootstrap_fd, dev, peer_id, flow_id, remote_ctx, remote_ip, *remote_dev, false);
    DCHECK(flow);
    {
        active_flows_spin_[dev].Lock();
        std::tie(std::ignore, ret) = active_flows_map_[dev].insert({flow_id, flow});
        active_flows_spin_[dev].Unlock();
        DCHECK(ret);
    }

    return ConnID{.context = flow,
                  .flow_id = flow_id,
                  .peer_id = peer_id,
                  .dev = dev};
}

bool UcclFlow::check_fifo_ready(int *ret_slot, int *ret_nmsgs)
{
    int slot = send_comm_.fifo_head % kMaxReq;
    auto rem_fifo = send_comm_.base.fifo;
    volatile struct FifoItem *slots = rem_fifo->elems[slot];

    auto idx = send_comm_.fifo_head + 1;
    if (slots[0].idx != idx)
        return false;

    // Wait until all slots are ready
    auto nmsgs = slots[0].nmsgs;
    for (int i = 1; i < nmsgs; i++) while(slots[i].idx != idx) {}

    VLOG(3) << "Receiver is ready to receive";

    __sync_synchronize();

    *ret_slot = slot;
    *ret_nmsgs = nmsgs;
    
    return true;
}

void UcclFlow::rc_send(struct ucclRequest *ureq)
{
    auto *qp = send_comm_.base.rc_qp;
    auto size = ureq->send.data_len;
    auto laddr = ureq->send.laddr;
    auto raddr = ureq->send.raddr;
    auto lkey = ureq->send.lkey;
    auto rkey = ureq->send.rkey;

    struct ibv_sge sge;
    struct ibv_send_wr wr, *bad_wr = nullptr;

    sge.addr = laddr;
    sge.lkey = lkey;
    sge.length = size;

    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;
    
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.imm_data = htonl(size);
    wr.wr.rdma.remote_addr = raddr;
    wr.wr.rdma.rkey = rkey;

    wr.send_flags = IBV_SEND_SIGNALED;

    wr.wr_id = (uint64_t)ureq->poll_ctx;

    DCHECK(ibv_post_send(qp, &wr, &bad_wr) == 0) << "Failed to post send";
    flow_cq_cnt_++;

}

void UcclFlow::post_multi_send(struct ucclRequest **ureqs, uint32_t engine_offset)
{
    if (engine_offset == RDMAEndpoint::RC_MAGIC) {
        ureqs[0]->type = ReqTxRC;
        rc_send(ureqs[0]);
        return;
    }

    uint32_t engine_idx = ep_->find_first_engine_idx_on_dev(dev_) + engine_offset;
    auto txq = ep_->channel_vec_[engine_idx]->tx_cmdq_;
    auto n = ureqs[0]->n;
    Channel::Msg msgs[kMaxRecv];
    for (int i = 0; i < n; i++) {
        msgs[i].opcode = Channel::Msg::Op::kTx;
        msgs[i].peer_id = peer_id_;
        ureqs[i]->n = i; // mid
        msgs[i].ureq = ureqs[i];
        msgs[i].poll_ctx = ureqs[i]->poll_ctx;
    }

    while (jring_mp_enqueue_bulk(txq, msgs, n, nullptr) != n) {}
}

int RDMAEndpoint::uccl_send_async(UcclFlow *flow, struct Mhandle *mhandle, const void *data,
                                   const size_t size, struct ucclRequest *ureq) 
{
    ureq->type = ReqTx;
    ureq->send.data_len = size;

    int slot, nmsg;
    
    if (!flow->check_fifo_ready(&slot, &nmsg))
        return -1;
    auto send_comm = &flow->send_comm_;
    auto ureqs = send_comm->fifo_ureqs[slot];
    auto rem_fifo = send_comm->base.fifo;
    volatile struct FifoItem *slots = rem_fifo->elems[slot];

    auto *poll_ctx = ctx_pool_->pop();

    for (int i = 0; i < nmsg; i++) {
        if (ureqs[i] != nullptr) continue;
        DCHECK(!(slots[i].size < 0 || slots[i].addr == 0 || slots[i].rkey == 0));

        if (size > slots[i].size) {
            // Can't send more than what the receiver can receive.
            // Adjust data_len to the actual size sent.
            ureq->send.data_len = slots[i].size;
        }

        ureq->send.laddr = (uint64_t)data;
        ureq->send.lkey = mhandle->mr->lkey;
        ureq->send.raddr = slots[i].addr;
        ureq->send.rkey = slots[i].rkey;
        ureq->n = nmsg;
        ureq->send.rid = slots[i].rid;
        ureq->poll_ctx = poll_ctx;
        ureq->context = flow;
        // Temporarily set tx_events to 1 to indicate size mismatch.
        ureq->send.tx_events = size < slots[i].size ? 1 : 0;
        // Track this request.
        ureqs[i] = ureq;

        // If this is a multi-recv, send only when all requests have matched.
        for (int i = 0; i < nmsg; i++) {
            if (ureqs[i] == nullptr) return 0;
        }

        // All requests have matched. Post works to the engine.
        flow->post_multi_send(ureqs, slots[i].engine_offset);
        
        // Move the head of the FIFO.
        send_comm->fifo_head++;

        memset((void*)slots, 0, sizeof(struct FifoItem));
        memset(ureqs, 0, kMaxRecv * sizeof(struct ucclRequest *));

        VLOG(3) << "send_async: posted " << nmsg << " requests" << " on engine " << slots[i].engine_offset
            << " size: " << size << " slot: " << slot;

        return 0;
    }

    return 0;
}

bool RDMAEndpoint::uccl_poll_ureq_once(struct ucclRequest *ureq) {
    UcclFlow *flow = reinterpret_cast<UcclFlow *>(ureq->context);
    if (ureq->type == ReqTxRC || ureq->type == ReqRxRC) {flow->poll_flow_cq();} 
    auto ret = uccl_poll_once(ureq->poll_ctx);
    if ((ureq->type == ReqRx || ureq->type == ReqRxRC) && ret) {flow->dec_outstanding_reqs();}
    return ret;
}

int RDMAEndpoint::uccl_flush(UcclFlow *flow, struct Mhandle **mhandles, void **data, int *size, int n, struct ucclRequest *ureq)
{
    flow->poll_flow_cq();

    int last = flow->check_need_flush(size, n);
    if (last == -1) return 0;

    auto *poll_ctx = ctx_pool_->pop();
    flow->post_flush(mhandles, data, size, n, poll_ctx, last);

    ureq->type = ReqFlush;
    ureq->poll_ctx = poll_ctx;

    return 0;
}

int RDMAEndpoint::uccl_recv_async(UcclFlow *flow, struct Mhandle **mhandles, void **data, int *size, int n, struct ucclRequest *ureq) 
{
    auto dev = flow->dev_;
    
    if (!flow->check_room()) {return -1;}

    flow->inc_outstanding_reqs();

    if (size[0] <= kRCSize && n == 1) {
        flow->rc_recv(data[0], size[0], mhandles[0], &ureq->recv.wr, &ureq->recv.sge, ureq);
        ureq->type = ReqRxRC;
        ureq->context = flow;
        ureq->poll_ctx = ctx_pool_->pop();

        flow->poll_flow_cq();

        return 0;
    }

    uint32_t candidate = find_first_engine_idx_on_dev(dev) + flow->next_engine_offset_;
    if constexpr (!kBindEngine) 
        flow->next_engine_offset_ = (flow->next_engine_offset_ + 1) % num_engines_per_dev_;
    
    auto elems = flow->post_fifo(candidate, data, size, n, mhandles, &ureq->recv.wr, &ureq->recv.sge);
    ureq->type = ReqRx;
    ureq->context = flow;
    ureq->n = n;
    for (int i = 0; i < n; i++) ureq->recv.data_len[i] = size[i];
    ureq->poll_ctx = ctx_pool_->pop();
    ureq->recv.elems = elems;
    ureq->recv.qp = flow->recv_comm_.base.fifo_qp;

    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .peer_id = flow->peer_id_,
        .ureq = ureq,
        .poll_ctx = ureq->poll_ctx,
    };

    auto rxq = channel_vec_[candidate]->rx_cmdq_;
    while (jring_mp_enqueue_bulk(rxq, &msg, 1, nullptr) != 1) {}

    VLOG(3) << "recv_async: posted " << n << " requests" << " on engine " << candidate
        << " size: " << size[0];
    
    flow->poll_flow_cq();

    return 0;
}

inline void RDMAEndpoint::put_load_on_engine(int engine_id)
{
    engine_load_vec_[engine_id]++;
}

inline int RDMAEndpoint::find_least_loaded_engine_idx_and_update(int dev) {
    std::lock_guard<std::mutex> lock(engine_load_vec_mu_);

    // Determine the range of engines serving the device.
    auto si = engine_load_vec_.at(dev * num_engines_per_dev_);
    auto ei = engine_load_vec_.at((dev + 1) * num_engines_per_dev_);

    auto minElementIter = std::min_element(engine_load_vec_.begin() + si, engine_load_vec_.begin() + ei);
    *minElementIter += 1;
    return std::distance(engine_load_vec_.begin(), minElementIter);
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
        uint32_t eidx = 0;
        for (auto &engine : engine_vec_) {
            s = engine->status_to_string();
            #ifdef STATS
            if (!s.empty()) {
                std::cout << "[Engine#" << std::to_string(eidx++) << "]\n";
                std::cout << s;
            }
            #endif
        }
    }
}

int RDMAEndpoint::uccl_regmr_dmabuf(UcclFlow *flow, void *addr, size_t len, int type, int offset, int fd, struct Mhandle **mhandle)
{
    auto factory_dev = RDMAFactory::get_factory_dev(flow->dev_);
    *mhandle = new Mhandle();

    (*mhandle)->mr = ibv_reg_dmabuf_mr(factory_dev->pd, offset, len, (uint64_t)addr, fd, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    return 0;
}

int RDMAEndpoint::uccl_regmr(UcclFlow *flow, void *addr, size_t len, int type /*unsed for now*/, struct Mhandle **mhandle)
{
    auto factory_dev = RDMAFactory::get_factory_dev(flow->dev_);

    *mhandle = new Mhandle();
    (*mhandle)->mr = ibv_reg_mr(factory_dev->pd, addr, len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);

    return 0;
}

void RDMAEndpoint::uccl_deregmr(struct Mhandle *mhandle)
{
    ibv_dereg_mr(mhandle->mr);
    delete mhandle;
}

std::string UcclRDMAEngine::status_to_string()
{
    std::string s;

    for (auto rdma_ctx : rdma_ctx_map_) {
        s += "    [Context#" + std::to_string(rdma_ctx.first) + "]";
        s += rdma_ctx.second->to_string();
    }

    return s;
}

}  // namespace uccl
