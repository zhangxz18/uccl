#include "transport.h"
#include "transport_config.h"
#include "util_rdma.h"
#include <infiniband/verbs.h>

namespace uccl {

std::string UcclFlow::to_string() const {
    std::string s;
    s += "\n\t\t\t" + pcb_.to_string();
    return s;
}

void UcclFlow::rx_messages() {

}

/**
 * @brief Application supplies a buffer to the flow for receiving data.
 * @param rx_work 
 */
void UcclFlow::rx_supply_app_buf(Channel::Msg &rx_work) {
    auto *buf = rx_work.data;
    auto len = rx_work.len_p;



}

void UcclFlow::tx_messages(Channel::Msg &tx_work) {
}

void UcclFlow::process_rttprobe_rsp(uint64_t ts1, uint64_t ts2, uint64_t ts3,
                                    uint64_t ts4) {
    auto rtt_ns = (ts4 - ts1) - (ts3 - ts2);
    auto sample_rtt_tsc = ns_to_cycles(rtt_ns, freq_ghz);
    pcb_.update_rate(rdtsc(), sample_rtt_tsc);

    VLOG(3) << "sample_rtt_us " << to_usec(sample_rtt_tsc, freq_ghz)
            << " us, avg_rtt_diff " << pcb_.timely.get_avg_rtt_diff()
            << " us, timely rate " << pcb_.timely.get_rate_gbps() << " Gbps";

#ifdef RTT_STATS
    rtt_stats_.update(rtt_ns / 1000);
    if (++rtt_probe_count_ % 100000 == 0) {
        FILE *fp = fopen("rtt_stats.txt", "w");
        rtt_stats_.print(fp);
        fclose(fp);
    }
#endif
}

bool UcclFlow::periodic_check() {
    return true;
}


void UcclFlow::fast_retransmit() {
}

void UcclFlow::rto_retransmit() {
}

/**
 * @brief Helper function to transmit a number of packets from the queue
 * of pending TX data.
 */
void UcclFlow::transmit_pending_packets() {
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

void UcclRDMAEngine::run() {
    Channel::Msg rx_work;
    Channel::Msg tx_work;

    while (!shutdown_) {
        // Calculate the cycles elapsed since last periodic processing.
        auto now_tsc = rdtsc();
        const auto elapsed_tsc = now_tsc - last_periodic_tsc_;

        if (elapsed_tsc >= kSlowTimerIntervalTsc_) {
            // Perform periodic processing.
            periodic_process();
            last_periodic_tsc_ = now_tsc;
        }

        if (jring_sc_dequeue_bulk(channel_->rx_cmdq_, &rx_work, 1, nullptr) ==
            1) {
            VLOG(3) << "Rx jring dequeue";
            active_flows_map_[rx_work.flow_id]->rx_supply_app_buf(rx_work);
        }

        if (jring_sc_dequeue_bulk(channel_->tx_cmdq_, &tx_work, 1, nullptr) ==
            1) {
            // Make data written by the app thread visible to the engine.
            std::ignore = std::atomic_load_explicit(&tx_work.poll_ctx->fence,
                                                    std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);

            (void)tx_work;
            VLOG(3) << "Tx jring dequeue";
        }
    }
    std::cout << "Engine " << local_engine_idx_ << " shutdown" << std::endl;
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
                    LOG(INFO) << "[Engine#" << local_engine_idx_ << "] " << "kInstallFlowRDMA";
                    handle_install_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kSyncFlowRDMA:
                    LOG(INFO) << "[Engine#" << local_engine_idx_ << "] " << "kSyncFlowRDMA";
                    handle_sync_flow_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kRegMR:
                    LOG(INFO) << "[Engine#" << local_engine_idx_ << "] " << "kRegMR";
                    handle_regmr_on_engine_rdma(ctrl_work);
                break;
            case Channel::CtrlMsg::kDeregMR:
                    LOG(INFO) << "[Engine#" << local_engine_idx_ << "] " << "kDeregMR";
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

    if (rdma_ctx->data_pd_ && rdma_ctx->data_mr_) {
        LOG(ERROR) << "Only one MR is allowed";
        return;
    }

    auto *pd = ibv_alloc_pd(rdma_ctx->context_);
    DCHECK(pd != nullptr);
    auto *mr = ibv_reg_mr(pd, ctrl_work.meta.ToEngine.addr, ctrl_work.meta.ToEngine.len, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    DCHECK(mr != nullptr);

    rdma_ctx->data_pd_ = pd;
    rdma_ctx->data_mr_ = mr;

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

    if (!rdma_ctx->data_pd_ || !rdma_ctx->data_mr_) {
        LOG(ERROR) << "MR not found";
        return;
    }

    ibv_dereg_mr(rdma_ctx->data_mr_);
    ibv_dealloc_pd(rdma_ctx->data_pd_);

    rdma_ctx->data_mr_ = nullptr;
    rdma_ctx->data_pd_ = nullptr;

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

    if (rdma_ctx->sync_cnt_ < kPortEntropy) {
        // UC QPs.
        auto qp = rdma_ctx->qp_vec_[rdma_ctx->sync_cnt_];
        rdma_ctx->remote_psn_[rdma_ctx->sync_cnt_] = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(qp, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn, false);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTR";
        ret = modify_qp_rts(qp, rdma_ctx, rdma_ctx->local_psn_[rdma_ctx->sync_cnt_], false);
        DCHECK(ret == 0) << "Failed to modify UC QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else if (rdma_ctx->sync_cnt_ == kPortEntropy) {
        // Ctrl QP.
        rdma_ctx->ctrl_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->ctrl_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn, false);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTR";
        ret = modify_qp_rts(rdma_ctx->ctrl_qp_, rdma_ctx, rdma_ctx->ctrl_local_psn_, false);
        DCHECK(ret == 0) << "Failed to modify Ctrl QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else if (rdma_ctx->sync_cnt_ == kPortEntropy + 1) {
        // Retr QP.
        rdma_ctx->retr_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->retr_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn, true);
        DCHECK(ret == 0) << "Failed to modify Retr QP to RTR";
        ret = modify_qp_rts(rdma_ctx->retr_qp_, rdma_ctx, rdma_ctx->retr_local_psn_, true);
        DCHECK(ret == 0) << "Failed to modify Retr QP to RTS";
        rdma_ctx->sync_cnt_++;
    } else if (rdma_ctx->sync_cnt_ == kPortEntropy + 2) {
        // Fifo Qp.
        rdma_ctx->fifo_remote_psn_ = meta.ToEngine.remote_psn;
        ret = modify_qp_rtr(rdma_ctx->fifo_qp_, rdma_ctx, meta.ToEngine.remote_qpn, meta.ToEngine.remote_psn, true);
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

    auto *flow = new UcclFlow(local_engine_idx_, channel_, flow_id, RDMAFactory::CreateContext(dev, local_engine_idx_, meta));

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
        auto qp = rdma_ctx->qp_vec_[i];
        DCHECK(qp != nullptr);
        ctrl_work_rsp[i].meta.ToEndPoint.local_psn = rdma_ctx->local_psn_[i];
        ctrl_work_rsp[i].meta.ToEndPoint.local_qpn = qp->qp_num;
        ctrl_work_rsp[i].opcode = Channel::CtrlMsg::kCompleteFlowRDMA;
    }

    ctrl_work_rsp[kPortEntropy].meta.ToEndPoint.local_psn = rdma_ctx->ctrl_local_psn_;
    ctrl_work_rsp[kPortEntropy].meta.ToEndPoint.local_qpn = rdma_ctx->ctrl_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy].opcode =  Channel::CtrlMsg::kCompleteFlowRDMA;

    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.local_psn = rdma_ctx->retr_local_psn_;
    ctrl_work_rsp[kPortEntropy + 1].meta.ToEndPoint.local_qpn = rdma_ctx->retr_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy + 1].opcode =  Channel::CtrlMsg::kCompleteFlowRDMA;

    ctrl_work_rsp[kPortEntropy + 2].meta.ToEndPoint.local_psn = rdma_ctx->fifo_local_psn_;
    ctrl_work_rsp[kPortEntropy + 2].meta.ToEndPoint.local_qpn = rdma_ctx->fifo_qp_->qp_num;
    ctrl_work_rsp[kPortEntropy + 2].opcode =  Channel::CtrlMsg::kCompleteFlowRDMA;

    while (jring_mp_enqueue_bulk(channel_->ctrl_rspq_, ctrl_work_rsp, RDMAContext::kTotalQP, nullptr) != RDMAContext::kTotalQP) {
    }

}

std::string UcclRDMAEngine::status_to_string() {
    std::string s;
    for (auto [flow_id, flow] : active_flows_map_) {
        s += Format("\n\t\tEngine %d Flow 0x%lx: %s (%u) <-> %s (%u)",
                    local_engine_idx_, flow_id,
                    ip_to_str(htonl(flow->local_addr_)).c_str(),
                    flow->local_engine_idx_,
                    ip_to_str(htonl(flow->remote_addr_)).c_str(),
                    flow->remote_engine_idx_);
        s += flow->to_string();
    }

    return s;
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
        
        auto local_ip_str = rdma_dev_list_[engine_id % num_devices].local_ip_str;
        
        engine_vec_.emplace_back(std::make_unique<UcclRDMAEngine>(
            engine_id, channel_vec_[engine_id], local_ip_str));
        
        engine_th_vec_.emplace_back(std::make_unique<std::thread>(
            [engine_ptr = engine_vec_.back().get(), engine_id, engine_cpu_id]() {
                LOG(INFO) << "[Engine] thread " << engine_id
                          << " running on CPU " << engine_cpu_id;
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
    
    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd);

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

    install_flow_on_engine_rdma(dev, flow_id, remote_ip, local_engine_idx, bootstrap_fd);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

PollCtx *RDMAEndpoint::uccl_send_async(ConnID conn_id, const void *data,
                                   const size_t len) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kTx,
        .flow_id = conn_id.flow_id,
        .data = const_cast<void *>(data),
        .len = len,
        .len_p = nullptr,
        .poll_ctx = poll_ctx,
    };
    std::atomic_store_explicit(&poll_ctx->fence, true,
                               std::memory_order_release);
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->tx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

PollCtx *RDMAEndpoint::uccl_recv_async(ConnID conn_id, void *data, size_t len) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .data = data,
        .len = len,
        .poll_ctx = poll_ctx,
    };
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
                                      int bootstrap_fd) {
    int ret;
    struct RDMAExchangeFormatLocal meta = { 0 };
    // We use this pointer to fill meta data.
    auto *to_engine_meta = &meta.ToEngine;
    struct RDMAExchangeFormatRemote xchg_meta[RDMAContext::kTotalQP];

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
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "[Endpoint] meta.local_gid.raw:\t";
        for (int i = 0; i < 16; ++i) {
            oss << ((i == 0)? "" : ":") << static_cast<int>(factory_dev->gid.raw[i]);
        }
        VLOG(1) << oss.str();
    }
    
    if (FLAGS_v >= 1) {
        std::ostringstream oss;
        oss << "[Endpoint] meta.remote_gid.raw:\t";
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
        while (jring_sc_dequeue_bulk(channel_vec_[local_engine_idx]->ctrl_rspq_, &rsp_msg, 1, nullptr) != 1);
        if (rsp_msg.opcode != Channel::CtrlMsg::Op::kCompleteFlowRDMA) continue;
        xchg_meta[qidx].qpn = rsp_msg.meta.ToEndPoint.local_qpn;
        xchg_meta[qidx].psn = rsp_msg.meta.ToEndPoint.local_psn;
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
        LOG(INFO) << s;
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