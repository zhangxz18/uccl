#include "transport.h"

namespace uccl {

std::string UcclFlow::to_string() const {
    std::string s;
    s += "\n\t\t\t" + pcb_.to_string();
    return s;
}

void UcclFlow::rx_messages() {

}

void UcclFlow::rx_supply_app_buf(Channel::Msg &rx_work) {

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

void UcclRdmaEngine::run() {
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
void UcclRdmaEngine::periodic_process() {
    // Advance the periodic ticks counter.
    periodic_ticks_++;
    process_ctl_reqs();
}

void UcclRdmaEngine::handle_rto() {
    for (auto [flow_id, flow] : active_flows_map_) {
        auto is_active_flow = flow->periodic_check();
        DCHECK(is_active_flow);
    }
}

void UcclRdmaEngine::process_ctl_reqs() {
    Channel::CtrlMsg ctrl_work;
    if (jring_sc_dequeue_bulk(channel_->ctrl_cmdq_, &ctrl_work, 1, nullptr) ==
        1) {
        switch (ctrl_work.opcode) {
            case Channel::CtrlMsg::kInstallFlow:
                    handle_install_flow_on_engine_rdma(ctrl_work);
                break;
            default:
                break;
        }
    }
}

void UcclRdmaEngine::handle_install_flow_on_engine_rdma(Channel::CtrlMsg &ctrl_work)
{

}

std::string UcclRdmaEngine::status_to_string() {
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

RdmaEndpoint::RdmaEndpoint(const char *infiniband_name, int num_engines, int engine_cpu_start)
    : num_engines_(num_engines), engine_cpu_start_(engine_cpu_start), stats_thread_([this]() { stats_thread_fn(); }) {
    
    char ethernet_name[64];
    DCHECK(util_rdma_ib2eth_name(infiniband_name, ethernet_name) == 0) << "Failed to convert IB name to Ethernet name";
    
    static std::once_flag flag_once;
    std::call_once(flag_once, [infiniband_name, num_engines]() {
        RDMAFactory::init(infiniband_name, num_engines);
    });

    local_ip_str_ = get_dev_ip(ethernet_name);
    local_mac_str_ = get_dev_mac(ethernet_name);

    CHECK_LE(num_engines, NUM_CPUS / 4)
        << "num_engines should be less than or equal to the number of CPUs / 4";

    // Create multiple engines. Each engine has its own thread and channel to let the endpoint communicate with.
    for (int i = 0; i < num_engines; i++) channel_vec_[i] = new Channel();

    for (int engine_id = 0, engine_cpu_id = engine_cpu_start;
         engine_id < num_engines; engine_id++, engine_cpu_id++) {
        
        engine_vec_.emplace_back(std::make_unique<UcclRdmaEngine>(
            engine_id, channel_vec_[engine_id], local_ip_str_, local_mac_str_));
        
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
    LOG(INFO) << "[RdmaEndpoint] server ready, listening on port "
              << kBootstrapPort;
}

RdmaEndpoint::~RdmaEndpoint() {
    for (auto &engine : engine_vec_) engine->shutdown();
    for (auto &engine_th : engine_th_vec_) engine_th->join();
    for (int i = 0; i < num_engines_; i++) delete channel_vec_[i];

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

ConnID RdmaEndpoint::uccl_connect(std::string remote_ip) {
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
    localaddr.sin_addr.s_addr = str_to_ip(local_ip_str_.c_str());
    bind(bootstrap_fd, (sockaddr *)&localaddr, sizeof(localaddr));

    LOG(INFO) << "[RdmaEndpoint] connecting to " << remote_ip << ":"
              << kBootstrapPort;

    // Connect and set nonblocking and nodelay
    while (connect(bootstrap_fd, (struct sockaddr *)&serv_addr,
                   sizeof(serv_addr))) {
        LOG(INFO) << "[RdmaEndpoint] connecting... Make sure the server is up.";
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
        LOG(INFO) << "[RdmaEndpoint] connect: receive proposed FlowID: " << std::hex
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
    
    install_flow_on_engine_rdma(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

ConnID RdmaEndpoint::uccl_accept(std::string &remote_ip) {
    struct sockaddr_in cli_addr;
    socklen_t clilen = sizeof(cli_addr);
    int bootstrap_fd;

    // Accept connection and set nonblocking and nodelay
    bootstrap_fd = accept(listen_fd_, (struct sockaddr *)&cli_addr, &clilen);
    DCHECK(bootstrap_fd >= 0);
    remote_ip = ip_to_str(cli_addr.sin_addr.s_addr);

    LOG(INFO) << "[RdmaEndpoint] accept from " << remote_ip << ":"
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

        LOG(INFO) << "[RdmaEndpoint] accept: propose FlowID: " << std::hex << "0x"
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

    install_flow_on_engine_rdma(flow_id, remote_ip, local_engine_idx, bootstrap_fd);

    return ConnID{.flow_id = flow_id,
                  .engine_idx = (uint32_t)local_engine_idx,
                  .boostrap_id = bootstrap_fd};
}

bool RdmaEndpoint::uccl_send(ConnID conn_id, const void *data, const size_t len,
                         bool busypoll) {
    auto *poll_ctx = uccl_send_async(conn_id, data, len);
    return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

bool RdmaEndpoint::uccl_recv(ConnID conn_id, void *data, size_t *len_p,
                         bool busypoll) {
    auto *poll_ctx = uccl_recv_async(conn_id, data, len_p);
    return busypoll ? uccl_poll(poll_ctx) : uccl_wait(poll_ctx);
}

PollCtx *RdmaEndpoint::uccl_send_async(ConnID conn_id, const void *data,
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

PollCtx *RdmaEndpoint::uccl_recv_async(ConnID conn_id, void *data, size_t *len_p) {
    auto *poll_ctx = ctx_pool_->pop();
    Channel::Msg msg = {
        .opcode = Channel::Msg::Op::kRx,
        .flow_id = conn_id.flow_id,
        .data = data,
        .len = 0,
        .len_p = len_p,
        .poll_ctx = poll_ctx,
    };
    while (jring_mp_enqueue_bulk(channel_vec_[conn_id.engine_idx]->rx_cmdq_,
                                 &msg, 1, nullptr) != 1);
    return poll_ctx;
}

bool RdmaEndpoint::uccl_wait(PollCtx *ctx) {
    {
        std::unique_lock<std::mutex> lock(ctx->mu);
        ctx->cv.wait(lock, [&ctx] { return ctx->done.load(); });
    }
    fence_and_clean_ctx(ctx);
    return true;
}

bool RdmaEndpoint::uccl_poll(PollCtx *ctx) {
    while (!uccl_poll_once(ctx));
    return true;
}

bool RdmaEndpoint::uccl_poll_once(PollCtx *ctx) {
    if (!ctx->done.load()) return false;
    fence_and_clean_ctx(ctx);
    return true;
}

void RdmaEndpoint::install_flow_on_engine_rdma(FlowID flow_id,
                                      const std::string &remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd) {

}

void RdmaEndpoint::install_flow_on_engine(FlowID flow_id,
                                      const std::string &remote_ip,
                                      uint32_t local_engine_idx,
                                      int bootstrap_fd) {
    int ret;

    char local_mac_char[ETH_ALEN];
    std::string local_mac = local_mac_str_;
    VLOG(3) << "[RdmaEndpoint] local MAC: " << local_mac;
    str_to_mac(local_mac, local_mac_char);
    ret = send_message(bootstrap_fd, local_mac_char, ETH_ALEN);
    DCHECK(ret == ETH_ALEN);

    char remote_mac_char[ETH_ALEN];
    ret = receive_message(bootstrap_fd, remote_mac_char, ETH_ALEN);
    DCHECK(ret == ETH_ALEN);
    std::string remote_mac = mac_to_str(remote_mac_char);
    VLOG(3) << "[RdmaEndpoint] remote MAC: " << remote_mac;

    // Sync remote engine index.
    uint32_t remote_engine_idx;
    ret = send_message(bootstrap_fd, &local_engine_idx, sizeof(uint32_t));
    ret = receive_message(bootstrap_fd, &remote_engine_idx, sizeof(uint32_t));
    DCHECK(ret == sizeof(uint32_t));

    // Install flow and dst ports on engine.
    auto *poll_ctx = new PollCtx();
    Channel::CtrlMsg ctrl_msg = {
        .opcode = Channel::CtrlMsg::Op::kInstallFlow,
        .flow_id = flow_id,
        .remote_ip = htonl(str_to_ip(remote_ip)),
        .remote_engine_idx = remote_engine_idx,
        .poll_ctx = poll_ctx,
    };
    str_to_mac(remote_mac, ctrl_msg.remote_mac);
    while (jring_mp_enqueue_bulk(channel_vec_[local_engine_idx]->ctrl_cmdq_,
                                 &ctrl_msg, 1, nullptr) != 1);

    // Wait until the flow has been installed on the engine.
    {
        std::unique_lock<std::mutex> lock(poll_ctx->mu);
        poll_ctx->cv.wait(lock, [poll_ctx] { return poll_ctx->done.load(); });
    }
    delete poll_ctx;

    // sync so to receive flow_id packets.
    net_barrier(bootstrap_fd);
}

inline int RdmaEndpoint::find_least_loaded_engine_idx_and_update() {
    std::lock_guard<std::mutex> lock(engine_load_vec_mu_);
    if (engine_load_vec_.empty()) return -1;  // Handle empty vector case

    auto minElementIter =
        std::min_element(engine_load_vec_.begin(), engine_load_vec_.end());
    *minElementIter += 1;
    return std::distance(engine_load_vec_.begin(), minElementIter);
}

inline void RdmaEndpoint::fence_and_clean_ctx(PollCtx *ctx) {
    // Make the data written by the engine thread visible to the app thread.
    std::ignore =
        std::atomic_load_explicit(&ctx->fence, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    ctx->clear();
    ctx_pool_->push(ctx);
}

void RdmaEndpoint::stats_thread_fn() {
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

}  // namespace uccl