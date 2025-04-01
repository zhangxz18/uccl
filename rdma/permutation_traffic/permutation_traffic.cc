#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <signal.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <vector>
#include <optional>
#include <thread>
#include <fstream>
#include <string>

#include "transport.h"
#include "transport_config.h"
#include "util_timer.h"

#define MPI_LOG(level) LOG(level) << "Rank:" << LOCAL_RANK << " "

#define MAX_BUFFER_SIZE (16 * 1024 * 1024)     // 16MB
#define NET_CHUNK_SIZE  (512 * 1024)           // 512KB
#define MAX_CHUNK (MAX_BUFFER_SIZE / NET_CHUNK_SIZE)

DEFINE_uint32(size, 4 * 1024 * 1024, "Message size.");
DEFINE_uint32(iterations, 1000000, "Number of iterations to run.");
DEFINE_string(benchtype, "SA", "Benchmark type. PT: Permutation Traffic, SA: Sequential AlltoAll, AA: AlltoAll");

using namespace uccl;

static constexpr uint32_t DEV = 0;
static constexpr uint32_t REMOTE_DEV = 0;

static int LOCAL_RANK;
static int NRANKS;

struct CommHandle {
    ConnID conn_id;
    void *buffer = nullptr;
    struct Mhandle *mhandle = nullptr;
    int nb_net_chunk = 0;
    struct ucclRequest ureq[MAX_CHUNK] = {};
};
std::optional<RDMAEndpoint> ep;

std::vector<std::string> ips;
class NodeInfo;
std::vector<NodeInfo> nodes;

std::thread stats_thread;
std::atomic<uint64_t>tx_cur_sec_bytes = 0;
uint64_t tx_prev_sec_bytes = 0;
std::atomic<uint64_t>rx_cur_sec_bytes = 0;
uint64_t rx_prev_sec_bytes = 0;

static volatile bool quit = false;

static volatile uint32_t cur_iteration = 0;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

static void server_setup_agent(int target_rank, std::string ip, struct CommHandle *recv_comm)
{
    int remote_dev;
    auto conn_id = ep->test_uccl_accept(DEV, ip, &remote_dev);
    
    recv_comm->conn_id = conn_id;

    MPI_LOG(INFO) << "Accepted from " << target_rank << " succesfully";
}

static void client_setup_agent(int target_rank, std::string ip, struct CommHandle *send_comm)
{
    auto conn_id = ep->test_uccl_connect(DEV, ip, REMOTE_DEV);
    
    send_comm->conn_id = conn_id;

    MPI_LOG(INFO) << "Connected to " << target_rank << " succesfully";
}

class NodeInfo {
public:
    NodeInfo(std::string &ip, int target_rank): ip_(ip), target_rank_(target_rank) {}
    ~NodeInfo() = default;

    void server_setup() {server_thread_ = std::thread(server_setup_agent, target_rank_, ip_, &recv_comm_);}

    void client_setup() {client_thread_ = std::thread(client_setup_agent, target_rank_, ip_, &send_comm_);}

    void wait_connect_done() {client_thread_.join();}

    void wait_accept_done() {server_thread_.join();}

    NodeInfo(const NodeInfo&) = delete;
    NodeInfo& operator=(const NodeInfo&) = delete;

    NodeInfo(NodeInfo&&) = default;
    NodeInfo& operator=(NodeInfo&&) = default;

    // Target rank.
    int target_rank_;
    // IP address.
    std::string ip_;
    // Communication handle for sending.
    struct CommHandle send_comm_;
    // Communication handle for receiving.
    struct CommHandle recv_comm_;
    // Thread for client stuff.
    std::thread client_thread_;
    // Thread for server stuff.
    std::thread server_thread_;
};

void setup_connections()
{
    for (int r = 0; r < NRANKS; r++) {
        if (r == LOCAL_RANK) continue;
        nodes[r].client_setup();
        nodes[r].server_setup();
    }
    MPI_LOG(INFO) << "Start to setup connections.";
}

void wait_setup_connections()
{

    for (int r = 0; r < NRANKS; r++) {
        if (r == LOCAL_RANK) continue;
        nodes[r].wait_connect_done();
        nodes[r].wait_accept_done();
    }
    MPI_LOG(INFO) << "Connections setup done.";
}

void allocate_buffers()
{
    for (int r = 0; r < NRANKS; r++) {
        if (r == LOCAL_RANK) continue;

        nodes[r].send_comm_.buffer = mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        DCHECK(nodes[r].send_comm_.buffer != MAP_FAILED);

        ep->uccl_regmr((UcclFlow *)nodes[r].send_comm_.conn_id.context, nodes[r].send_comm_.buffer,
        MAX_BUFFER_SIZE, 0, &nodes[r].send_comm_.mhandle);

        nodes[r].recv_comm_.buffer = mmap(nullptr, MAX_BUFFER_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        DCHECK(nodes[r].recv_comm_.buffer != MAP_FAILED);

        ep->uccl_regmr((UcclFlow *)nodes[r].recv_comm_.conn_id.context, nodes[r].recv_comm_.buffer,
        MAX_BUFFER_SIZE, 0, &nodes[r].recv_comm_.mhandle);
    }
}

void free_buffers()
{
    for (int r = 0; r < NRANKS; r++) {
        if (r == LOCAL_RANK) continue;

        ep->uccl_deregmr(nodes[r].send_comm_.mhandle);
        munmap(nodes[r].send_comm_.buffer, MAX_BUFFER_SIZE);

        ep->uccl_deregmr(nodes[r].recv_comm_.mhandle);
        munmap(nodes[r].recv_comm_.buffer, MAX_BUFFER_SIZE);
    }
}

static void init_benchmark()
{
    ep.emplace(DEVNAME_SUFFIX_LIST, NUM_DEVICES, NUM_ENGINES);

    setup_connections();

    wait_setup_connections();

    allocate_buffers();
}

static void exit_benchmark()
{
    free_buffers();

    stats_thread.join();
}

bool test_p2p_send_done(int target_rank, int chunk_id)
{
    auto *send_comm_ = &nodes[target_rank].send_comm_;
    bool done =  ep->uccl_poll_ureq_once(&send_comm_->ureq[chunk_id]);
    if (done)
        tx_cur_sec_bytes.fetch_add(send_comm_->ureq[chunk_id].send.data_len);
    return done;
}

bool test_p2p_recv_done(int target_rank, int chunk_id)
{
    auto *recv_comm_ = &nodes[target_rank].recv_comm_;
    bool done = ep->uccl_poll_ureq_once(&recv_comm_->ureq[chunk_id]);
    if (done)
        rx_cur_sec_bytes.fetch_add(recv_comm_->ureq[chunk_id].recv.data_len[0]);
    return done;
}

void p2p_send(int target_rank, int size)
{
    MPI_LOG(INFO) << "p2p send to " << target_rank;
    
    auto *send_comm_ = &nodes[target_rank].send_comm_;

    uint32_t offset = 0;
    uint32_t chunk_id = 0;

    while (offset < size) {
        int net_chunk_size = std::min(size - offset, (uint32_t)NET_CHUNK_SIZE);
        while (ep->uccl_send_async((UcclFlow *)send_comm_->conn_id.context, 
        send_comm_->mhandle, send_comm_->buffer, net_chunk_size, &send_comm_->ureq[chunk_id])) {}

        offset += net_chunk_size;
        chunk_id++;
    }

    send_comm_->nb_net_chunk = chunk_id;
}

void p2p_receive(int target_rank, int size)
{
    MPI_LOG(INFO) << "p2p receive from " << target_rank;

    auto *recv_comm_ = &nodes[target_rank].recv_comm_;

    uint32_t offset = 0;
    uint32_t chunk_id = 0;

    while (offset < size) {
        int net_chunk_size = std::min(size - offset, (uint32_t)NET_CHUNK_SIZE);
        DCHECK(ep->uccl_recv_async((UcclFlow *)recv_comm_->conn_id.context, 
        &recv_comm_->mhandle, &recv_comm_->buffer, &net_chunk_size, 1, &recv_comm_->ureq[chunk_id]) == 0);

        offset += net_chunk_size;
        chunk_id++;
    }

    recv_comm_->nb_net_chunk = chunk_id;
}

void test_all_send(int target_rank)
{
    auto nb_net_chunk = nodes[target_rank].send_comm_.nb_net_chunk;
    for (int i = 0; i < nb_net_chunk; i++) {
        while (!test_p2p_send_done(target_rank, i)) {}
    }
}

void test_all_recv(int target_rank)
{
    auto nb_net_chunk = nodes[target_rank].recv_comm_.nb_net_chunk;
    for (int i = 0; i < nb_net_chunk; i++) {
        while (!test_p2p_recv_done(target_rank, i)) {}
    }
}

void net_sync(int target_rank)
{
    test_all_send(target_rank);
    test_all_recv(target_rank);
}

void readIPsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    int rank = 0;
    while (std::getline(file, line)) {
        size_t space_pos = line.find_first_of(" \t");
        if (space_pos != std::string::npos) {
            std::string ip = line.substr(0, space_pos);
            ips.push_back(ip);
        } else {
            ips.push_back(line);
        }
        rank++;
    }
    
    file.close();
}

void launch_stats_thread()
{
    stats_thread = std::thread ([&] {
        while (!quit) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            printf(
                "(%d)TX Tput: %.4f Gbps, RX Tput: %.4f Gbps\n",
                cur_iteration,
                (tx_cur_sec_bytes.load() - tx_prev_sec_bytes) * 8.0 / 1e9,
                (rx_cur_sec_bytes.load() - rx_prev_sec_bytes) * 8.0 / 1e9);

            tx_prev_sec_bytes = tx_cur_sec_bytes.load();
            rx_prev_sec_bytes = rx_cur_sec_bytes.load();
        }
        return 0;
    });
}

void verify_params() {
    CHECK(FLAGS_size <= MAX_BUFFER_SIZE);
}

void seq_alltoall()
{
    while (cur_iteration++ < FLAGS_iterations && !quit) {
        for (int r = 0; r < NRANKS; r++) {
            int target_rank = (LOCAL_RANK + r) % NRANKS;
            if (target_rank == LOCAL_RANK) continue;
            p2p_receive(target_rank, FLAGS_size);
            p2p_send(target_rank, FLAGS_size);
            net_sync(target_rank);
        }
    }

    MPI_LOG(INFO) << "Sequential alltoall done.";
}

void alltoall()
{
    while (cur_iteration++ < FLAGS_iterations && !quit) {
        for (int r = 0; r < NRANKS; r++) {
            if (r == LOCAL_RANK) continue;
            p2p_receive(r, FLAGS_size);
            p2p_send(r, FLAGS_size);
            net_sync(r);
        }
    }

    MPI_LOG(INFO) << "Alltoall done.";
}

int find_target_rank(const std::string& filePath, int sourceNode) {
    std::ifstream file(filePath);
    std::string line;
    
    while (std::getline(file, line)) {
        size_t arrowPos = line.find("->");
        if (arrowPos == std::string::npos) continue;
        
        int from = std::stoi(line.substr(0, arrowPos));
        if (from != sourceNode) continue;
        
        return std::stoi(line.substr(arrowPos + 2));
    }
    
    return -1;
}

void permutation_traffic()
{
    int target_rank = find_target_rank("matrix.txt", LOCAL_RANK);

    while (cur_iteration++ < FLAGS_iterations && !quit) {
        p2p_receive(target_rank, FLAGS_size);
        p2p_send(target_rank, FLAGS_size);
        net_sync(target_rank);
    }

    MPI_LOG(INFO) << "Permutation traffic done.";
}

int main(int argc, char** argv) {

    google::InitGoogleLogging(argv[0]);

    signal(SIGINT, interrupt_handler);

    verify_params();

    readIPsFromFile("hostname.txt");

    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &LOCAL_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NRANKS);

    for (int i = 0; i < NRANKS; i++)
        nodes.push_back(NodeInfo(ips[i], i));

    // Initialize connections, allocate buffers, etc.
    init_benchmark();

    // Wait until all nodes are ready.
    MPI_Barrier(MPI_COMM_WORLD);

    // Launch stats thread.
    launch_stats_thread();

    // Run benchmark.
    if (FLAGS_benchtype == "SA")
        seq_alltoall();
    else if (FLAGS_benchtype == "PT")
        permutation_traffic();
    else if (FLAGS_benchtype == "AA")
        alltoall();

    // Destroy connections, free buffers, etc.
    exit_benchmark();
    
    MPI_Finalize();
    return 0;
}