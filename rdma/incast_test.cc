#include <chrono>
#include <thread>
#include <numeric>
#include <algorithm>
#include <fstream>

#include <signal.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "transport.h"
#include "transport_config.h"
#include "util_timer.h"

using namespace uccl;

// #define VERBOSE

static volatile bool quit = false;

std::optional<RDMAEndpoint> ep;

void interrupt_handler(int signal) {
    (void)signal;
    quit = true;
}

#define INCAST_DEV 0
#define NUM_RTT_SAMPLE 1000000
#define NUM_WARMUP (NUM_RTT_SAMPLE / 2)

DEFINE_bool(server, false, "Whether this is a server receiving traffic.");
DEFINE_string(serverip, "", "Server IP address the client tries to connect.");
DEFINE_uint32(nflow, 1, "Number of flows.");
DEFINE_uint32(nmsg, 1, "Number of messages within one request to post.");
DEFINE_uint32(nreq, 8, "Outstanding requests to post.");
DEFINE_uint32(msize, 56 << 10, "Size of message.");
DEFINE_uint32(iterations, 10000000, "Number of iterations to run.");
DEFINE_uint32(clients_per_nic, 1, "Number of clients(threads) per NIC.");
DEFINE_uint32(server_threads_per_nic, 5, "Number of server threads per NIC.");
DEFINE_bool(flush, false, "Whether to flush after receiving.");
DEFINE_bool(bi, false, "Whether to run bidirectional test.");

std::atomic<uint64_t>tx_cur_sec_bytes = 0;
uint64_t tx_prev_sec_bytes = 0;
std::atomic<uint64_t>rx_cur_sec_bytes = 0;
uint64_t rx_prev_sec_bytes = 0;

thread_local uint64_t c_itr = 0;
thread_local uint64_t s_itr = 0;

std::thread client_threads[8][1024];
std::thread server_threads[1024];

std::vector<uint64_t> rtts; 

std::atomic<uint32_t> ready_client;

static void server_tpt(std::vector<ConnID> &conn_ids,
                       std::vector<struct Mhandle *> &mhandles,
                       std::vector<void *> &datas, int nflows) {
    s_itr = FLAGS_iterations;
    s_itr *= nflows;

    int len[nflows][FLAGS_nreq][FLAGS_nmsg];
    struct Mhandle *mhs[nflows][FLAGS_nreq][FLAGS_nmsg];
    void *recv_data[nflows][FLAGS_nreq][FLAGS_nmsg];

    int flag[nflows][FLAGS_nreq];
    memset(flag, 0, sizeof(int) * nflows * FLAGS_nreq);

    std::vector<std::vector<ucclRequest>> ureq_vec(nflows);
    for (int i = 0; i < nflows; i++) {
        ureq_vec[i].resize(FLAGS_nreq);
    }

    for (int f = 0; f < nflows; f++) {
        for (int r = 0; r < FLAGS_nreq; r++)
            for (int m = 0; m < FLAGS_nmsg; m++) {
                len[f][r][m] = FLAGS_msize;
                recv_data[f][r][m] = reinterpret_cast<char *>(datas[f]) +
                                     r * FLAGS_msize * FLAGS_nmsg +
                                     m * FLAGS_msize;
                mhs[f][r][m] = mhandles[f];
            }
    }

    for (int f = 0; f < nflows; f++) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            DCHECK(ep->uccl_recv_async((UcclFlow *)conn_ids[f].context,
                                       mhs[f][r], recv_data[f][r], len[f][r],
                                       FLAGS_nmsg, &ureq_vec[f][r]) == 0);
            s_itr--;
            rx_cur_sec_bytes.fetch_add(FLAGS_msize * FLAGS_nmsg);
        }
    }

    while (s_itr && !quit) {
        for (int f = 0; f < nflows; f++) {
            for (int r = 0; r < FLAGS_nreq; r++) {
                if (quit) {
                    s_itr = 0;
                    break;
                }
                if (!ep->uccl_poll_ureq_once(&ureq_vec[f][r]))
                    continue;

                if (!FLAGS_flush) {
                    s_itr--;
                    if (s_itr == 0)
                        break;
                    DCHECK(ep->uccl_recv_async((UcclFlow *)conn_ids[f].context,
                                               mhs[f][r], recv_data[f][r],
                                               len[f][r], FLAGS_nmsg,
                                               &ureq_vec[f][r]) == 0);
                    rx_cur_sec_bytes.fetch_add(FLAGS_msize * FLAGS_nmsg);
                    continue;
                }

                if (flag[f][r] == 0) {
                    DCHECK(ep->uccl_flush((UcclFlow *)conn_ids[f].context,
                                          mhs[f][r], recv_data[f][r], len[f][r],
                                          FLAGS_nmsg, &ureq_vec[f][r]) == 0);
                    flag[f][r] = 1;
                } else if (flag[f][r] == 1) {
                    s_itr--;
                    if (s_itr == 0)
                        break;
                    DCHECK(ep->uccl_recv_async((UcclFlow *)conn_ids[f].context,
                                               mhs[f][r], recv_data[f][r],
                                               len[f][r], FLAGS_nmsg,
                                               &ureq_vec[f][r]) == 0);
                    rx_cur_sec_bytes.fetch_add(FLAGS_msize * FLAGS_nmsg);
                    flag[f][r] = 0;
                }
            }
        }
    }
}

static void client_tpt(std::vector<ConnID> &conn_ids,
                       std::vector<struct Mhandle *> &mhandles,
                       std::vector<void *> &datas, bool dump) {
    c_itr = FLAGS_iterations;
    c_itr *= FLAGS_nflow;

    std::vector<std::vector<std::vector<struct ucclRequest>>> ureq_vec(
        FLAGS_nflow);
    for (int f = 0; f < FLAGS_nflow; f++) {
        ureq_vec[f].resize(FLAGS_nreq);
        for (int r = 0; r < FLAGS_nreq; r++) {
            ureq_vec[f][r].resize(FLAGS_nmsg);
            for (int n = 0; n < FLAGS_nmsg; n++)
                ureq_vec[f][r][n] = {};
        }
    }
    for (int f = 0; f < FLAGS_nflow; f++) {
        for (int r = 0; r < FLAGS_nreq; r++) {
            for (int n = 0; n < FLAGS_nmsg; n++) {
                void *send_data = reinterpret_cast<char *>(datas[f]) +
                                  r * FLAGS_msize * FLAGS_nmsg +
                                  n * FLAGS_msize;
                while (ep->uccl_send_async((UcclFlow *)conn_ids[f].context,
                                           mhandles[f], send_data, FLAGS_msize,
                                           &ureq_vec[f][r][n]) && !quit) {
                }
                ureq_vec[f][r][n].rtt_tsc = rdtsc();
                tx_cur_sec_bytes.fetch_add(FLAGS_msize);
            }
            c_itr--;
        }
    }

    int fin_msg = 0;

    while (c_itr && !quit) {
        for (int f = 0; f < FLAGS_nflow; f++) {
            for (int r = 0; r < FLAGS_nreq; r++) {
                for (int n = 0; n < FLAGS_nmsg; n++) {
                    if (ureq_vec[f][r][n].rtt_tsc == 0 || ep->uccl_poll_ureq_once(&ureq_vec[f][r][n])) {
                        if (dump && rtts.size() < NUM_RTT_SAMPLE && ureq_vec[f][r][n].rtt_tsc) {
                            rtts.push_back(rdtsc() - ureq_vec[f][r][n].rtt_tsc);
                        }

                        ureq_vec[f][r][n].rtt_tsc = 0;

                        void *send_data = reinterpret_cast<char *>(datas[f]) +
                                          r * FLAGS_msize * FLAGS_nmsg +
                                          n * FLAGS_msize;
                        if (ep->uccl_send_async(
                                            (UcclFlow *)conn_ids[f].context,
                                            mhandles[f], send_data, FLAGS_msize,
                                            &ureq_vec[f][r][n])) {
                                                continue;
                                            }
                        ureq_vec[f][r][n].rtt_tsc = rdtsc();
                        tx_cur_sec_bytes.fetch_add(FLAGS_msize);
                        if (++fin_msg == FLAGS_nreq) {
                            c_itr--;
                            fin_msg = 0;
                        }
                    }
                    if (quit) {
                        c_itr = 0;
                        break;
                    }
                }
            }
        }
    }
}

static void server_worker(int dev) {
    std::string remote_ip;

    std::vector<ConnID> conn_ids;
    std::vector<void *> datas;
    std::vector<struct Mhandle *> mhandles;

    int nflows = (NUM_DEVICES + NUM_DEVICES - 1) * FLAGS_clients_per_nic * FLAGS_nflow / FLAGS_server_threads_per_nic;

    mhandles.resize(nflows);

    for (int i = 0; i < nflows; i++) {
        int remote_dev;
        auto conn_id = ep->test_uccl_accept(dev, remote_ip, &remote_dev);
        #ifdef VERBOSE
        printf("Server accepted connection from %s (flow#%d), local/remote dev:%d/%d\n",
               remote_ip.c_str(), i, dev, remote_dev);
        #endif
        void *data =
            mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg,
                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(data != MAP_FAILED);
        ep->uccl_regmr((UcclFlow *)conn_id.context, data,
                       FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

        conn_ids.push_back(conn_id);
        datas.push_back(data);
    }

    server_tpt(conn_ids, mhandles, datas, nflows);

    while (!quit) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int i = 0; i < nflows; i++) {
        ep->uccl_deregmr(mhandles[i]);
        munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
    }
}

static void client_worker(int dev, int remote_dev, bool dump) {
    std::vector<ConnID> conn_ids;
    std::vector<void *> datas;
    std::vector<struct Mhandle *> mhandles;

    mhandles.resize(FLAGS_nflow);

    for (int i = 0; i < FLAGS_nflow; i++) {
        auto conn_id = ep->test_uccl_connect(dev, FLAGS_serverip, remote_dev);
        #ifdef VERBOSE
        printf("Client connected to %s, local/remote dev: %d/%d\n", FLAGS_serverip.c_str(), dev, remote_dev);
        #endif
        void *data =
            mmap(nullptr, FLAGS_msize * FLAGS_nreq * FLAGS_nmsg,
                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(data != MAP_FAILED);
        ep->uccl_regmr((UcclFlow *)conn_id.context, data,
                       FLAGS_msize * FLAGS_nreq * FLAGS_nmsg, 0, &mhandles[i]);

        conn_ids.push_back(conn_id);
        datas.push_back(data);
    }

    ready_client.fetch_add(1);

    while (!quit) {
        if (FLAGS_server == false) {
            if (ready_client.load() == NUM_DEVICES * FLAGS_clients_per_nic)
                break;
        } else {
            if (ready_client.load() == (NUM_DEVICES - 1) * FLAGS_clients_per_nic)
            break;
        }
    }
    
    client_tpt(conn_ids, mhandles, datas, dump);

    for (int i = 0; i < FLAGS_nflow; i++) {
        ep->uccl_deregmr(mhandles[i]);
        munmap(datas[i], FLAGS_msize * FLAGS_nreq * FLAGS_nmsg);
    }
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    signal(SIGINT, interrupt_handler);
    signal(SIGTERM, interrupt_handler);
    signal(SIGHUP, interrupt_handler);

    ep.emplace(DEVNAME_SUFFIX_LIST, NUM_DEVICES, NUM_ENGINES, ENGINE_CPU_START);

    // Create a thread to print throughput every second
    bool rtt_dump = false;
    std::thread t([&] {
        while (!quit) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            printf(
                "(%d flows) TX Tpt: %.2f Gbps(%lu), RX Tpt: %.2f Gbps(%lu), %ld\n",
                FLAGS_nflow * (NUM_DEVICES + NUM_DEVICES - 1) * FLAGS_clients_per_nic,
                (tx_cur_sec_bytes.load() - tx_prev_sec_bytes) * 8.0 / 1000 / 1000 /
                    1000,
                c_itr,
                (rx_cur_sec_bytes - rx_prev_sec_bytes) * 8.0 / 1000 / 1000 /
                    1000,
                s_itr, rtts.size());

            tx_prev_sec_bytes = tx_cur_sec_bytes.load();
            rx_prev_sec_bytes = rx_cur_sec_bytes.load();

            if (rtts.size() == NUM_RTT_SAMPLE && !rtt_dump) {

                rtt_dump = true;

                rtts.erase(rtts.begin(), rtts.begin() + NUM_WARMUP);
                
                for (int i = 0; i < NUM_RTT_SAMPLE - NUM_WARMUP; i++) {
                    rtts[i] = to_usec(rtts[i], freq_ghz);
                }

                uint64_t sum = std::accumulate(rtts.begin(), rtts.end(), 0ULL);
                double average = static_cast<double>(sum) / rtts.size();

                std::sort(rtts.begin(), rtts.end());

                auto percentile = [&](double p) -> uint64_t {
                    if (p < 0.0 || p > 1.0) return 0;
                    size_t idx = static_cast<size_t>(p * (rtts.size() - 1) + 0.5);
                    return rtts[idx];
                };

                uint64_t p50 = percentile(0.50);
                uint64_t p90 = percentile(0.90);
                uint64_t p99 = percentile(0.99);
                uint64_t p99_9 = percentile(0.999);

                std::cout << "Average:\t" << average << "us" << std::endl;
                std::cout << "P50:\t\t" << p50 << "us" << std::endl;
                std::cout << "P90:\t\t" << p90 << "us" << std::endl;
                std::cout << "P99:\t\t" << p99 << "us" << std::endl;
                std::cout << "P99.9:\t\t" << p99_9 << "us" << std::endl;


                std::ofstream outFile("log.txt");
                if (!outFile.is_open()) {
                    return 1;
                }
                
                for (const auto& rtt : rtts) {
                    outFile << rtt << '\n';
                }
                
                outFile.close();
                std::cout << "Write file done." << std::endl;
            }
        }
        return 0;
    });

    if (FLAGS_bi) {
        CHECK(!FLAGS_serverip.empty()) << "Server IP address is required.";
        std::thread server_thread(server_worker, 0);
        std::thread client_thread(client_worker, 0, 0, false);

        server_thread.join();
        client_thread.join();
    } else if (FLAGS_server) {

        for (int t = 0; t < FLAGS_server_threads_per_nic; t++) {
            server_threads[t] = std::thread(server_worker, INCAST_DEV);
        }

        for (int dev = 1; dev < NUM_DEVICES; dev++) {
            for (int t = 0; t < FLAGS_clients_per_nic; t++) {
                client_threads[dev][t] = std::thread(client_worker, dev, INCAST_DEV, false);
            }
        }

        for (int dev = 1; dev < NUM_DEVICES; dev++) {
            for (int t = 0; t < FLAGS_clients_per_nic; t++) {
                client_threads[dev][t].join();
            }
        }

        for (int t = 0; t < FLAGS_server_threads_per_nic; t++) {
            server_threads[t].join();
        }

    } else {
        CHECK(!FLAGS_serverip.empty()) << "Server IP address is required.";

        std::cout << "Incast ratio: " << NUM_DEVICES * FLAGS_clients_per_nic << "-1." << std::endl;

        for (int dev = 0; dev < NUM_DEVICES; dev++) {
            for (int t = 0; t < FLAGS_clients_per_nic; t++) {
                client_threads[dev][t] = std::thread(client_worker, dev, INCAST_DEV, dev == 0 && t == 0);
            }
        }

        for (int dev = 0; dev < NUM_DEVICES; dev++) {
            for (int t = 0; t < FLAGS_clients_per_nic; t++) {
                client_threads[dev][t].join();
            }
        }
    }

    t.join();

    return 0;
}
