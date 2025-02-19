# UCCL-EFA

AWS EFA support for UCCL.

## Install dependencies
```
sudo apt-get install libibverbs-dev libpci-dev -y
```

## Install lastest perftest to benchmark EFA
```
pushd /tmp
git clone https://github.com/linux-rdma/perftest.git && cd perftest
./autogen.sh
./configure
make
sudo make install
popd
```

Throughput benchmark: 
```
ib_send_bw -d rdmap16s27 --report_gbits -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 4096 -m 8192 -F
ib_send_bw -d rdmap16s27 --report_gbits 172.31.42.140 -x 0 -c UD -t 128 -Q 1 -q 32 -l 2 -s 4096 -m 8192 -F
```

Latency benchmark: 
```
ib_send_lat -d rdmap16s27 --report_gbits -x 0 -c UD -F
ib_send_lat -d rdmap16s27 --report_gbits 172.31.42.140 -x 0 -c UD -F
```
