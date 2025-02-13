# UCCL-RDMA

RDMA support for UCCL.

## Install dependencies
```
sudo apt-get install libibverbs-dev libpci-dev -y
```

## Install lastest perftest for EFA
```
pushd /tmp
git clone git@github.com:linux-rdma/perftest.git && cd perftest
./autogen.sh
./configure
make
sudo make install
popd
```