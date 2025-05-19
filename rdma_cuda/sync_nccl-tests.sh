#!/bin/bash

SOURCE_DIR="/home/aleria/uccl_rdma/nccl-tests"

TARGET_MACHINES=("87.120.213.6" "87.120.213.5")

TARGET_DIR="/home/aleria/uccl_rdma/"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo "Installing on machine: $MACHINE"

    rsync -avz --delete "$SOURCE_DIR/" "aleria@$MACHINE:$TARGET_DIR/nccl-tests" > /dev/null 2>&1
  ) &
done

wait

echo "Done."