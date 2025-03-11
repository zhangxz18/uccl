#!/bin/bash

SOURCE_DIR="/opt/zhongjie/uccl_rdma/efa"

TARGET_MACHINES=("172.31.42.140" "172.31.39.44" "172.31.32.200" "172.31.36.4")

TARGET_DIR="/opt/zhongjie/uccl_rdma"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo ""
    echo "Installing on machine: $MACHINE"

    rsync -avz --delete "$SOURCE_DIR/" "ubuntu@$MACHINE:$TARGET_DIR/efa" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "Copy done on machine: $MACHINE"

      ssh "ubuntu@$MACHINE" "cd $TARGET_DIR/efa && make -j" > /dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "Compile successfully on machine: $MACHINE"
      else
        echo "Compile error on machine: $MACHINE"
      fi
    else
      echo "Can't access machine: $MACHINE"
    fi
  ) &
done

wait

echo "Done."