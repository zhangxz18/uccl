#!/bin/bash

SOURCE_DIR="/home/azureuser/uccl_rdma/rdma"

TARGET_MACHINES=("74.179.81.10" "74.179.81.170" "74.179.82.138" "74.179.82.149")

TARGET_DIR="/home/azureuser/uccl_rdma/"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo "Installing on machine: $MACHINE"

    rsync -avz --delete "$SOURCE_DIR/" "azureuser@$MACHINE:$TARGET_DIR/rdma" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "Copy done on machine: $MACHINE"

      ssh "azureuser@$MACHINE" "cd $TARGET_DIR/rdma && bash ~/uccl_rdma.sh && make -j" > /dev/null 2>&1
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
