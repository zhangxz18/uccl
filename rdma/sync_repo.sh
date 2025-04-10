#!/bin/bash

SOURCE_DIR="/home/azureuser/uccl_rdma/"

# Read IPs from the 'ips' file into an array
IFS=$'\n' read -d '' -r -a TARGET_MACHINES < "ips"

TARGET_DIR="/home/azureuser/uccl_rdma/"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo "Installing on machine: $MACHINE"

    ssh -o StrictHostKeyChecking=no "azureuser@$MACHINE" "mkdir $TARGET_DIR" > /dev/null 2>&1

    rsync -avz -e 'ssh -o StrictHostKeyChecking=no' --delete "$SOURCE_DIR/" "azureuser@$MACHINE:$TARGET_DIR" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "Copy done on machine: $MACHINE"

      ssh -o StrictHostKeyChecking=no "azureuser@$MACHINE" "cd $TARGET_DIR/rdma && bash configure_rdma_ip.sh && make -j" > /dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "Compile rdma successfully on machine: $MACHINE"
      else
        echo "Compile error on machine: $MACHINE"
      fi
      ssh -o StrictHostKeyChecking=no "azureuser@$MACHINE" "cd $TARGET_DIR/rdma/permutation_traffic/ && make clean && make" > /dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "Compile pt successfully on machine: $MACHINE"
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
