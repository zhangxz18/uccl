#!/bin/bash

SOURCE_DIR="/home/azureuser/uccl_rdma/"

TARGET_MACHINES=(
  "74.179.81.170" 
  "74.179.82.149" 
  "74.179.81.10"
  "74.179.82.138"
  "128.85.41.27" 
  "128.85.40.255" 
  "128.85.41.120" 
  "128.85.41.236" 
  "128.85.40.216" 
  "128.85.40.197" 
  "128.85.41.81" 
  "128.85.41.11" 
  "128.85.41.55" 
  "128.85.41.253" 
  "128.85.40.243" 
  "128.85.42.40"
)

TARGET_DIR="/home/azureuser/uccl_rdma/"

if [ ! -d "$SOURCE_DIR" ]; then
  exit 1
fi

for MACHINE in "${TARGET_MACHINES[@]}"; do
  (
    echo "Installing on machine: $MACHINE"

    ssh "azureuser@$MACHINE" "mkdir $TARGET_DIR" > /dev/null 2>&1

    rsync -avz --delete "$SOURCE_DIR/" "azureuser@$MACHINE:$TARGET_DIR" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
      echo "Copy done on machine: $MACHINE"

      ssh "azureuser@$MACHINE" "cd $TARGET_DIR/rdma && bash configure_rdma_ip.sh && make -j" > /dev/null 2>&1
      if [ $? -eq 0 ]; then
        echo "Compile rdma successfully on machine: $MACHINE"
      else
        echo "Compile error on machine: $MACHINE"
      fi
      ssh "azureuser@$MACHINE" "cd $TARGET_DIR/rdma/permutation_traffic/ && make clean && make" > /dev/null 2>&1
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
