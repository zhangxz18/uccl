#!/bin/bash

ip_address=$(ifconfig eth0 | awk '/inet / {print $2}')

if [ -z "$ip_address" ]; then
    echo "Error!"
    exit 1
fi

FILE="transport_config.h"
sed -i "s/static std::string SINGLE_IP(\".*\");/static std::string SINGLE_IP(\"$ip_address\");/g" "$FILE"

echo "Modity SINGLE_IP: $ip_address"