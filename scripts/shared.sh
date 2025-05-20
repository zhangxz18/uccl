# !/bin/bash

if [[ -z "${UCCL_HOME}" ]]; then
  echo "UCCL_HOME is not set or is empty"
  exit 1
else
  echo "UCCL_HOME is set to: ${UCCL_HOME}"
fi

get_nodes() {
    local input="$1"  # Path to the file containing the input
    awk '!/^#|^$/ {print $0}' "$input" | paste -sd "," -
}