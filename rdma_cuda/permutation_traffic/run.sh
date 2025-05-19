CTRL_NIC="ens10f0np0"

# BT, SA
BENCH=PT

mpirun --bind-to none -np 2 \
    -hostfile hostname.txt \
    --mca btl_tcp_if_include ${CTRL_NIC} \
    --mca plm_rsh_args "-o StrictHostKeyChecking=no" \
    --mca orte_base_help_aggregate 0 \
    -x GLOG_logtostderr=0 \
    ./permutation_traffic -benchtype ${BENCH}