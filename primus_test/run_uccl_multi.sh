cd ~/source/base/Primus
# get host_name and map it to rank_n
host_name=$(hostname)
if [ "$host_name" == "tw051" ]; then
    rank_n=0
elif [ "$host_name" == "tw045" ]; then
    rank_n=1
elif [ "$host_name" == "tw040" ]; then
    rank_n=2
elif [ "$host_name" == "tw035" ]; then
    rank_n=3
fi

for comm_backend in {uccl,rccl}
do
    for channel_n in {8,16,32}
    do
        for qp_n in {4,8,16}
        do
            # if the file ~/source/uccl_test_result/${COMM_BACKEND}_channel${NCCL_MIN_NCHANNELS}_qp${NCCL_IB_QPS_PER_CONNECTION}_rank${NODE_RANK}.ansi exists, skip
            if [ "$comm_backend" == "rccl" ]; then
                if [ -f ~/source/uccl_test_result/${comm_backend}_channel${channel_n}_qp${qp_n}_rank${rank_n}.ansi ]; then
                    echo "file ~/source/uccl_test_result/${comm_backend}_channel${channel_n}_qp${qp_n}_rank${rank_n}.ansi exist, skip"
                    continue
                fi
            elif [ "$comm_backend" == "uccl" ]; then
                if [ -f ~/source/uccl_test_result/${comm_backend}_channel${channel_n}_qp_rank${rank_n}.ansi ]; then
                    echo "file ~/source/uccl_test_result/${comm_backend}_channel${channel_n}_qp_rank${rank_n}.ansi exist, skip"
                    break
                fi
            fi
            if [ "$comm_backend" == "rccl" ]; then
                cmd="COMM_BACKEND=${comm_backend} NCCL_MIN_NCHANNELS=${channel_n} NCCL_MAX_NCHANNELS=${channel_n} NCCL_IB_QPS_PER_CONNECTION=${qp_n} NODE_RANK=${rank_n} bash uccl.sh"
                echo "Command: $cmd"
                eval $cmd
            elif [ "$comm_backend" == "uccl" ]; then
                cmd="COMM_BACKEND=${comm_backend} NCCL_MIN_NCHANNELS=${channel_n} NCCL_MAX_NCHANNELS=${channel_n} NODE_RANK=${rank_n} bash uccl.sh"
                echo "Command: $cmd"
                eval $cmd
            fi
            if [ "$rank_n" == 0 ]; then
                sleep 5
            else
                sleep 15
            fi
            if [ "$comm_backend" == "uccl" ]; then
                break
            fi
        done
    done
done