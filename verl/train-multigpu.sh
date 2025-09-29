#!/bin/bash

# Extract device names and merge them into a comma-separated string
THIS_UP_IB_DEV=$(ibdev2netdev | grep Up | grep ib | awk '{print $1}' | paste -sd ',' -)
export NCCL_IB_HCA=$THIS_UP_IB_DEV

#- Log infomation

node_dev_msg="
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Task run on: $(hostname -s);
GPU devices: $(nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit);
InfiniBand devices: $(ibdev2netdev);
NCCL_IB_HCA=$THIS_UP_IB_DEV;
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"

node_task_msg="
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Task run on: $(hostname -s), PID: ${SLURM_TASK_PID},
USE GPU ${CUDA_VISIBLE_DEVICES} of this node (GPUs_PER_Node, not PER_Task);
GlobalID : $SLURM_PROCID    of $SLURM_NTASKS,
NodeID   : $SLURM_NODEID    of $SLURM_JOB_NUM_NODES,
LocalID  : $SLURM_LOCALID    of $SLURM_NTASKS_PER_NODE;
GPUs_PER_Task = $USER_NGPUS / $SLURM_NTASKS = $(($USER_NGPUS/$SLURM_NTASKS)),
MASTER_ADDR   = $MASTER_ADDR
MASTER_PORT   = $MASTER_PORT
WORLD_SIZE    = $WORLD_SIZE
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
"

echo $node_dev_msg
echo $node_task_msg

#- Important setting!!!
##  otherwise it will cause an error of insufficient RDMA resources:
ulimit -l unlimited
##  otherwise it will result in an insufficient virtual memory size error, especially when loading LLM:
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

# You may need to load environments in slurm
# #- Load environments
# source /tools/module_env.sh
# ##- language
# # module load python3/3.8.16
# module load gcc/9.3.0

# ##- CUDA
# module load cuda-cudnn/11.8-8.8.1
# export CUDA_HOME=/tools/cluster-software/cuda-cudnn/cuda-11.8.0-8.8.1
# which nvcc
# echo $CUDA_HOME

echo "Task $SLURM_PROCID: "$(module list)              # list modules loaded
echo "Task $SLURM_PROCID: "$(which gcc)
echo "Task $SLURM_PROCID: "$(which python)
echo "Task $SLURM_PROCID: "$(which python3)

#- WARNING! DO NOT MODIFY your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Node $SLURM_NODEID, LocalID $SLURM_LOCALID: Use GPU ${CUDA_VISIBLE_DEVICES}"
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM

# ##- Monitor
# # The script continues executing other tasks while the following command will execute after a while
# module load slurm-tools/v1.0
# (sleep 3h && slurm-gpu-atop-log-stats $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES) &
# echo "Main program continues to run. Monitoring information will be exported after three hours."

#- Main program execution

##- virtualenv
source ~/.bashrc
conda activate verl-codev-r1

echo "Python path after activation:"
python -c "import sys; print(sys.path)"

##- Job step TODO

# ray's default GCS(Global Control Store) port is 6379 
# and default dashboard port is 8265
# need to set `"working_dir": "."` in --runtime-env-json, otherwise working_dir will set to ~(/home/S/your_name) by default

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=120
export RAY_record_ref_creation_sites=1
export RAY_IGNORE_UNHANDLED_ERRORS=0
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=TRUE
export VLLM_USE_V1=1

export RAY_TEMP_DIR="/tmp/ray_$SLURM_JOBID"
echo "RAY TEMP DIR is $RAY_TEMP_DIR"

USER=$(whoami)

export WANDB_API_KEY=YOUR_WANDB_API_KEY
wandb login $WANDB_API_KEY
export WANDB_MODE=offline

echo "USER GPUS PER NODE IS $USER_GPUS_PER_NODE"
ray stop --force

MASTER_IP=$([[ $MASTER_ADDR =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && echo $MASTER_ADDR || nslookup $MASTER_ADDR | awk '/^Address: / { print $2 }')
DASHBOARD_PORT=8265

IP_LIST=$(hostname -I)
echo "IP list is $IP_LIST\n"
echo "Master IP is $MASTER_IP\n"

if [[ $IP_LIST == *"$MASTER_IP"* ]]; then
    # launch the master node of ray in container
    ray start --head --node-ip-address $MASTER_ADDR --port $MASTER_PORT --num-gpus $USER_GPUS_PER_NODE --temp-dir=$RAY_TEMP_DIR # https://github.com/OpenRLHF/OpenRLHF/issues/339
fi

# sleep 99999
export RAY_START_TIMEOUT=180
# wait for master node
timeout $RAY_START_TIMEOUT bash -c "while ! nc -z $MASTER_ADDR ${MASTER_PORT}; do sleep 2; done"
if [ $? -ne 0 ]; then
    echo "Ray start on master node time out!!!"
    ray stop --force
    exit 1
fi

if [[ $IP_LIST != *"$MASTER_IP"* ]]; then
    # # if you want to launch ray on more nodes, use
    ray start --address $MASTER_ADDR:$MASTER_PORT --num-gpus $USER_GPUS_PER_NODE --temp-dir=$RAY_TEMP_DIR
fi
# wait for other nodes
timeout $RAY_START_TIMEOUT bash -c "while [ \$(ray status | grep -c 'node_') -lt \$SLURM_NTASKS ]; do sleep 2; done"
if [ $? -ne 0 ]; then
    echo "Timeout waiting for worker nodes!"
    ray stop --force
    exit 1
fi

echo "All worker nodes are ready!"
ray status
# only need to submit job on the master node, 
# and submitting on other nodes will cause network errors
if [[ $IP_LIST == *"$MASTER_IP"* ]]; then
    ray list nodes

    SCRIPT_TO_RUN="$PWD/recipe/dapo/run_dapo_codev_7b_3.1k.sh"
    export SAVE_DIR="$PWD/results/codev_3.1k_dapo"

    mkdir -p $SAVE_DIR
    cp $SCRIPT_TO_RUN $SAVE_DIR

    RUNTIME_ENV=$(jq -n --arg save_dir "$SAVE_DIR" '{
            "pip": ["ray"],
            "working_dir": ".",
            "excludes": ["ckpt/", "tmp/", "ret_one/", "data/", "results/", ".git/"],
            "disable_caching": true,
            "env_vars": {"SAVE_DIR": $save_dir, "WANDB_DIR": $save_dir}
        }')
    ray job submit --address="http://127.0.0.1:$DASHBOARD_PORT" --runtime-env-json="$RUNTIME_ENV" -- bash $SCRIPT_TO_RUN
    
    ray stop --force
else
    # Function to check connection to master node
    check_connection() {
        timeout 60 bash -c "while ! nc -z $MASTER_ADDR ${MASTER_PORT}; do sleep 5; done"
        return $?
    }
    
    while true; do
        if ! check_connection; then
            echo "Connection to master node lost. Exiting worker node."
            break
        fi
        sleep 60  # Check every 60 seconds
    done
    ray stop --force
fi

echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
# This will overwrite any existing atop logs from previous runs.
# WARNING: If your program times out or is terminated by scancel,
#          the above script part might not execute correctly.

