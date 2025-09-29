#!/bin/bash

#==========================================================================#
# Please add the SLURM parameter configuration above this horizontal line, #
# and read the README and F.A.Q. at the end of this document.              #
#==========================================================================#

# The number of GPUs is SLURM_NNODES * USER_GPUS_PER_NODE
export SLURM_JOBID=1
export SLURM_NNODES=1
export SLURM_NTASKS_PER_NODE=1
export USER_GPUS_PER_NODE=8          # <--------------------- Modify it in time!

# Remember to modify MASTER_ADDR
export MASTER_ADDR=YOUR_SERVER_IP
export MASTER_PORT=55555

export SLURM_JOB_NUM_NODES=$SLURM_NNODES
export SLURM_NTASKS=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export USER_NGPUS=$(($USER_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))

#- Check

if [[ -z $SLURM_NTASKS ]]; then
    echo "SLURM_NTASKS is empty, please check your SBATCH parameter."
    exit -1
fi
if [[ -z $SLURM_NTASKS_PER_NODE ]]; then
    echo "SLURM_NTASKS_PER_NODE is empty, please check your SBATCH parameter."
    exit -1
fi
task_size=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
if [[ $task_size != $SLURM_NTASKS ]]; then
    echo "NTASKS_PER_NODE * NNODE != NNTASK, please check your SBATCH parameter."
    exit -1
fi

if [[ $task_size != $USER_NGPUS ]]; then
    echo "INFO..."
    echo "That's a total of $SLURM_NTASKS tasks, requiring a total of $USER_NGPUS GPUs"
    echo "Becareful whether your program requires \$SLURM_NTASKS or NGPUS"
fi

#- Global Info

export WORLD_SIZE=${USER_NGPUS}

#- NCCL Setting

###
### IB here refers to RDMA, not the InfiniBand network in the narrow sense,
### it consists of RDMA over IB network, or RDMA over Converged Ethernet
### 
### RDMA's advantages: Zero-Copy and Kernel Bypass, make it faster than TCP stack
### Since the cluster is basically configured with IB NICs, the best performance is obtained when using RDMA
###
### The NCCL_DEBUG variable controls the debug information that is displayed from NCCL
### INFO - Prints debug information
### export NCCL_DEBUG="INFO"
###

export NCCL_IB_DISABLE=0                          # 0: Using RDMA,        1: Using TCP/IP
export NCCL_P2P_DISABLE=0                         # 0: Using P2P,         1: Not P2P, using cpu forwarding (high latency)
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3"


#- Log information
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo "Ntasks of jobs:=  " $SLURM_NTASKS
echo "NGPUs of jobs:=   " $USER_NGPUS
echo "MASTER_ADDR:=     " $MASTER_ADDR
echo "MASTER_PORT:=     " $MASTER_PORT
echo "WORLD_SIZE:=      " $WORLD_SIZE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "The job is triggered on node:"
echo "$(hostnamectl)"

echo "$(df -h | grep -v tmpfs)"

#- Job step
# (TODO) Be sure to modify the template.multi-gpus-task.sh file as well.
echo "=============== srun begins =================="

echo "HTTP PROXY is $http_proxy"
echo "HTTPS PROXY is $https_proxy"
unset http_proxy
unset https_proxy
echo "ROCR visible devices is $ROCR_VISIBLE_DEVICES"
unset ROCR_VISIBLE_DEVICES
bash train-multigpu.sh

#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
