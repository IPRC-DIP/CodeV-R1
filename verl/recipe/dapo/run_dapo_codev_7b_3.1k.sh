#!/bin/bash
set -x
set -euxo pipefail

project_name='DAPO'
exp_name='DAPO-Early-Qwen2.5-32B'

adv_estimator=grpo

kl_coef=0.0
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

# An early version for DAPO
enable_filter_groups=True
train_prompt_bsz=128
train_prompt_mini_bsz=64
n_resp_per_prompt=16
use_token_level_loss=True

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-16}
# Paths
# Algorithm
## Train
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout


# Performance Related Parameter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=4
ppo_max_token_len_per_gpu=32768
num_gpu=$(($USER_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))


export VLLM_USE_V1=1

echo "$WANDB_DIR"
echo "$SAVE_DIR"
echo "$WANDB_API_KEY"


# Set default model path if not provided
MODEL_PATH=YOUR_MODEL_PATH
DATA_PATH=YOUR_DATA_PATH

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/val.parquet \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=$max_response_length \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=999 \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.accelerate=True \
    data.gen_batch_size=$((($train_prompt_bsz + $num_gpu - 1) / $num_gpu * $num_gpu)) \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.use_token_level_loss=${use_token_level_loss} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(($ppo_max_token_len_per_gpu*2)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    reward_model.reward_manager=prime \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.overlong_buffer.enable=${enable_overlong_buffer} \
    custom_reward_function.overlong_buffer.len=${overlong_buffer_len} \
    custom_reward_function.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.train.path=verl/utils/reward_score/codev.py \
    custom_reward_function.train.name=compute_score_wrapper \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='codev' \
    trainer.experiment_name='codev-7b-3.1kdata' \
    trainer.n_gpus_per_node=$USER_GPUS_PER_NODE \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    +trainer.val_before_train=False \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 "${@:1}"

    
    # custom_reward_function.path=/nfs_global/S/zhuyaoyu/projects/dapo/verl/utils/reward_score/codev.py \