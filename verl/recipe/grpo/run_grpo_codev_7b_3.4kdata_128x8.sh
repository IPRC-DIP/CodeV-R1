#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/nfs_global/S/lvhanqi/LLaMA-Factory/saves/Qwen2.5-Coder-7B-Instruct-codev-r1-87k/full/sft_6epoch"
fi

echo "GLOO_SOCKET_IFNAME is $GLOO_SOCKET_IFNAME!"

MAX_TOKEN_PER_GPU=32768
GPU_MEMORY_UTILIZATION=0.8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$CURR_DIR/data/codev/v1/4.8k_r1_filtered/train.parquet \
    data.val_files=$CURR_DIR/data/codev/v1_1/10k_qwq/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1152 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    +actor_rollout_ref.actor.optim.betas=[0.9,0.999] \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.use_dynamic_bsz=True\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_PER_GPU \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(($MAX_TOKEN_PER_GPU*2)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.000 \
    reward_model.reward_manager=prime \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='codev' \
    trainer.experiment_name='codev-distill-7b-16k-t1-kl0' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    +trainer.val_before_train=True \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=100 "${@:1}"