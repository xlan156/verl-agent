#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=ccsft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=job_log/%j_sft_gigpo/cc_sft_output_%j.txt
#SBATCH --error=job_log/%j_sft_gigpo/cc_sft_error_%j.txt

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~
cd projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

export WANDB_KEY="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"
wandb login $WANDB_KEY

export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Scenario / difficulty selection (override when calling the script)
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Easy}"

num_cpus_per_env_worker=0.2 # CPU per DiscoveryWorld env worker; reduce to save CPU.
train_data_size=8 # number of parallel tasks (matches other PPO examples)
val_data_size=4

# 1) Merge SFT LoRA into base model
python sft/merge_sft_lora.py

# 2) Use merged SFT model for verl PPO
model_name=SFT-Qwen2.5-1.5B-Instruct-merged
model_path=sft/models/${model_name}

experiment_name="SFT-GIGPO-${model_name}-${SCENARIO_NAME}-${DIFFICULTY}"

num_gpus_per_node=1

# Data preparation: only indicates modality (text) and data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$num_gpus_per_node \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.15 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${model_path} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=discoveryworld \
    env.seed=0 \
    env.max_steps=35 \
    +env.discoveryworld.scenario_name="${SCENARIO_NAME}" \
    +env.discoveryworld.difficulty="${DIFFICULTY}" \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='verl_agent_discoveryworld' \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=$num_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True "$@"
