#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=job_log/%j/agent_dis_eval_output_%j.txt
#SBATCH --error=job_log/%j/agent_dis_eval_error_%j.txt

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~
cd projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

# Optional: enable Weights & Biases logging by exporting your key
# export WANDB_KEY="<your_wandb_key_here>"
# wandb login $WANDB_KEY

export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Scenario / difficulty selection (override when calling the script)
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Easy}"

# Which PPO experiment to evaluate (must match checkpoint directory name)
# Default matches the existing directory checkpoints/verl_agent_discoveryworld/ppo_Qwen2.5-1.5B-Instruct
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_Qwen2.5-1.5B-Instruct}"

num_cpus_per_env_worker=0.5 # CPU per DiscoveryWorld env worker
train_data_size=1          # number of parallel tasks
val_data_size=1

model_name=Qwen2.5-1.5B-Instruct
num_gpus_per_node=1

# Data preparation: only indicates modality (text) and data size.
python3 -m examples.data_preprocess.prepare \
	--mode 'text' \
	--train_data_size $train_data_size \
	--val_data_size $val_data_size

# Run PPO in evaluation-only mode, resuming from the latest checkpoint
python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=gae \
	data.train_files=$HOME/data/verl-agent/text/train.parquet \
	data.val_files=$HOME/data/verl-agent/text/test.parquet \
	data.train_batch_size=$train_data_size \
	data.val_batch_size=$val_data_size \
	data.max_prompt_length=4096 \
	data.max_response_length=512 \
	data.filter_overlong_prompts=True \
	data.truncation='error' \
	data.return_raw_chat=True \
	actor_rollout_ref.model.path=Qwen/$model_name \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=1 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.01 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$num_gpus_per_node \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
	actor_rollout_ref.rollout.enable_chunked_prefill=False \
	actor_rollout_ref.rollout.enforce_eager=False \
	actor_rollout_ref.rollout.free_cache_engine=False \
	actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
	actor_rollout_ref.rollout.val_kwargs.do_sample=True \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	actor_rollout_ref.actor.use_invalid_action_penalty=True \
	actor_rollout_ref.actor.invalid_action_penalty_coef=0.15 \
	critic.optim.lr=1e-5 \
	critic.model.use_remove_padding=True \
	critic.model.path=Qwen/$model_name \
	critic.model.enable_gradient_checkpointing=True \
	critic.ppo_micro_batch_size_per_gpu=1 \
	critic.model.fsdp_config.param_offload=False \
	critic.model.fsdp_config.optimizer_offload=False \
	algorithm.use_kl_in_reward=False \
	env.env_name=discoveryworld \
	env.seed=0 \
	env.max_steps=50 \
	+env.discoveryworld.scenario_name="${SCENARIO_NAME}" \
	+env.discoveryworld.difficulty="${DIFFICULTY}" \
	env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
	trainer.critic_warmup=0 \
	trainer.logger="['console','wandb']" \
	trainer.project_name='verl_agent_discoveryworld' \
	trainer.experiment_name=$EXPERIMENT_NAME \
	trainer.n_gpus_per_node=$num_gpus_per_node \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=5 \
	trainer.total_epochs=5 \
	trainer.val_before_train=True \
	trainer.val_only=True \
	trainer.resume_mode=auto \
	trainer.log_eval_steps=True "$@"

