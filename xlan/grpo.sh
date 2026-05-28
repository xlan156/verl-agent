#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GR
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=06:00:00
#SBATCH --output=job_log/GRPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GRPO-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

module load 2023
module load CUDA/12.4.0
cd ~/projects/verl-agent
source ~/venvs/verlagentdis/bin/activate

# Setting VLLM and Ray address 
set -x
unset ROCR_VISIBLE_DEVICES
ENGINE=${1:-vllm}
export RAY_BACKEND_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export head_node_ip=$(hostname --ip-address)
export port=6400
export RAY_ADDRESS="${head_node_ip}:${port}"

# Experiment configuration
model_name=Qwen2.5-0.5B-Instruct
project_name="GRPO-discoveryworld"
model_path="sft/models/SFT-${model_name}-merged"

train_data_size=10
val_data_size=10
num_cpus_per_env_worker=0.1
experiment_name="GRPO-SFT-${model_name}-0514"
group_size=4
num_gpus_per_node=1
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Challenge}"

#python -m sft.SFTtrain

ray start --head \
    --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --include-dashboard=false \
    --block &
sleep 5

# Data preparation: only indicates modality (text) and data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.optim.lr=2e-8 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$num_gpus_per_node \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.15 \
    algorithm.use_kl_in_reward=False \
    env.env_name=discoveryworld \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    +env.discoveryworld.scenario_name="${SCENARIO_NAME}" \
    +env.discoveryworld.difficulty="${DIFFICULTY}" \
    +env.discoveryworld.max_chemical_n=3 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=$num_gpus_per_node \
    trainer.nnodes=1 \
    trainer.log_llm_steps=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True "$@"

