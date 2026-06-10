#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GiG
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

module load 2023
module load CUDA/12.4.0
cd ~/projects/verl-agent
source ~/venvs/verlagentdis/bin/activate

# Setting VLLM and Ray address 
set -x
set -euo pipefail

unset ROCR_VISIBLE_DEVICES
ENGINE=${1:-vllm}
export RAY_BACKEND_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export head_node_ip=$(hostname --ip-address)
export port=6300
export RAY_ADDRESS="${head_node_ip}:${port}"

# Experiment configuration
model_name=Qwen2.5-0.5B-Instruct
project_name="GiGPO-discoveryworld"
model_path="sft/models/SFT-${model_name}-merged"
experiment_name="GiGPO-${model_name}-0608"
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Challenge}"

train_data_size=32
val_data_size=32
num_cpus_per_env_worker=0.1
group_size=1
num_gpus_per_node=1

#python3 -m sft.SFTtrain

ray start --head \
    --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --include-dashboard=false \
    --block &
RAY_HEAD_PID=$!
sleep 5

# Data preparation: only indicates modality (text) and data size.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Common configs
LEARNING_RATE=1e-7
KL_LOSS_COEF=0.15
EPOCHS=40

# Curriculum configuration
MAX_CHEMICAL_N=1
CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-True}"
CURRICULUM_TRAIN_FRACTION="${CURRICULUM_TRAIN_FRACTION:-0.8}"
CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
CURRICULUM_SEED="${CURRICULUM_SEED:-0}"

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
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$num_gpus_per_node \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.15 \
    critic.optim.lr=$LEARNING_RATE \
    critic.model.use_remove_padding=True \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=discoveryworld \
    env.seed=0 \
    env.max_steps=20 \
    env.rollout.n=$group_size \
    +env.discoveryworld.scenario_name="${SCENARIO_NAME}" \
    +env.discoveryworld.difficulty="${DIFFICULTY}" \
    +env.discoveryworld.save_frames=False \
    +env.discoveryworld.max_chemical_n=${MAX_CHEMICAL_N} \
    +env.discoveryworld.curriculum_enabled=${CURRICULUM_ENABLED} \
    +env.discoveryworld.curriculum_train_fraction=${CURRICULUM_TRAIN_FRACTION} \
    +env.discoveryworld.curriculum_mix_ratios="${CURRICULUM_MIX_RATIOS}" \
    +env.discoveryworld.curriculum_seed=${CURRICULUM_SEED} \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$num_gpus_per_node \
    trainer.nnodes=1 \
    trainer.log_llm_steps=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.resume_mode=auto \
    trainer.val_before_train=True "$@"

ray stop

