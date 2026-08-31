#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=gigpon3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen2.5-1.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwe2.5-1.5B-error.txt


export MAX_CHEMICAL_N=2
export DISCOVERY_TASK="chemistry"
export MODEL_NAME="Qwen2.5-1.5B-Instruct"
export EXPERIMENT_NAME="gigpo-n${MAX_CHEMICAL_N}-0822-dynamic-group4-${MODEL_NAME}"
export EPOCHS=50
export MAX_STEP=35
export ENV_HISTORY_LENGTH=4

export TRAIN_DATA_SIZE=4
export VAL_DATA_SIZE=16
export GROUP_SIZE=4
export MINI_BATCH_SIZE=16
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=1
export CRITIC_MICRO_BATCH_SIZE_PER_GPU=1
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.40
export NUM_CPUS_PER_ENV_WORKER=0.1

export TEACHER_REWARD_COEF=1.0
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export LR_WARMUP_STYLE=cosine
export KL_LOSS_COEF=0.05
export GIGPO_MODE=mean_std_norm
export GIGPO_STEP_ADVANTAGE_W=1.0
export INVALID_ACTION_PENALTY_COEF=2.0

export ENABLE_FILTER_GROUP=True
export DYNAMIC_SEED_SAMPLER=True

export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n2-0822-dynamic-group4-Qwen2.5-1.5B-Instruct/best_val_success"

export SAVE_FREQ=50
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@"
