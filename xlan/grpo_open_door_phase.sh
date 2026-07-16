#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=grpo-open
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/grpo-open-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-open-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=2
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EPOCHS=40
export MAX_STEP=1
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.02
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=16
export MINI_BATCH_SIZE=32
export GROUP_SIZE=8
export NUM_CPUS_PER_ENV_WORKER=0.1
export EXPERIMENT_NAME="grpo-open-door-phase-0702"
export ROLLOUT_TEMPERATURE=1.0
export ROLLOUT_TOP_P=1.0
export VAL_TEMPERATURE=0.1
export VAL_DO_SAMPLE=False


bash xlan/grpo_base.sh "$@"
