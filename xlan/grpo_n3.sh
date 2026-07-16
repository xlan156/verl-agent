#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=grpo-n3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --output=job_log/grpo-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EPOCHS=100
export MAX_STEP=15
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.02
export SAVE_FREQ=200
export TRAIN_DATA_SIZE=32
export VAL_DATA_SIZE=50
export MINI_BATCH_SIZE=24
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export EXPERIMENT_NAME="grpo-n3-0711"
export SAVE_BEST_VAL_SUCCESS=True


bash xlan/grpo_base.sh  "$@"
