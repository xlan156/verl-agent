#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt


export MAX_CHEMICAL_N=3
export EXPERIMENT_NAME="gigpo-0720"
export EPOCHS=60
export MAX_STEP=30
export ENV_HISTORY_LENGTH=4

export MODEL_PATH="/home/xlan1/projects/verl-agent/qwen_hf_model"
export TRAIN_DATA_SIZE=10
export VAL_DATA_SIZE=8
export MINI_BATCH_SIZE=20
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STYLE=cosine
export KL_LOSS_COEF=0.2

export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-0720-2/best_val_success"

export SAVE_FREQ=-1
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@"
