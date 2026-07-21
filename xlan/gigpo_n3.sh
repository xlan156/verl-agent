#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=2:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt


export MAX_CHEMICAL_N=3
export EXPERIMENT_NAME="gigpo-0721"
export EPOCHS=60
export MAX_STEP=30
export ENV_HISTORY_LENGTH=4


export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=8
export MINI_BATCH_SIZE=32
export GROUP_SIZE=4
export NUM_CPUS_PER_ENV_WORKER=0.1

export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export LR_WARMUP_STYLE=cosine
export ENTROPY_COEFF=0.0
export KL_LOSS_COEF=0.2
export GIGPO_MODE=mean_std_norm
export GIGPO_STEP_ADVANTAGE_W=0.25

export MODEL_PATH="/home/xlan1/projects/verl-agent/grpo_trained_model"
##export RESUME_MODE=resume_path
#export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-0721/best_val_success"


export SAVE_FREQ=-1
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@"
