#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=grpon2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=job_log/grpo-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-%j/Qwen0.5B-error.txt


export EPOCHS=200
export MAX_STEP=15
export KL_LOSS_COEF=0.02
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=32
export VAL_DATA_SIZE=32

bash xlan/grpo_base.sh  "$@"