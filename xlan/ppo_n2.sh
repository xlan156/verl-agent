#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --job-name=pn2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=/home/xlan1/projects/verl-agent/job_log/ppo-%j/ppo-output.txt
#SBATCH --error=/home/xlan1/projects/verl-agent/job_log/ppo-%j/ppo-error.txt

export MODEL_PATH="/home/xlan1/projects/verl-agent/qwen_hf_model"

export MAX_CHEMICAL_N=2
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EPOCHS=100
export MAX_STEP=20
export LEARNING_RATE=1e-6

export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=8
export MINI_BATCH_SIZE=8
export SAVE_FREQ=10


bash xlan/ppo_gae_base.sh "$@"
