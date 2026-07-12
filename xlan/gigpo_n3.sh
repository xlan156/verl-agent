#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=gig-n3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt


export MAX_CHEMICAL_N=3
export EXPERIMENT_NAME="gigpo-n3-0712"
export CURRICULUM_ENABLED=False
export EPOCHS=100
export MAX_STEP=20
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=30
export GROUP_SIZE=4
export MINI_BATCH_SIZE=32
export NUM_CPUS_PER_ENV_WORKER=0.1
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export KL_LOSS_COEF=0.05

export MODEL_PATH="/home/xlan1/projects/verl-agent/qwen_hf_model"
export DISCOVERYWORLD_ANCHOR_MODE="belief_summary"

export SAVE_FREQ=-1
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@"
