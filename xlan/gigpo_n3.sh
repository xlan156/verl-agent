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
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EXPERIMENT_NAME="gigpo-n3-0716-group1215"
export EPOCHS=100
export MAX_STEP=30
export ENV_HISTORY_LENGTH=8

export TRAIN_DATA_SIZE=4
export TRAIN_SEED_POOL='[12,13,14,15]'
export VAL_DATA_SIZE=8
export GROUP_SIZE=8
export MINI_BATCH_SIZE=16
export NUM_CPUS_PER_ENV_WORKER=0.1
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export KL_LOSS_COEF=0.05

#export RESUME_MODE=resume_path
#export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n3-0712/best_val_success"

export SAVE_FREQ=-1
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@"
