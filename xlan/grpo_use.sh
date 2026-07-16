#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=grpoj
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
#SBATCH --output=job_log/grpo-jar-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-jar-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=2
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EPOCHS=200
export MAX_STEP=15
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.02
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=16
export MINI_BATCH_SIZE=32
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export EXPERIMENT_NAME="grpo-derustmoderate-0708"
export ENV_VARIANT=derustmoderate

export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-n2-0707/global_step_145"


bash xlan/grpo_base.sh  "$@"
