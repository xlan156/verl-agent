#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=grpon3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/grpo-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=2
export EPOCHS=150
export MAX_STEP=15
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.02
export SAVE_FREQ=5
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=16
export MINI_BATCH_SIZE=32
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export EXPERIMENT_NAME="grpo-n2-0707"
export CURRICULUM_ENABLED=True
export CURRICULUM_TRAIN_FRACTION=0.7
export CURRICULUM_MIX_RATIOS="[0.3,0.6,0.1]"
export CURRICULUM_TERMINAL_RESET_RATIO=0.125


bash xlan/grpo_base.sh  "$@"
