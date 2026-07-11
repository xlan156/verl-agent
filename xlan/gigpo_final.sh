#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=gigpo
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=2:00:00
#SBATCH --output=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-output.txt
#SBATCH --error=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-error.txt

export MAX_CHEMICAL_N=2
export EPOCHS=150
export MAX_STEP=20
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.1
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=12
export VAL_DATA_SIZE=10
export MINI_BATCH_SIZE=24
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export CURRICULUM_ENABLED=True
export CURRICULUM_TRAIN_FRACTION=0.7
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.5,0.4,0.1]}"
export CURRICULUM_TERMINAL_RESET_RATIO=0.125
export LR_WARMUP_STYLE=constant
export GIGPO_STEP_ADVANTAGE_W=0.8
export GIGPO_MODE=mean_std_norm
export GIGPO_ENABLE_SIMILARITY=False
export EXPERIMENT_TAG="gigpo-final-0709"
export DISCOVERYWORLD_ENV_VARIANT=original

export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-use-0709-envseed0-cseed0/global_step_70"


bash xlan/gigpo_base.sh "$@"
