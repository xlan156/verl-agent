#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=gigp
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=8:00:00
#SBATCH --output=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-output.txt
#SBATCH --error=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-error.txt

export MAX_CHEMICAL_N=2
export EPOCHS=100
export MAX_STEP=20
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.01
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=12
export VAL_DATA_SIZE=10
export MINI_BATCH_SIZE=24
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export CURRICULUM_ENABLED=False
export LR_WARMUP_STYLE=constant
export LR_WARMUP_STEPS_RATIO=0.01
export LR_MIN_RATIO=0.1
export LR_NUM_CYCLES=0.5
export GIGPO_STEP_ADVANTAGE_W=1.0
export GIGPO_MODE=mean_std_norm
export GIGPO_ENABLE_SIMILARITY=False
export EXPERIMENT_TAG="gigpo-pickjar-0709"
export DISCOVERYWORLD_ENV_VARIANT=pickjar


bash xlan/gigpo_base.sh "$@"
