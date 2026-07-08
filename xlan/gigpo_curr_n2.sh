#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiGn2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=5:00:00
#SBATCH --output=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-output.txt
#SBATCH --error=/home/xlan1/projects/verl-agent/job_log/gigpo-%j/gigpo-error.txt

export MAX_CHEMICAL_N=2
export EPOCHS=100
export MAX_STEP=15
export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.2
export SAVE_FREQ=10
export TRAIN_DATA_SIZE=16
export VAL_DATA_SIZE=16
export MINI_BATCH_SIZE=32
export GROUP_SIZE=2
export NUM_CPUS_PER_ENV_WORKER=0.1
export CURRICULUM_ENABLED=True
export CURRICULUM_TRAIN_FRACTION=0.7
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.3,0.6,0.1]}"
export CURRICULUM_TERMINAL_RESET_RATIO=0.125
export LR_WARMUP_STYLE=cosine
export LR_WARMUP_STEPS_RATIO=0.01
export LR_MIN_RATIO=0.1
export LR_NUM_CYCLES=0.5
export GIGPO_STEP_ADVANTAGE_W=1.0
export GIGPO_MODE=mean_norm
export GIGPO_ENABLE_SIMILARITY=False
export EXPERIMENT_TAG="gigpo-from-grpo-0707"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-n2-0707/global_step_90/actor"
#export MODEL_PATH="/home/xlan1/projects/verl-agent/qwen_hf_model"

bash xlan/gigpo_base.sh "$@"
