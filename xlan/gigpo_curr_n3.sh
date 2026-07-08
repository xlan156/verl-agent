#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GiG-cN3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=5:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

export MAX_CHEMICAL_N=2
export PROJECT_NAME="${PROJECT_NAME:-GiGPO-discoveryworld}"
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export EXPERIMENT_TAG="curr-n3-from-grpo-gs130-step15"
export EPOCHS=180
export MAX_STEP=15
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=20
export GROUP_SIZE=4
export MINI_BATCH_SIZE=32
export NUM_CPUS_PER_ENV_WORKER=0.1
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export KL_LOSS_COEF=0.02
export SAVE_FREQ=10
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-n2-0628/global_step_90"

bash xlan/gigpo_base.sh "$@"
