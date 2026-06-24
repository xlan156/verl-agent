#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GRPOn2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GRPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GRPO-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

export MAX_CHEMICAL_N=2
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.5,0.4,0.1]}"
export EPOCHS=30
export MAX_STEP=15
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=24
export GROUP_SIZE=4
export KL_LOSS_COEF=0.04

bash xlan/grpo_base.sh "$@"