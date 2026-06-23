#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN4c3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

: "${C3_CKPT:?Set C3_CKPT to a C3 global_step checkpoint directory. Example: C3_CKPT=checkpoints/GiGPO-discoveryworld/<c3-exp>/global_step_25}"

export MAX_CHEMICAL_N=4
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="$C3_CKPT"
export EXPERIMENT_TAG="curr-n4-from-c3"

bash xlan/gigpo_base.sh "$@"
