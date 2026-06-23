#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN3c2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

: "${C2_CKPT:?Set C2_CKPT to a C2 global_step checkpoint directory. Example: C2_CKPT=checkpoints/GiGPO-discoveryworld/<c2-exp>/global_step_25}"

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="$C2_CKPT"
export EXPERIMENT_TAG="curr-n3-from-c2"

bash xlan/gigpo_base.sh "$@"
