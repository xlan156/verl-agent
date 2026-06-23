#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=2
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export EXPERIMENT_TAG="curr-n2-mix702010"

bash xlan/gigpo_base.sh "$@"
