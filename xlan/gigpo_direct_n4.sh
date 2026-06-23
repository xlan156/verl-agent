#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-dN4
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=4
export CURRICULUM_ENABLED=False
export EXPERIMENT_TAG="direct-n4"

bash xlan/gigpo_base.sh "$@"
