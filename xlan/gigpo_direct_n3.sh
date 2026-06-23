#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-dN3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=False
export EXPERIMENT_TAG="direct-n3"

bash xlan/gigpo_base.sh "$@"
