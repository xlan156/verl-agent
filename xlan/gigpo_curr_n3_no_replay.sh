#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN3nr
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=3:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="[1.0,0.0,0.0]"
export EXPERIMENT_TAG="curr-n3-mix-1-0-0-no-replay"
export EPOCHS=20
export MAX_STEP=15

bash xlan/gigpo_base.sh "$@"
