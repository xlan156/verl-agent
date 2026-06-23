#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN3r
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=True
export CURRICULUM_MIX_RATIOS="[0.5,0.3,0.2]"
export EXPERIMENT_TAG="curr-n3-mix-5-3-2"
export EPOCHS=30
export MAX_STEP=15
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=64
export NUM_CPUS_PER_ENV_WORKER=0.1
export GROUP_SIZE=8

bash xlan/gigpo_base.sh "$@"
