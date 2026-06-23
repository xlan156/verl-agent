#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GiG-cN3tf90
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=True
export CURRICULUM_TRAIN_FRACTION=0.9
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export EXPERIMENT_TAG="curr-n3-trainfrac90"
export EPOCHS=30
export MAX_STEP=15
export TRAIN_DATA_SIZE=20
export VAL_DATA_SIZE=30
export GROUP_SIZE=2


bash xlan/gigpo_base.sh "$@"
