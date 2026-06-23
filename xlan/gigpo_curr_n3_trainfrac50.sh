#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=GiG-cN3tf50
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --output=job_log/GiGPO-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export CURRICULUM_ENABLED=True
export CURRICULUM_TRAIN_FRACTION=0.5
export CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
export EXPERIMENT_TAG="curr-n3-trainfrac50"

export EPOCHS=20
export MAX_STEP=30
export TRAIN_DATA_SIZE=50
export VAL_DATA_SIZE=64
export GROUP_SIZE=2
export DO_SFT=True

bash xlan/gigpo_base.sh "$@"
