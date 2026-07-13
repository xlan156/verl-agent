#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=gig-anc
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=6:00:00
#SBATCH --output=job_log/GiGPO-Anchor-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GiGPO-Anchor-%j/Qwen0.5B-error.txt

set -euo pipefail

export MAX_CHEMICAL_N=3
export EPOCHS=100
export MAX_STEP=20
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE=10
export GROUP_SIZE=2
export MINI_BATCH_SIZE=16
export NUM_CPUS_PER_ENV_WORKER=0.1
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export KL_LOSS_COEF=0.02
export SAVE_FREQ=-1

run_ablation() {
    local anchor_mode="$1"
    local experiment_name="$2"
    shift 2

    echo "===== Running ${experiment_name} with anchor_mode=${anchor_mode} ====="
    DISCOVERYWORLD_ANCHOR_MODE="${anchor_mode}" \
    EXPERIMENT_NAME="${experiment_name}" \
        bash xlan/gigpo_base.sh "$@"
}

run_ablation raw_obs "gigpo-n3-anchor-rawobs" "$@"
run_ablation state_summary "gigpo-n3-anchor-state-summary" "$@"
run_ablation belief_summary "gigpo-n3-anchor-belief-summary" "$@"
