#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=dw-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=job_log/eval-ckpt/eval-ckpt-%j.out
#SBATCH --error=job_log/eval-ckpt/eval-ckpt-%j.err

set -euo pipefail
set -x

module load 2023
module load CUDA/12.4.0

cd "${SLURM_SUBMIT_DIR:-$HOME/projects/verl-agent}"
source "$HOME/venvs/verlagentdis/bin/activate"

unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-local}"

# This default can be overridden when submitting, for example:
# CHECKPOINT_PATH=checkpoints/.../global_step_100 sbatch xlan/eval_model.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n3-0712/best_val_success}"

if [[ ! -d "$CHECKPOINT_PATH/actor" ]]; then
    echo "Checkpoint actor directory not found: $CHECKPOINT_PATH/actor" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
# VAL_SIZE means the number of distinct environment seeds selected from the
# fixed validation pool. It is NOT the
# total number of evaluation episodes. Each selected seed is evaluated
# ROLLOUTS_PER_SEED times, so:
#   total episodes = VAL_SIZE * ROLLOUTS_PER_SEED
# VAL_SIZE cannot exceed the validation-pool size determined by
# MAX_CHEMICAL_N and TARGET_TRAIN_FRACTION. With the defaults below (N=3,
# train_fraction=0.8), there are 20 target combinations: seeds 0-15 are train
# and seeds [16, 17, 18, 19] are validation, so the maximum VAL_SIZE is 4.
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-4}"
VAL_SIZE="${VAL_SIZE:-7}"
ROLLOUTS_PER_SEED="${ROLLOUTS_PER_SEED:-20}"

TARGET_TRAIN_FRACTION="${TARGET_TRAIN_FRACTION:-0.8}"
MAX_STEP="${MAX_STEP:-30}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
DISCOVERYWORLD_ENV_VARIANT="${DISCOVERYWORLD_ENV_VARIANT:-original}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
OUTPUT_PATH="${OUTPUT_PATH:-xlan/results/discoveryworld-eval-${SLURM_JOB_ID:-local}.json}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$MPLCONFIGDIR"

python xlan/eval_discoveryworld_checkpoint.py \
    "$CHECKPOINT_PATH" \
    --model-path "$MODEL_PATH" \
    --val-size "$VAL_SIZE" \
    --rollouts-per-seed "$ROLLOUTS_PER_SEED" \
    --max-chemical-n "$MAX_CHEMICAL_N" \
    --target-train-fraction "$TARGET_TRAIN_FRACTION" \
    --max-steps "$MAX_STEP" \
    --temperature "$VAL_TEMPERATURE" \
    --top-p "$VAL_TOP_P" \
    --env-variant "$DISCOVERYWORLD_ENV_VARIANT" \
    --seed "$EVAL_SEED" \
    --num-gpus "$NUM_GPUS" \
    --num-cpus-per-env "$NUM_CPUS_PER_ENV_WORKER" \
    --output "$OUTPUT_PATH"
