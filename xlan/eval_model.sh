#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=dw-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=job_log/dw-eval-%j.out
#SBATCH --error=job_log/dw-eval-%j.err

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

# Required: pass the checkpoint directory when submitting, for example:
# CHECKPOINT_PATH=checkpoints/.../best_val_success sbatch xlan/eval_model.sh
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a global_step_N or best_val_success directory}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
VAL_SIZE="${VAL_SIZE:-2}"
ROLLOUTS_PER_SEED="${ROLLOUTS_PER_SEED:-10}"
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-2}"
TARGET_TRAIN_FRACTION="${TARGET_TRAIN_FRACTION:-0.8}"
MAX_STEP="${MAX_STEP:-30}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
DISCOVERYWORLD_ENV_VARIANT="${DISCOVERYWORLD_ENV_VARIANT:-original}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
OUTPUT_PATH="${OUTPUT_PATH:-results/discoveryworld-eval-${SLURM_JOB_ID:-local}.json}"

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
