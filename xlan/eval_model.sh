#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --job-name=dw-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:10:00
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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/dapo-0723/best_val_success}"

if [[ ! -d "$CHECKPOINT_PATH/actor" ]]; then
    echo "Checkpoint actor directory not found: $CHECKPOINT_PATH/actor" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
# EVAL_SPLIT selects the fixed seed pool: "val" (default) or "train".
# EVAL_SIZE means the number of distinct environment seeds selected from that
# pool. It is NOT the
# total number of evaluation episodes. Each selected seed is evaluated
# ROLLOUTS_PER_SEED times, so:
#   total episodes = EVAL_SIZE * ROLLOUTS_PER_SEED
# EVAL_SIZE cannot exceed the selected pool size determined by
# MAX_CHEMICAL_N and TARGET_TRAIN_FRACTION. With the defaults below (N=3,
# train_fraction=0.8), there are 20 target combinations: seeds 0-15 are train
# and seeds [16, 17, 18, 19] are validation. The maximum EVAL_SIZE is therefore
# 16 for EVAL_SPLIT=train and 4 for EVAL_SPLIT=val.
# Examples:
#   EVAL_SPLIT=train EVAL_SIZE=16 sbatch xlan/eval_model.sh
#   EVAL_SPLIT=val   EVAL_SIZE=4  sbatch xlan/eval_model.sh
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-3}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
# Keep VAL_SIZE as a fallback for older submission commands.
EVAL_SIZE="${EVAL_SIZE:-${VAL_SIZE:-16}}"
ROLLOUTS_PER_SEED="${ROLLOUTS_PER_SEED:-5}"
# Optional whitespace-separated explicit seeds. This overrides EVAL_SIZE in the
# Python entry point, e.g. EVAL_SPLIT=val EVAL_SEEDS=19.
EVAL_SEEDS="${EVAL_SEEDS:-}"

if [[ "$EVAL_SPLIT" != "val" && "$EVAL_SPLIT" != "train" ]]; then
    echo "EVAL_SPLIT must be 'val' or 'train', got: $EVAL_SPLIT" >&2
    exit 1
fi

TARGET_TRAIN_FRACTION="${TARGET_TRAIN_FRACTION:-0.8}"
MAX_STEP="${MAX_STEP:-40}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
DISCOVERYWORLD_ENV_VARIANT="${DISCOVERYWORLD_ENV_VARIANT:-original}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.05}"
OUTPUT_PATH="${OUTPUT_PATH:-xlan/results/discoveryworld-eval-${SLURM_JOB_ID:-local}.json}"
LOG_LLM_STEPS="${LOG_LLM_STEPS:-true}"
PROJECT_NAME="${PROJECT_NAME:-Combinatorial-Chemistry-Agent}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-checkpoint-eval-${SLURM_JOB_ID:-local}}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$MPLCONFIGDIR"

EXPLICIT_SEED_ARGS=()
if [[ -n "$EVAL_SEEDS" ]]; then
    # Intentional word splitting permits a list such as EVAL_SEEDS="17 19".
    read -r -a EVAL_SEED_VALUES <<< "$EVAL_SEEDS"
    EXPLICIT_SEED_ARGS=(--eval-seeds "${EVAL_SEED_VALUES[@]}")
fi

LLM_STEP_LOG_ARGS=(--log-llm-steps)
if [[ "$LOG_LLM_STEPS" != "true" ]]; then
    LLM_STEP_LOG_ARGS=(--no-log-llm-steps)
fi

python xlan/eval_discoveryworld_checkpoint.py \
    "$CHECKPOINT_PATH" \
    --model-path "$MODEL_PATH" \
    --eval-split "$EVAL_SPLIT" \
    --eval-size "$EVAL_SIZE" \
    "${EXPLICIT_SEED_ARGS[@]}" \
    --rollouts-per-seed "$ROLLOUTS_PER_SEED" \
    --max-chemical-n "$MAX_CHEMICAL_N" \
    --target-train-fraction "$TARGET_TRAIN_FRACTION" \
    --max-steps "$MAX_STEP" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --temperature "$VAL_TEMPERATURE" \
    --top-p "$VAL_TOP_P" \
    --env-variant "$DISCOVERYWORLD_ENV_VARIANT" \
    --seed "$EVAL_SEED" \
    --num-gpus "$NUM_GPUS" \
    --num-cpus-per-env "$NUM_CPUS_PER_ENV_WORKER" \
    --project-name "$PROJECT_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    "${LLM_STEP_LOG_ARGS[@]}" \
    --output "$OUTPUT_PATH"

ray stop
