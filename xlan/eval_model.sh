#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=dw-eval
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
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

# Model. EVAL_MODE=checkpoint restores CHECKPOINT_PATH; EVAL_MODE=base uses MODEL_PATH directly.
EVAL_MODE="${EVAL_MODE:-checkpoint}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/$MODEL_NAME}"
# n3
#CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n2-0822-dynamic-group4-Qwen2.5-1.5B-Instruct/best_val_success}"

#n2
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n2-0820-uniform-teacher-group4/best_val_success}"


# DiscoveryWorld task. Chemistry remains the default for backward compatibility.
DISCOVERY_TASK="${DISCOVERY_TASK:-chemistry}"
case "$DISCOVERY_TASK" in
    chemistry)
        SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
        DIFFICULTY="${DIFFICULTY:-Challenge}"
        ;;
    plant)
        SCENARIO_NAME="${SCENARIO_NAME:-Plant Nutrients}"
        DIFFICULTY="${DIFFICULTY:-Normal}"
        ;;
    *)
        echo "DISCOVERY_TASK must be 'chemistry' or 'plant', got: $DISCOVERY_TASK" >&2
        exit 1
        ;;
esac

# Evaluation set. EVAL_SEEDS overrides split/size selection when provided.
# N = 1: EVAL_SIZE=2
# N = 2: EVAL_SIZE=4
# N = 3: EVAL_SIZE=8
# N = 4: EVAL_SIZE=14
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-3}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
case "$MAX_CHEMICAL_N" in
    1) DEFAULT_EVAL_SIZE=1 ;;
    2) DEFAULT_EVAL_SIZE=2 ;;
    3) DEFAULT_EVAL_SIZE=4 ;;
    4) DEFAULT_EVAL_SIZE=7 ;;
    *)
        echo "Unsupported MAX_CHEMICAL_N=$MAX_CHEMICAL_N; set EVAL_SIZE explicitly" >&2
        exit 1
        ;;
esac
EVAL_SIZE="${EVAL_SIZE:-${VAL_SIZE:-$DEFAULT_EVAL_SIZE}}"
ROLLOUTS_PER_SEED="${ROLLOUTS_PER_SEED:-5}"
EVAL_SEEDS="${EVAL_SEEDS:-}"
TARGET_TRAIN_FRACTION="${TARGET_TRAIN_FRACTION:-0.8}"

# Environment and generation
MAX_STEP="${MAX_STEP:-80}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_TOP_P="${VAL_TOP_P:-0.9}"
DISCOVERYWORLD_ENV_VARIANT="${DISCOVERYWORLD_ENV_VARIANT:-original}"
EVAL_SEED="${EVAL_SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.05}"

# Output and logging
OUTPUT_PATH="${OUTPUT_PATH:-xlan/results/discoveryworld-eval-${DISCOVERY_TASK}-${DIFFICULTY,,}-${EVAL_MODE}-${MODEL_NAME}-${SLURM_JOB_ID:-local}.json}"
LOG_LLM_STEPS="${LOG_LLM_STEPS:-true}"
PROJECT_NAME="${PROJECT_NAME:-Eval DiscoveryWorld}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EVAL_MODE}-eval-${DISCOVERY_TASK}-${DIFFICULTY,,}-${SLURM_JOB_ID:-local}}"

MODEL_SOURCE_ARGS=()
case "$EVAL_MODE" in
    checkpoint)
        if [[ ! -d "$CHECKPOINT_PATH/actor" ]]; then
            echo "Checkpoint actor directory not found: $CHECKPOINT_PATH/actor" >&2
            exit 1
        fi
        MODEL_SOURCE_ARGS=("$CHECKPOINT_PATH")
        ;;
    base)
        MODEL_SOURCE_ARGS=(--base-model)
        ;;
    *)
        echo "EVAL_MODE must be 'checkpoint' or 'base', got: $EVAL_MODE" >&2
        exit 1
        ;;
esac

case "$EVAL_SPLIT" in
    val|train|all) ;;
    *)
        echo "EVAL_SPLIT must be 'val', 'train', or 'all', got: $EVAL_SPLIT" >&2
        exit 1
        ;;
esac

mkdir -p "$(dirname "$OUTPUT_PATH")" "$MPLCONFIGDIR"

EXPLICIT_SEED_ARGS=()
if [[ -n "$EVAL_SEEDS" ]]; then
    read -r -a EVAL_SEED_VALUES <<< "$EVAL_SEEDS"
    EXPLICIT_SEED_ARGS=(--eval-seeds "${EVAL_SEED_VALUES[@]}")
fi

LLM_STEP_LOG_ARGS=(--log-llm-steps)
if [[ "${LOG_LLM_STEPS,,}" != "true" ]]; then
    LLM_STEP_LOG_ARGS=(--no-log-llm-steps)
fi

python xlan/eval_discoveryworld_checkpoint.py \
    "${MODEL_SOURCE_ARGS[@]}" \
    --model-path "$MODEL_PATH" \
    --scenario-name "$SCENARIO_NAME" \
    --difficulty "$DIFFICULTY" \
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
