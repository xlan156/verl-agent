#!/usr/bin/env bash
# GiGPO | DiscoveryWorld | Qwen2.5 | FSDP + vLLM
#
# Thin experiment wrappers (for example gigpo_n3.sh) should export only the
# values that differ from these defaults. Additional Hydra overrides can be
# appended to the command line and are forwarded through "$@".

set -xeuo pipefail

module load 2023
module load CUDA/12.4.0
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/verl-agent}"
cd "${PROJECT_ROOT}"
source "$HOME/venvs/verlagentdis/bin/activate"

unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=info
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# ------------------------- user-adjustable -------------------------

# Model and runtime
ENGINE="${ENGINE:-vllm}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
MODEL_PATH="${MODEL_PATH:-Qwen/${MODEL_NAME}}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-${NUM_GPUS_PER_NODE:-1}}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

# Data and sequence lengths
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/verl-agent/text/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/verl-agent/text/test.parquet}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-32}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"

# Actor optimization
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.3}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
ACTOR_CLIP_RATIO_LOW="${ACTOR_CLIP_RATIO_LOW:-0.2}"
ACTOR_CLIP_RATIO_HIGH="${ACTOR_CLIP_RATIO_HIGH:-0.28}"
INVALID_ACTION_PENALTY_COEF="${INVALID_ACTION_PENALTY_COEF:-0.1}"
LR_WARMUP_STYLE="${LR_WARMUP_STYLE:-constant}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"

# Rollout and validation generation
GROUP_SIZE="${GROUP_SIZE:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-${NGPUS_PER_NODE}}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.4}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"

# Critic
CRITIC_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-4}"

# GiGPO
GIGPO_STEP_ADVANTAGE_W="${GIGPO_STEP_ADVANTAGE_W:-1.0}"
GIGPO_MODE="${GIGPO_MODE:-mean_norm}"
GIGPO_SIMILARITY_THRESH="${GIGPO_SIMILARITY_THRESH:-0.95}"

# DiscoveryWorld environment
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Challenge}"
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-2}"
MAX_STEP="${MAX_STEP:-30}"
ENV_HISTORY_LENGTH="${ENV_HISTORY_LENGTH:-8}"
ENV_SEED="${ENV_SEED:-0}"
TRAIN_SEED_POOL="${TRAIN_SEED_POOL:-null}"
TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-0.1}"
DISCOVERYWORLD_ENV_VARIANT="${DISCOVERYWORLD_ENV_VARIANT:-original}"

# Trainer, logging, and checkpoints
PROJECT_NAME="${PROJECT_NAME:-Combinatorial-Chemistry-Agent}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-gigpo-n${MAX_CHEMICAL_N}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_TAG}-envseed${ENV_SEED}}"
EPOCHS="${EPOCHS:-20}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SAVE_BEST_VAL_SUCCESS="${SAVE_BEST_VAL_SUCCESS:-False}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

# Optional preprocessing/SFT stage
DO_SFT="${DO_SFT:-False}"

# Ray
RAY_PORT="${RAY_PORT:-6300}"
RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-18}"

# ----------------------- end user-adjustable -----------------------

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=gigpo
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.train_batch_size=${TRAIN_DATA_SIZE}"
    "data.val_batch_size=${VAL_DATA_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
)

MODEL=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.warmup_style=${LR_WARMUP_STYLE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}"
    "actor_rollout_ref.actor.optim.min_lr_ratio=${LR_MIN_RATIO}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.clip_ratio_low=${ACTOR_CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH}"
    actor_rollout_ref.actor.use_invalid_action_penalty=True
    "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF}"
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    "actor_rollout_ref.rollout.name=${ENGINE}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=False
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
)

REF=(
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

CRITIC=(
    "critic.model.path=${MODEL_PATH}"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    "critic.ppo_micro_batch_size_per_gpu=${CRITIC_MICRO_BATCH_SIZE_PER_GPU}"
    critic.model.fsdp_config.param_offload=True
    critic.model.fsdp_config.optimizer_offload=False
)

ALGORITHM=(
    algorithm.use_kl_in_reward=False
    algorithm.gamma=0.99
    "algorithm.gigpo.step_advantage_w=${GIGPO_STEP_ADVANTAGE_W}"
    "algorithm.gigpo.mode=${GIGPO_MODE}"
    # Belief states that differ by one experiment can require opposite actions;
    # never merge them by fuzzy text similarity.
    algorithm.gigpo.enable_similarity=False
    "algorithm.gigpo.similarity_thresh=${GIGPO_SIMILARITY_THRESH}"
)

ENVIRONMENT=(
    env.env_name=discoveryworld
    "env.seed=${ENV_SEED}"
    "env.max_steps=${MAX_STEP}"
    "env.rollout.n=${GROUP_SIZE}"
    "env.history_length=${ENV_HISTORY_LENGTH}"
    "+env.discoveryworld.scenario_name=${SCENARIO_NAME}"
    "+env.discoveryworld.difficulty=${DIFFICULTY}"
    "+env.discoveryworld.env_variant=${DISCOVERYWORLD_ENV_VARIANT}"
    +env.discoveryworld.anchor_mode=belief_summary
    "+env.discoveryworld.max_chemical_n=${MAX_CHEMICAL_N}"
    "+env.discoveryworld.teacher_skill_reward_coef=${TEACHER_REWARD_COEF}"
    "+env.discoveryworld.train_seed_pool=${TRAIN_SEED_POOL}"
    "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER}"
)

TRAINER=(
    trainer.critic_warmup=0
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    trainer.log_llm_steps=True
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.save_best_val_success=${SAVE_BEST_VAL_SUCCESS}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.resume_from_path=${RESUME_FROM_PATH}"
    trainer.val_before_train=True
)

EXTRA=()

########################### setup ###########################

HEAD_NODE_IP="$(hostname --ip-address)"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

cleanup() {
    ray stop || true
}
trap cleanup EXIT

ray start --head \
    "--port=${RAY_PORT}" \
    "--num-cpus=${RAY_NUM_CPUS}" \
    --include-dashboard=false \
    --block &
sleep 5

python3 -m examples.data_preprocess.prepare \
    --mode text \
    "--train_data_size=${TRAIN_DATA_SIZE}" \
    "--val_data_size=${VAL_DATA_SIZE}"

if [[ "${DO_SFT}" == "True" ]]; then
    python3 -m sft.SFTtrain
fi

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${ALGORITHM[@]}" \
    "${ENVIRONMENT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
