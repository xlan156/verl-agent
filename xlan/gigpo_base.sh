#!/bin/bash

module load 2023
module load CUDA/12.4.0
cd ~/projects/verl-agent
source ~/venvs/verlagentdis/bin/activate

set -x
set -euo pipefail

unset ROCR_VISIBLE_DEVICES
ENGINE="${ENGINE:-vllm}"
export RAY_BACKEND_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export head_node_ip=$(hostname --ip-address)
export port="${RAY_PORT:-6300}"
export RAY_ADDRESS="${head_node_ip}:${port}"

# Shared experiment configuration. Thin wrapper scripts should override these
# variables so comparison runs differ only in the intended experimental factor.
MODEL_NAME="${MODEL_NAME:-Qwen2.5-0.5B-Instruct}"
PROJECT_NAME="${PROJECT_NAME:-Combinatorial-Chemistry-Agent}"
MODEL_PATH="${MODEL_PATH:-sft/models/SFT-${MODEL_NAME}-merged}"
SCENARIO_NAME="${SCENARIO_NAME:-Combinatorial Chemistry}"
DIFFICULTY="${DIFFICULTY:-Challenge}"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-32}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-64}"
GROUP_SIZE="${GROUP_SIZE:-2}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

LEARNING_RATE="${LEARNING_RATE:-5e-8}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.3}"
EPOCHS="${EPOCHS:-20}"
MAX_STEP="${MAX_STEP:-30}"
SAVE_FREQ="${SAVE_FREQ:-5}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

DO_SFT="${DO_SFT:-False}"
MAX_CHEMICAL_N="${MAX_CHEMICAL_N:-2}"
CURRICULUM_ENABLED="${CURRICULUM_ENABLED:-True}"
CURRICULUM_TRAIN_FRACTION="${CURRICULUM_TRAIN_FRACTION:-0.5}"
CURRICULUM_MIX_RATIOS="${CURRICULUM_MIX_RATIOS:-[0.7,0.2,0.1]}"
CURRICULUM_SEED="${CURRICULUM_SEED:-0}"
ENV_SEED="${ENV_SEED:-0}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-curr-n${MAX_CHEMICAL_N}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${EXPERIMENT_TAG}-envseed${ENV_SEED}-cseed${CURRICULUM_SEED}}"

ray start --head \
    --port=$port \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --include-dashboard=false \
    --block &
RAY_HEAD_PID=$!
sleep 5

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $TRAIN_DATA_SIZE \
    --val_data_size $VAL_DATA_SIZE

if [ "$DO_SFT" = "True" ]; then
    python3 -m sft.SFTtrain
fi

cleanup() {
    ray stop || true
}
trap cleanup EXIT

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS_PER_NODE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.actor.clip_ratio_low=0.1 \
    actor_rollout_ref.actor.clip_ratio_high=0.1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.15 \
    critic.optim.lr=$LEARNING_RATE \
    critic.model.use_remove_padding=True \
    critic.model.path=$MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=discoveryworld \
    env.seed=$ENV_SEED \
    env.max_steps=$MAX_STEP \
    env.rollout.n=$GROUP_SIZE \
    +env.discoveryworld.scenario_name="${SCENARIO_NAME}" \
    +env.discoveryworld.difficulty="${DIFFICULTY}" \
    +env.discoveryworld.save_frames=False \
    +env.discoveryworld.max_chemical_n=${MAX_CHEMICAL_N} \
    +env.discoveryworld.curriculum_enabled=${CURRICULUM_ENABLED} \
    +env.discoveryworld.curriculum_train_fraction=${CURRICULUM_TRAIN_FRACTION} \
    +env.discoveryworld.curriculum_mix_ratios="${CURRICULUM_MIX_RATIOS}" \
    +env.discoveryworld.curriculum_seed=${CURRICULUM_SEED} \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.log_llm_steps=True \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=5 \
    trainer.total_epochs=$EPOCHS \
    trainer.resume_mode=$RESUME_MODE \
    trainer.val_before_train=True "$@"

ray stop
