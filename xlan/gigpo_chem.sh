#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=gigpon2
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=5:00:00
#SBATCH --output=job_log/GiGPO-teacher-coef-%j/output.txt
#SBATCH --error=job_log/GiGPO-teacher-coef-%j/error.txt


export MAX_CHEMICAL_N=2
export DISCOVERY_TASK="chemistry"
export MODEL_NAME="Qwen2.5-0.5B-Instruct"
export ingame_reward_coef=1.0
export EXPERIMENT_NAME="gigpo-n${MAX_CHEMICAL_N}-vabss-teacher0.5-${MODEL_NAME}-0904"
export EPOCHS=60
export MAX_STEP=30
export ENV_HISTORY_LENGTH=4

export TRAIN_DATA_SIZE=4
export VAL_DATA_SIZE=16
export GROUP_SIZE=4
export MINI_BATCH_SIZE=16
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=2
export CRITIC_MICRO_BATCH_SIZE_PER_GPU=2
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.40
export NUM_CPUS_PER_ENV_WORKER=0.1

export TEACHER_REWARD_COEF=0.5
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STYLE=cosine
export KL_LOSS_COEF=0.3
export GIGPO_MODE=mean_std_norm
export GIGPO_STEP_ADVANTAGE_W=1.0
export INVALID_ACTION_PENALTY_COEF=1.0

export ENABLE_FILTER_GROUP=True
export DYNAMIC_SEED_SAMPLER=True

#export RESUME_MODE=resume_path
#export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/gigpo-n2-0822-dynamic-group4-Qwen2.5-1.5B-Instruct/best_val_success"

export TEST_FREQ=2
export SAVE_FREQ=-1
export SAVE_BEST_VAL_SUCCESS=True

bash xlan/gigpo_base.sh "$@" || exit $?

BEST_VAL_CHECKPOINT="/home/xlan1/projects/verl-agent/checkpoints/Combinatorial-Chemistry-Agent/${EXPERIMENT_NAME}/best_val_success"

for eval_n in 1 2 3 4; do
    CHECKPOINT_PATH="${BEST_VAL_CHECKPOINT}" \
    MAX_CHEMICAL_N="${eval_n}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-eval-n${eval_n}" \
    OUTPUT_PATH="xlan/results/${EXPERIMENT_NAME}/eval-n${eval_n}.json" \
        bash xlan/eval_model.sh || exit $?
done
