#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=grpo-n3
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=5:00:00
#SBATCH --output=job_log/grpo-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/grpo-%j/Qwen0.5B-error.txt

export MAX_CHEMICAL_N=3
export TEACHER_REWARD_COEF="${TEACHER_REWARD_COEF:-1.0}"
export EPOCHS=100
export MAX_STEP=20

export TRAIN_DATA_SIZE=6
export VAL_DATA_SIZE=12
export GROUP_SIZE=4
export MINI_BATCH_SIZE=24
export TRAIN_SEED_POOL=null
export DYNAMIC_SEED_SAMPLER=True

export LEARNING_RATE=1e-6
export KL_LOSS_COEF=0.02
export ENTROPY_COEFF=0.001
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=4
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=4
export NUM_CPUS_PER_ENV_WORKER=0.1

export EXPERIMENT_NAME="grpo-n3-0726-dynamics"
export RESUME_DATALOADER=False
#export RESUME_MODE=resume_path
#export RESUME_FROM_PATH="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-n3-0725-dynamics/global_step_50"
export SAVE_FREQ=10
export SAVE_BEST_VAL_SUCCESS=True


bash xlan/grpo_base.sh  "$@"
