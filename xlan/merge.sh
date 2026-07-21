#!/bin/bash
#SBATCH --partition=cbuild
#SBATCH --job-name=merge-grpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/xlan1/projects/verl-agent/job_log/merge-%j.out
#SBATCH --error=/home/xlan1/projects/verl-agent/job_log/merge-%j.err

set -euo pipefail
set -x

cd "${SLURM_SUBMIT_DIR:-$HOME/projects/verl-agent}"
source "$HOME/venvs/verlagentdis/bin/activate"

# CHECKPOINT_DIR must be the directory containing actor/, for example
# best_val_success or global_step_100. The defaults merge this run's best model.
ACTOR_DIR="/home/xlan1/projects/verl-agent/checkpoints/GRPO-discoveryworld/grpo-0721/best_val_success/actor"
TARGET_DIR="/home/xlan1/projects/verl-agent/grpo_trained_model"

if [[ ! -f "$ACTOR_DIR/config.json" ]]; then
    echo "Missing Hugging Face config in actor checkpoint: $ACTOR_DIR/config.json" >&2
    exit 1
fi

if ! compgen -G "$ACTOR_DIR/model_world_size_*_rank_0.pt" > /dev/null; then
    echo "Missing FSDP model shard in: $ACTOR_DIR" >&2
    exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"

python "$PWD/scripts/model_merger.py" merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

echo "Merged Hugging Face model written to: $TARGET_DIR"
