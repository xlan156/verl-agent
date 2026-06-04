#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=GEN
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=job_log/GEN-%j/Qwen0.5B-output.txt
#SBATCH --error=job_log/GEN-%j/Qwen0.5B-error.txt
#SBATCH --reservation=terv92681

cd ~/projects/verl-agent
source ~/venvs/verlagentdis/bin/activate

python -m xlan.gensft \
    --episodes 150 \
    --max-steps 60 \
    --max-chemical-n 2 \
    --is-train \

python -m xlan.gensft \
    --episodes 120 \
    --max-steps 60 \
    --max-chemical-n 1 \
    --is-train \

python -m xlan.gensft \
    --episodes 20 \
    --max-steps 60 \
    --max-chemical-n 2 \

python -m xlan.gensft \
    --episodes 10 \
    --max-steps 60 \
    --max-chemical-n 1 \