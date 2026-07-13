#!/bin/bash
#SBATCH --partition=cbuild
#SBATCH --job-name=gen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=job_log/GEN-%j/gendata-output.txt
#SBATCH --error=job_log/GEN-%j/gendata-error.txt

cd ~/projects/verl-agent
source ~/venvs/verlagentdis/bin/activate



python -m xlan.gen_data \
    --episodes 30 \
    --max-steps 60 \
    --max-chemical-n 1

python -m xlan.gen_data \
    --episodes 30 \
    --max-steps 60 \
    --max-chemical-n 2
