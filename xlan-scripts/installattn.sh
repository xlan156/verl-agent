#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=flash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=flash-attn_output_%A.txt
#SBATCH --error=flash-attn_error_%A.txt
#SBATCH --reservation=terv92681

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

mkdir -p "$HOME/.tmp"
export TMPDIR="$HOME/.tmp"

# Optionally avoid pip cache to simplify IO
# pip install --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1
cd ~
cd projects/verl-agent

echo 
pip3 install -e .
pip3 install vllm==0.8.2