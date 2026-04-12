#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=lora_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=job_log/lora_%j/lora_output_%j.txt
#SBATCH --error=job_log/lora_%j/lora_error_%j.txt
#SBATCH --reservation=terv92681

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~
cd projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

python -m sft.eval_sft_in_env \
    --episodes 3 \
    --max_env_steps 30 \
    --temperature 0.7 \
