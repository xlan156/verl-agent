#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --job-name=coll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=job_log/generate_sft_%j/output_%j.txt
#SBATCH --error=job_log/generate_sft_%j/error_%j.txt
#SBATCH --reservation=terv92681

module load 2023
module load Miniconda3/23.5.2-0

cd ~
cd projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

for i in {1..5}; do
  python -m agent_system.environments.env_package.discovery.generate_discoveryworld_sft \
    --teacher_model qwen-plus \
    --num_episodes 1 \
    --max_env_steps 30 \
    --env_num 1 \
    --seed 0
done