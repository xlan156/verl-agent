#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=agent_dis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --output=job_log/%A/agent_dis_output_%A.txt
#SBATCH --error=job_log/%A/agent_dis_error_%A.txt
#SBATCH --reservation=terv92681

module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~
cd projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh

conda activate verl-agent-dis

export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES
sh examples/ppo_trainer/run_discoveryworld.sh
