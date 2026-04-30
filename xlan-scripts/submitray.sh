#!/bin/bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_mig
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --output=job_log/ppo_discovery_%A.out
#SBATCH --error=job_log/ppo_discovery_%A.err

# load necessary modules
module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.4.0

cd ~/projects/verl-agent
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh
conda activate verl-agent-dis

# replace these information with your own
verl_workdir=$HOME/data/verl-agent
train_files=$HOME/data/verl-agent/text/train.parquet
val_files=$HOME/data/verl-agent/text/test.parquet
apptainer_image_path=$HOME/containers/verl_rocm.sif
# replace these information with your own

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=("$nodes")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer run --nv --bind $verl_workdir $apptainer_image_path \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        apptainer run --nv --bind $verl_workdir $apptainer_image_path \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer run --nv --bind $verl_workdir $apptainer_image_path \
    python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 128 \
    --val_data_size 128

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    apptainer run --nv --bind $verl_workdir $apptainer_image_path \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=discoveryworld \
    env.seed=0 \
    env.max_steps=50 \
    +env.discoveryworld.scenario_name="Combinatorial Chemistry" \
    +env.discoveryworld.difficulty="easy" \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${SLURM_GPUS_PER_NODE}" \
    trainer.nnodes="${SLURM_NNODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee verl_demo_slurm.log
