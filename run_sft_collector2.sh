for i in {1..10}; do
    python -m agent_system.environments.env_package.discovery.generate_discoveryworld_sft \
            --teacher_model qwen-plus \
            --num_episodes 1 \
            --max_env_steps 30 \
            --env_num 1 \
            --seed 0
done