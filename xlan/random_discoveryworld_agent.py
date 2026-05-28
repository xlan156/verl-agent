import random
import time
from typing import Dict, List

import ray
from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs


DISCOVERYWORLD_SKILLS: List[str] = [
    "move_to_key",
    "move_to_jar",
    "pick_up_key",
    "put_key_in_jar",
    "pick_up_jar",
    "wash_jar",
    "open_door",
    "move_to_dispenser_A",
    "move_to_dispenser_B",
    "move_to_dispenser_C",
    "move_to_dispenser_D",
    "use_dispenser_A_on_jar",
    "use_dispenser_B_on_jar",
    "use_dispenser_C_on_jar",
    "use_dispenser_D_on_jar",
    "remove_chemical_A",
    "remove_chemical_B",
    "remove_chemical_C",
    "remove_chemical_D",
]

def build_test_config(env_num: int = 1, max_steps: int = 30, seed: int = 0):
    """Minimal OmegaConf config so make_envs builds DiscoveryWorldEnvironmentManager."""
    cfg_dict = {
        "env": {
            "env_name": "discoveryworld",
            "seed": seed,
            "max_steps": max_steps,
            "history_length": 3,
            "rollout": {"n": 1},
            "resources_per_worker": {"num_cpus": 0.2, "num_gpus": 0.0},
            "discoveryworld": {
                "scenario_name": "Combinatorial Chemistry",
                "difficulty": "Challenge",
                "save_frames": False,
                "max_chemical_n": 1,
                "curriculum_enabled": True,
                "curriculum_train_fraction": 0.8,
                "curriculum_mix_ratios": [0.7, 0.2, 0.1],
                "curriculum_seed": seed,
                "curriculum": {
                    "enabled": True,
                    "train_fraction": 0.8,
                    "mix_ratios": [0.7, 0.2, 0.1],
                    "seed": seed,
                },
            },
        },
        "data": {
            "train_batch_size": env_num,
            "val_batch_size": 0,
        },
    }
    return OmegaConf.create(cfg_dict)
def sample_random_skill() -> str:
    return random.choice(DISCOVERYWORLD_SKILLS)


def run_env_manager_rollout(env_num: int = 1, max_env_steps: int = 5) -> None:
    """Use DiscoveryWorldEnvironmentManager + curriculum-enabled make_envs with random skills."""
    # 在构建 env manager 之前，先用 local_mode 初始化 Ray，方便本地单进程调试
    if not ray.is_initialized():
        #ray.init(address="auto")
        ray.init(address="auto", local_mode=True)

    config = build_test_config(env_num=env_num, max_steps=max_env_steps)
    env_manager, _ = make_envs(config)

    try:
        observations, infos = env_manager.reset(kwargs={})
        print("Initial text observation (env 0, truncated):")
        print(observations["text"][0][:500])
        print("---")

        dones = [False] * env_num
        total_rewards = [0.0] * env_num

        for step_idx in range(max_env_steps):
            if all(dones):
                break

            text_actions = ["done" if dones[i] else sample_random_skill() for i in range(env_num)]

            print(f"Step {step_idx:02d} text actions: {text_actions}")
            observations, rewards, step_dones, infos = env_manager.step(text_actions)
            projected_actions = [info.get("projected_action") for info in infos]
            valid_flags = [info.get("is_action_valid") for info in infos]
            print(f"Step {step_idx:02d} projected: {projected_actions} valid={valid_flags}")

            for i in range(env_num):
                dones[i] = bool(dones[i] or step_dones[i])
                total_rewards[i] += float(rewards[i])

            print(
                f"Step {step_idx:02d}: mean_reward={sum(rewards)/len(rewards):.4f}, "
                f"finished_envs={sum(dones)}/{env_num}"
            )
            # 打印一下当前 step 的合法性标记
            # print("  is_action_valid:", [info.get("is_action_valid") for info in infos])

        print("Rollout finished. Total rewards:", total_rewards)
    finally:
        env_manager.close()


if __name__ == "__main__":
    random.seed(42)
    start = time.time()
    run_env_manager_rollout(env_num=1, max_env_steps=5)
    print(f"Total wall time: {time.time() - start:.2f}s")
