import json
import random
import time
from typing import Any, Dict

import ray
from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs
from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import (  # type: ignore
    DiscoveryWorldAPI,
)

AVAILABLE_ACTIONS = DiscoveryWorldAPI.listKnownActionsStatic()

def build_test_config(env_num: int = 1, max_steps: int = 30, seed: int = 0):
    """Minimal OmegaConf config so make_envs builds DiscoveryWorldEnvironmentManager."""
    cfg_dict = {
        "env": {
            "env_name": "discoveryworld-test",  # 只要包含 "discoveryworld" 即可
            "seed": seed,
            "max_steps": max_steps,
            "history_length": 3,
            "rollout": {"n": 1},
            "resources_per_worker": {"num_cpus": 0.2, "num_gpus": 0.0},
            "discoveryworld": {
                "scenario_name": "Combinatorial Chemistry",
                "difficulty": "Easy",
            },
        },
        "data": {
            "train_batch_size": env_num,
            "val_batch_size": 0,
        },
    }
    return OmegaConf.create(cfg_dict)


def sample_random_action(info: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a simple random *valid* action based on the DiscoveryWorld info dict."""
    
    action_name = random.choice(["MOVE_DIRECTION", "ROTATE_DIRECTION", "PICKUP", "USE", "TELEPORT_TO_LOCATION"])

    # Get accessible objects from the raw observation
    ui = (info.get("raw_observation") or {}).get("ui", {})
    inventory = ui.get("inventoryObjects", [])
    accessible = ui.get("accessibleEnvironmentObjects", [])
    objects = inventory + accessible
    
    teleport_locations = info.get("teleport_locations")

    def any_object_uuid(items) -> Any:
        if not items:
            return None
        return random.choice(items).get("uuid")

    action: Dict[str, Any] = {"action": action_name, "arg1": None, "arg2": None}

    if action_name in ("PICKUP", "TALK"):
        action["arg1"] = any_object_uuid(accessible)
    elif action_name in ("PUT", "USE"):
        action["arg1"] = any_object_uuid(inventory)
        action["arg2"] = any_object_uuid(accessible)
    elif action_name in ("MOVE_DIRECTION", "ROTATE_DIRECTION"):
        action["arg1"] = random.choice(["north", "east", "south", "west"])
    elif action_name == "TELEPORT_TO_LOCATION":
        action["arg1"] = random.choice(list(teleport_locations.keys()))
        # Handle any other actions that might have different argument structures
        pass

    return action


def simulate_llm_response(obs_text: str, info: Dict[str, Any]) -> str:
    """Simulate an LLM response: sometimes valid, sometimes malformed."""
    string_candidates = [
        "OPEN_JAR",
        "I choose to pick up the key",
        "Use the substance A on the jar",
        "move to the east",
        "USE_SUBSTANCE ON JAR",
        'USE("jar", "Substance A")',
    ]
    action_json_str = json.dumps({"action": "MOVE_DIRECTION", "arg1": "west"})
    
    string_candidates = [f"<think> This step is .... </think><action>{action_json_str}</action> "]
    out = random.choice(string_candidates)
    return out


def build_natural_response(info: Dict[str, Any]) -> str:
    """Generate natural language that mentions directions or object names."""
    object_seen = info.get("object_seen") or {}
    names = [str(name) for name in object_seen.keys() if name]

    directions = ["north", "south", "east", "west"]
    templates = [
        "move {direction}",
        "rotate {direction}",
        "go {direction}",
        "turn to the {direction}",
        "pick up the {obj}",
        "open the {obj}",
        "use the {obj} on the {obj2}",
        "put the {obj} in the {obj2}",
    ]

    template = random.choice(templates)
    direction = random.choice(directions)
    obj = random.choice(names) if names else "thing"
    obj2 = random.choice(names) if len(names) > 1 else obj

    return template.format(direction=direction, obj=obj, obj2=obj2)


def run_env_manager_rollout(env_num: int = 1, max_env_steps: int = 20) -> None:
    """Use DiscoveryWorldEnvironmentManager + discoveryworld_projection with a fake LLM."""
    # 在构建 env manager 之前，先用 local_mode 初始化 Ray，方便本地单进程调试
    if not ray.is_initialized():
        ray.init(local_mode=True, num_cpus=2)

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

            text_actions = []
            for i in range(env_num):
                if dones[i]:
                    # 给一个占位字符串；projection 会用记忆/默认动作兜底
                    text_actions.append("done")
                else:
                    response = simulate_llm_response(observations["text"][i], infos[i])
                    text_actions.append(response)

            print(f"Step {step_idx:02d} fakeLLM: {response}")
            observations, rewards, step_dones, infos = env_manager.step(text_actions)
            print(
                f"Step {step_idx:02d} projected: {infos[0].get('projected_action')} "
                f"valid={infos[0].get('is_action_valid')}"
            )

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
    run_env_manager_rollout(env_num=1, max_env_steps=30)
    print(f"Total wall time: {time.time() - start:.2f}s")
