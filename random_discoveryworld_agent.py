import json
import random
import time
from typing import Any, Dict

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv


def sample_random_action(info: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a simple random action based on the DiscoveryWorld info dict.

    This is only meant to exercise the environment, not to be a good policy.
    """
    known_actions: Dict[str, Any] = info.get("known_actions") or {}
    available_names = list(known_actions.keys())

    preferred = [
        "MOVE_DIRECTION",
        "ROTATE_DIRECTION",
        "PICKUP",
        "PUT",
        "USE",
        "TALK",
    ]
    candidate_names = [a for a in available_names if a in preferred] or available_names
    action_name = random.choice(candidate_names) if candidate_names else "MOVE_DIRECTION"

    # Get accessible objects from the raw observation
    ui = (info.get("raw_observation") or {}).get("ui", {})
    inventory = ui.get("inventoryObjects", [])
    accessible = ui.get("accessibleEnvironmentObjects", [])
    objects = inventory + accessible

    def any_object_uuid() -> Any:
        if not objects:
            return None
        return random.choice(objects).get("uuid")

    action: Dict[str, Any] = {"action": action_name, "arg1": None, "arg2": None}

    if action_name in ("PICKUP", "TALK"):
        action["arg1"] = any_object_uuid()
    elif action_name in ("PUT", "USE"):
        action["arg1"] = any_object_uuid()
        action["arg2"] = any_object_uuid()
    elif action_name in ("MOVE_DIRECTION", "ROTATE_DIRECTION"):
        action["arg1"] = random.choice(["north", "east", "south", "west"])
    else:
        args = (known_actions.get(action_name) or {}).get("args", [])
        for arg_name in args:
            if arg_name in ("arg1", "arg2"):
                action[arg_name] = any_object_uuid()

    return action


def run_random_rollout(max_env_steps: int = 30) -> None:
    # Directly create a single DiscoveryWorldEnv without any Ray actors.
    env = DiscoveryWorldEnv(
        seed=0,
        scenario_name="Combinatorial Chemistry",
        difficulty="Easy",
        max_steps=max_env_steps,
        thread_id=0,
    )

    text_obs, info = env.reset()

    print("Initial text observation (truncated):")
    print(text_obs[:500])
    print("---")

    done = False
    total_reward = 0.0

    for step_idx in range(max_env_steps):
        if done:
            break

        action_dict = sample_random_action(info)
        print(f"Step {step_idx:02d} action: {json.dumps(action_dict)}")

        text_obs, reward, done, info = env.step(action_dict)
        ui = info.get("raw_observation").get("ui")
        task_desc = ui.get("taskProgress")[0].get("description")
        total_reward += float(reward)

        print(
            f"  reward={float(reward):.4f}, done={done}, "
            f"score_normalized={info.get('score_normalized', 0.0):.4f}"
        )

    print(f"Rollout finished. Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    random.seed(42)
    start = time.time()
    run_random_rollout(max_env_steps=30)
    print(f"Total wall time: {time.time() - start:.2f}s")
