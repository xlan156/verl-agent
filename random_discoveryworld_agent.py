import json
import random
import time
from typing import Any, Dict

import ray
from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs


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


def simulate_llm_response(obs_text: str, info: Dict[str, Any]) -> str:
    """Simulate an LLM response: sometimes valid, sometimes malformed."""
    p = random.random()

    # 60%: 完全合法的 <think>/<action> 包装
    if p < 0.6:
        act = sample_random_action(info)
        act_json = json.dumps(act)
        return f"<think>I will try a reasonable action.</think><action>{act_json}</action>"

    # 20%: 有 JSON 但 action 字段缺失或错误
    if p < 0.8:
        bad = sample_random_action(info)
        # 删除 action 字段或改成非法动作
        if random.random() < 0.5:
            bad.pop("action", None)
        else:
            bad["action"] = "JUMP_OVER_WALL"
        act_json = json.dumps(bad)
        return f"<think>Trying something odd.</think><action>{act_json}</action>"

    # 10%: JSON 包在 ```json``` 代码块里，外面有说明文字
    if p < 0.9:
        act = sample_random_action(info)
        act_json = json.dumps(act, indent=2)
        return (
            "Here is my action in JSON:\n"  # projection 应该能从代码块中挖出 JSON
            "```json\n" + act_json + "\n```"
        )

    # 10%: 完全乱写、带中文，应该被判 invalid，并触发 projection 里的回退逻辑
    return "我觉得向左走两步，然后随便看看。This is not a valid action JSON."


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
    run_env_manager_rollout(env_num=1, max_env_steps=20)
    print(f"Total wall time: {time.time() - start:.2f}s")
