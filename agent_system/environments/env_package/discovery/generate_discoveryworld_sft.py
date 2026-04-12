import argparse
import json
import os
import random
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import trange

from agent_system.environments.prompts.discoveryworld import (
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
    DISCOVERYWORLD_TEMPLATE,
)
from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.projection import (
    discoveryworld_projection,
)
from agent_system.memory import SimpleMemory
from sft.cartesian_dataset import extract_unique_actions, copy_action_order


with open("qwen_apikey.txt", "r") as f:
    API_KEY = f.read().strip()
    
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S", time.localtime())


def read_human_trajectory(file_path: str) -> str:
    """Read human trajectory data from a JSONL file and return a pretty string."""
    trajectory: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                trajectory.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in trajectory)


def call_teacher_llm(prompt: str, model: str, temperature: float = 0.2) -> str:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    content = response.choices[0].message.content
    return content or ""


def collect_sft_episodes(
    env: DiscoveryWorldEnv,
    teacher_model: str,
    num_episodes: int,
    max_env_steps: int,
    output_path: str,
    seed: int,
    history_length: int,
) -> None:
    """Generate SFT samples using DiscoveryWorld templates (with or without history).

    We directly step a single DiscoveryWorldEnv (no Ray manager) and maintain a
    lightweight in-script memory so that when history_length > 0 we can format
    prompts with DISCOVERYWORLD_TEMPLATE, matching the runtime behaviour.
    """
    random.seed(seed)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    human_trajectory = (
        "Here is a human's playing trajectory as an example, you can use this as a reference. "
        "The assistant's response means human's action choice.\n"
    )
    human_trajectory += read_human_trajectory("sft/sft_pairs_combinatorial_chemistry_easy4.jsonl")

    # single-env memory, similar to DiscoveryWorldEnvironmentManager + SimpleMemory
    mem = SimpleMemory()
    
    human_actions = copy_action_order("sft/sft_pairs_combinatorial_chemistry_easy4.jsonl")
    human_actions.insert(0, json.dumps({"action": "MOVE_DIRECTION", "arg1": "west"}, ensure_ascii=False))  # Add an initial action to align with the first observation
    unique_actions = extract_unique_actions("sft/sft_pairs_combinatorial_chemistry_easy4.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for ep in trange(num_episodes, desc="Generating SFT pairs"):
            # Reset a fresh DiscoveryWorld episode
            text_obs, info = env.reset()
            mem.reset(batch_size=1)

            for step_idx in range(max_env_steps):
                # Gather real UI/text and metadata from env
                task_description = info.get("task_description", "") or f"Episode {ep} task in DiscoveryWorld."
                ui_json = text_obs

                teleport_locations: Dict[str, Any] = info.get("teleport_locations", {}) or {}
                teleport_str = "\n".join(loc for loc in teleport_locations.keys())
                last_action_result: Dict[str, Any] = info.get("last_action_result", {}) or {}
                last_result_str = json.dumps(last_action_result, ensure_ascii=False)

                # Decide whether to use history template
                if history_length <= 0 or len(mem[0]) == 0:
                    # No history yet, or disabled
                    prompt = DISCOVERYWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_description,
                        ui_json=ui_json,
                        teleport_locations=teleport_str,
                        last_action_result=last_result_str,
                    )
                else:
                    memory_contexts, valid_lens = mem.fetch(
                        history_length,
                        obs_key="text_obs",
                        action_key="action",
                    )
                    history_block = memory_contexts[0]
                    history_len = valid_lens[0]

                    prompt = DISCOVERYWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=len(mem[0]),
                        history_length=history_len,
                        action_history=history_block,
                        current_step=len(mem[0]) + 1,
                        ui_json=ui_json,
                        teleport_locations=teleport_str,
                        last_action_result=last_result_str,
                    )
                
                #actions = extract_unique_actions("sft/sft_pairs_combinatorial_chemistry_easy5.jsonl")
                random_action = random.choice(unique_actions)
                random_action_str = json.dumps(random_action, ensure_ascii=False)

                #teacher_output = call_teacher_llm(prompt + "\n\n" + human_trajectory, model=teacher_model)
                #projected_actions, _valids = discoveryworld_projection([teacher_output])
                #env_action = projected_actions[0]
                #random_action_str = human_actions[step_idx]
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": random_action_str},
                    ]
                }
                f_out.write(json.dumps(example, ensure_ascii=False) + "\n")

                mem.store({"text_obs": [text_obs], "action": [random_action_str]})

                text_obs, _reward, done, info = env.step(random_action_str)
                
                #print(f"Episode {ep+1}, Step {step_idx+1}")
                #print(f"Action taken: {random_action_str}")
                #print(f"Reward: {_reward}, Done: {done}")
                if done:
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DiscoveryWorld SFT data using a teacher LLM."
    )
    parser.add_argument("--output", type=str, default=f"sft/discoveryworld_sft_{TIMESTAMP}.jsonl")
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_env_steps", type=int, default=50)
    # Single-env SFT generation; env_num is kept for backward compatibility but unused.
    parser.add_argument("--env_num", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario_name", type=str, default="Combinatorial Chemistry")
    parser.add_argument("--difficulty", type=str, default="Easy")
    parser.add_argument("--history_length", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DiscoveryWorldEnv(
        seed=args.seed,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_steps=args.max_env_steps,
        thread_id=0,
    )

    start = time.time()
    collect_sft_episodes(
        env=env,
        teacher_model=args.teacher_model,
        num_episodes=args.num_episodes,
        max_env_steps=args.max_env_steps,
        output_path=args.output,
        seed=args.seed,
        history_length=args.history_length,
    )
    elapsed = time.time() - start
    print(f"Finished collecting SFT data to {args.output} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
