import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from tqdm import trange

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.curriculum import format_chemical_state
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.utils import format_rust_update
from agent_system.environments.prompts.discoveryworld import (
    DISCOVERYWORLD_TEMPLATE,
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
)
from agent_system.environments.env_package.discovery.seed import build_fixed_seed_pools_by_amount


def build_prompt(
    state_obs: str,
    curriculum_state: Any,
    step_count: int,
    max_steps: int,
    action_history: List[Dict[str, Any]],
    max_chemical_n: int,
) -> str:
    step_info = f"Step: {step_count} / {max_steps}"
    curriculum_state_text = format_chemical_state(curriculum_state) if curriculum_state is not None else "None"

    recent_records = action_history[-3:]
    if recent_records:
        memory_start_index = len(action_history) - len(recent_records)
        if memory_start_index > 0:
            previous_rust_level = action_history[memory_start_index - 1].get("rust_level")
        else:
            previous_rust_level = None

        memory_lines: List[str] = []
        for step_offset, record in enumerate(recent_records):
            action = record.get("action")
            rust_level = record.get("rust_level")
            if not action:
                continue
            rust_update = format_rust_update(previous_rust_level, rust_level)
            memory_lines.append(f"{step_offset + 1}. {action} -> {rust_update}")
            previous_rust_level = rust_level

        memory_actions = "\n".join(memory_lines)
        return DISCOVERYWORLD_TEMPLATE.format(
            max_chemical_n=max_chemical_n,
            curriculum_state=curriculum_state_text,
            state_obs=state_obs,
            step_info=step_info,
            memory_actions=memory_actions,
        )

    return DISCOVERYWORLD_TEMPLATE_NO_HIS.format(
        max_chemical_n=max_chemical_n,
        curriculum_state=curriculum_state_text,
        state_obs=state_obs,
        step_info=step_info,
    )


def create_env(seed: int, max_steps: int, **kwargs) -> DiscoveryWorldEnv:
    return DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Challenge",
        seed=seed,
        max_steps=max_steps,
        **kwargs
    )


def rollout_episode(seed: int, max_steps: int, **kwargs) -> List[Dict[str, Any]]:
    env = create_env(seed=seed, max_steps=max_steps, **kwargs)
    state_obs, info = env.reset()
    teacher = RulebasedAgentSkill(env)
    skill_agent = CombinatorialChemistrySkill(env)

    records: List[Dict[str, Any]] = []
    action_history: List[Dict[str, Any]] = []
    done = False

    while not done:
        teacher_skill = teacher.select_skill(info)
        if teacher_skill:
            prompt = build_prompt(
                state_obs=state_obs.replace("\\n", "\n"),
                curriculum_state=info.get("curriculum_state"),
                step_count=env._steps,
                max_steps=env._max_steps,
                action_history=action_history,
                max_chemical_n=env._max_chemical_n,
            )
            records.append(
                {
                    "messages":[
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": teacher_skill,
                        }
                    ]
                }
            )
        else:
            with open("teacher_skill.txt", "a") as f:
                f.write("======================\n")
                f.write(f"self.is_key_in_jar: {env.is_key_in_jar}\n")
                f.write(f"used_dispensers: {env.used_dispensers}\n")
                f.write(f"Observation:\n{state_obs}\n\n")

        if random.random() < 0.5:
            random_skill = skill_agent.sample_random_skill()
        else:
            random_skill = teacher_skill
        state_obs, _, done, info = env.step(random_skill)
        action_history.append(
            {
                "action": random_skill,
                "rust_level": info.get("key_rust_level"),
            }
        )

    env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT data from DiscoveryWorld rollouts.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to sample.")
    parser.add_argument("--seed", type=int, default=2, help="Base env seed used for rollout scenarios.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max environment steps per episode.")
    parser.add_argument("--is-train", action="store_true", help="Generate training data (default: validation data).")
    parser.add_argument("--max-chemical-n", type=int, default=2, help="Total chemical amount.")
    parser.add_argument("--curriculum-seed", type=int, default=0, help="Seed used to build the fixed curriculum pool (default: --seed).")
    parser.add_argument("--curriculum-train-fraction", type=float, default=0.8, help="Train fraction for the fixed curriculum pool.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records: List[Dict[str, Any]] = []
    curriculum_seed = args.seed if args.curriculum_seed is None else int(args.curriculum_seed)
    seed_pools = build_fixed_seed_pools_by_amount(
        base_seed=curriculum_seed,
        max_amount=args.max_chemical_n,
        num_chemicals=4,
        min_chemicals=1,
        train_fraction=args.curriculum_train_fraction,
    )
    selected_split = "train" if args.is_train else "val"
    seeds_to_use = list(seed_pools.get(args.max_chemical_n, {}).get(selected_split, []))
    if not seeds_to_use:
        seeds_to_use = [int(args.seed)]

    len_all_records = 0
    if args.is_train:
        output_file = Path("sft/skill_sft_data_train.jsonl")
    else:
        output_file = Path("sft/skill_sft_data_val.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for ep in trange(args.episodes):
            episode_seed = seeds_to_use[ep % len(seeds_to_use)]
            episode_records = rollout_episode(
                seed=episode_seed,
                max_steps=args.max_steps,
                max_chemical_n=args.max_chemical_n,
                curriculum_enabled=True,
                curriculum_seed=curriculum_seed,
                curriculum_train_fraction=args.curriculum_train_fraction,
                is_train=args.is_train,
            )
            for record in episode_records:
                len_all_records += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Total state-action records collected: {len_all_records}")


if __name__ == "__main__":
    main()