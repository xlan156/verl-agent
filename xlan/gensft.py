import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from tqdm import trange

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.prompts.discoveryworld import (
    DISCOVERYWORLD_TEMPLATE,
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
)
from agent_system.environments.env_package.discovery.seed import assign_split_seeds
from agent_system.environments.env_package.discovery.helpers import compress_ui_observation


def build_prompt(
    info: Dict[str, Any],
    step_count: int,
    max_steps: int,
    recent_actions: List[str],
    chemical_N: int,
) -> str:
    ui = (info.get("raw_observation") or {}).get("ui", {})
    state_obs = compress_ui_observation(ui)
    step_info = f"Step: {step_count} / {max_steps}"

    if recent_actions:
        memory_actions = "\n".join(recent_actions)
        return DISCOVERYWORLD_TEMPLATE.format(
            chemical_N=chemical_N,
            state_obs=state_obs,
            step_info=step_info,
            memory_actions=memory_actions,
        )

    return DISCOVERYWORLD_TEMPLATE_NO_HIS.format(
        chemical_N=chemical_N,
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
    _, info = env.reset()
    teacher = RulebasedAgentSkill(env)
    skill_agent = CombinatorialChemistrySkill(env)

    records: List[Dict[str, Any]] = []
    done = False

    while not done:
        teacher_skill = teacher.select_skill(info)
        if teacher_skill:
            prompt = build_prompt(
                info,
                env._steps,
                env._max_steps,
                chemical_N=env._chemical_N,
                recent_actions=[str(action) for action in env.action_history[-3:]],
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
                ui = (info.get("raw_observation") or {}).get("ui", {})
                compressed_obs = env.compress_ui_observation(ui)
                f.write("======================\n")
                f.write(f"self.is_key_in_jar: {env.is_key_in_jar}\n")
                f.write(f"used_dispensers: {env.used_dispensers}\n")
                f.write(f"Observation:\n{compressed_obs}\n\n")

        if random.random() < 0.5:
            random_skill = skill_agent.sample_random_skill()
        else:
            random_skill = teacher_skill
        _, _, done, info = env.step(random_skill)

    env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT data from DiscoveryWorld rollouts.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to sample.")
    parser.add_argument("--seed", type=int, default=2, help="Env seed.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max environment steps per episode.")
    parser.add_argument("--is-train", type=bool, default=True, help="To generate training data or validation data.")
    parser.add_argument("--chemical-n", type=int, default=2, help="Total chemical amount.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records: List[Dict[str, Any]] = []
    seed_split = assign_split_seeds(
        base_seed=0,
        train_size=30,
        val_size=30,
        num_chemicals=4,
        min_chemicals=1,
        min_amount=args.chemical_n,
        max_amount=args.chemical_n,
    )
    train_seeds = seed_split.get("train", [])
    val_seeds = seed_split.get("val", [])

    len_all_records = 0
    if args.is_train:
        output_file = Path("sft/skill_sft_data_train.jsonl")
        seeds_to_use = train_seeds
    else:
        output_file = Path("sft/skill_sft_data_val.jsonl")
        seeds_to_use = val_seeds
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for ep in trange(args.episodes):
            episode_seed = random.choice(seeds_to_use)  # Vary seed for each episode to get diverse trajectories
            episode_records = rollout_episode(
                seed=episode_seed,
                max_steps=args.max_steps,
                chemical_N=args.chemical_n,
            )
            for record in episode_records:
                len_all_records += 1
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Total state-action records collected: {len_all_records}")


if __name__ == "__main__":
    main()