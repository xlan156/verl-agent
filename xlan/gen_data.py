import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import trange

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.utils import format_rust_update
from agent_system.environments.prompts.discoveryworld import (
    DISCOVERYWORLD_TEMPLATE,
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
    format_current_chemicals,
    format_key_status,
)
from agent_system.environments.env_package.discovery.seed import build_ordered_seed_pools_by_amount


SFT_think = {
    "move_to_key": "The key is not yet collected and not accessible from my current position.",
    "pick_up_key": "I'm at the key's location now, but it is still not in my inventory.",
    "move_to_jar": "The jar is not collected and I'm still not at the jar's position.",
    "pick_up_jar": "The jar is not collected and I am at the jar's location.",
    "put_key_in_jar": "The key is currently not in the jar. To apply chemicals to the key, it must be placed in the jar first.",
    "use_dispenser_A_on_jar": "The jar is collected and the key is in the jar. Now I need to apply one chemical to reduce the rust level of the key.",
    "use_dispenser_B_on_jar": "The jar is collected and the key is in the jar. Now I need to apply one chemical to reduce the rust level of the key.",
    "use_dispenser_C_on_jar": "The jar is collected and the key is in the jar. Now I need to apply one chemical to reduce the rust level of the key.",
    "use_dispenser_D_on_jar": "The jar is collected and the key is in the jar. Now I need to apply one chemical to reduce the rust level of the key.",
    "remove_chemical_A": "The chemical amount is excessive while the key is still rusted.",
    "remove_chemical_B": "The chemical amount is excessive while the key is still rusted.",
    "remove_chemical_C": "The chemical amount is excessive while the key is still rusted.",
    "remove_chemical_D": "The chemical amount is excessive while the key is still rusted.",
    "wash_jar": "The chemical solution in the jar is not correct, so I need to wash it to apply different chemicals again.",
    "open_door": "The key has been derusted, so I can now open the door.",
}


def build_prompt(
    state_obs: str,
    step_count: int,
    max_steps: int,
    action_history: List[Dict[str, Any]],
    max_chemical_n: int,
    chemical_dict: Dict[str, int] | None = None,
    key_rust_status: Any = None,
) -> str:
    step_info = f"Step: {step_count} / {max_steps}"
    chemical_state = format_current_chemicals(chemical_dict, max_chemical_n)
    key_state = format_key_status(key_rust_status)

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
            chemical_state=chemical_state,
            key_state=key_state,
            state_obs=state_obs,
            step_info=step_info,
            memory_actions=memory_actions,
        )

    return DISCOVERYWORLD_TEMPLATE_NO_HIS.format(
        max_chemical_n=max_chemical_n,
        chemical_state=chemical_state,
        key_state=key_state,
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
    chemical_total = sum((info.get("chemical_dict") or {}).values())
    if (
        chemical_total > 0
        or bool(info.get("has_key"))
        or bool(info.get("has_jar"))
        or bool(info.get("is_key_in_jar"))
        or info.get("key_rust_status") == "no rust"
    ):
        env.close()
        raise RuntimeError(f"episode did not start from scratch: seed={seed}, info={info}")
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
                step_count=env._steps,
                max_steps=env._max_steps,
                action_history=action_history,
                max_chemical_n=env._max_chemical_n,
                chemical_dict=info.get("chemical_dict"),
                key_rust_status=info.get("key_rust_status"),
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
                            "content": f"<think>{SFT_think.get(teacher_skill, 'No valid skill selected.')}</think>\n<action>{teacher_skill}</action>",
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


def extract_target_action(response: str) -> str:
    marker_start = "<action>"
    marker_end = "</action>"
    if marker_start not in response or marker_end not in response:
        raise ValueError(f"Response does not contain an action block: {response}")
    return response.split(marker_start, 1)[1].split(marker_end, 1)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT data from DiscoveryWorld rollouts.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to sample.")
    parser.add_argument("--seed", type=int, default=2, help="Base env seed used for rollout scenarios.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max environment steps per episode.")
    parser.add_argument("--is-train", action="store_true", help="Generate training data (default: validation data).")
    parser.add_argument("--max-chemical-n", type=int, default=2, help="Total chemical amount.")
    parser.add_argument("--target-train-fraction", type=float, default=0.8, help="Train fraction for the ordered target pool.")
    parser.add_argument("--output-file", type=str, default=None, help="Output parquet file path.")
    parser.add_argument("--append", action="store_true", help="Append to an existing parquet file instead of overwriting it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records: List[Dict[str, Any]] = []
    seed_pools = build_ordered_seed_pools_by_amount(
        max_amount=args.max_chemical_n,
        num_chemicals=4,
        min_chemicals=1,
        train_fraction=args.target_train_fraction,
    )
    selected_split = "train" if args.is_train else "val"
    seeds_to_use = list(seed_pools.get(args.max_chemical_n, {}).get(selected_split, []))
    if not seeds_to_use:
        seeds_to_use = [int(args.seed)]

    if args.output_file is not None:
        output_file = Path(args.output_file)
    elif args.is_train:
        output_file = Path("sft/skill_sft_data_train.parquet")
    else:
        output_file = Path("sft/skill_sft_data_val.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for ep in trange(args.episodes):
        episode_seed = seeds_to_use[ep % len(seeds_to_use)]
        episode_records = rollout_episode(
            seed=episode_seed,
            max_steps=args.max_steps,
            max_chemical_n=args.max_chemical_n,
            is_train=args.is_train,
        )
        for step_idx, record in enumerate(episode_records):
            messages = record["messages"]
            prompt = messages[0]["content"]
            response = messages[1]["content"]
            target_action = extract_target_action(response)
            rows.append(
                {
                    "data_source": "discoveryworld_action",
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "ability": "agent_action",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": target_action,
                    },
                    "response": response,
                    "extra_info": {
                        "split": selected_split,
                        "episode_idx": ep,
                        "step_idx": step_idx,
                        "seed": episode_seed,
                        "max_chemical_n": args.max_chemical_n,
                        "teacher_response": response,
                    },
                }
            )

    df = pd.DataFrame(rows)
    if args.append and output_file.exists():
        old_df = pd.read_parquet(output_file)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_parquet(output_file, index=False)

    print(f"Total state-action records collected: {len(rows)}")
    print(f"Wrote parquet: {output_file}")


if __name__ == "__main__":
    main()
