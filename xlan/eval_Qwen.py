import argparse
import json
import os
from typing import Any, Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.projection import (
    discoveryworld_projection,
)
from agent_system.environments.prompts.discoveryworld import (
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
    DISCOVERYWORLD_TEMPLATE,
    format_current_chemicals,
)
from agent_system.memory import SimpleMemory


def call_local_sft(
    prompt: str,
    model,
    tokenizer,
    temperature: float = 0.7,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, top_p=0.9, temperature=temperature, max_new_tokens=512)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])


def build_prompt(
    text_obs: str,
    info: Dict[str, Any],
    mem: Optional[SimpleMemory] = None,
    history_length: int = 0,
) -> str:

    max_chemical_n = int(info.get("max_chemical_n", 2) or 2)
    step_count = len(mem) if mem is not None else 0
    step_info = f"Step: {step_count}"
    chemical_state = format_current_chemicals(info.get("chemical_dict"), max_chemical_n)
    state_obs = text_obs.replace("\\n", "\n")

    if len(mem) <= 0:
        prompt = DISCOVERYWORLD_TEMPLATE_NO_HIS.format(
            max_chemical_n=max_chemical_n,
            chemical_state=chemical_state,
            state_obs=state_obs,
            step_info=step_info,
        )
        return prompt

    memory_contexts, valid_lens = mem.fetch(
        history_length,
        obs_key="text_obs",
        action_key="action",
    )
    history_block = memory_contexts[0]
    history_len = valid_lens[0]

    prompt = DISCOVERYWORLD_TEMPLATE.format(
        max_chemical_n=max_chemical_n,
        chemical_state=chemical_state,
        state_obs=state_obs,
        step_info=step_info,
        memory_actions=history_block,
    )
    return prompt


def inference_one_step(
    env: DiscoveryWorldEnv,
    model,
    tokenizer,
    max_env_steps: int,
    temperature: float,
    history_length: int,
) -> Tuple[int, float]:
    """Run a single episode with the SFT policy.

    Returns (steps_taken, cumulative_reward).
    """
    text_obs, info = env.reset()
    done = False
    step_idx = 0
    cum_reward = 0.0

    mem = SimpleMemory()
    mem.reset(batch_size=1)

    while not done and step_idx < max_env_steps:
        prompt = build_prompt(text_obs, info, mem=mem, history_length=history_length)
        raw_output = call_local_sft(
            prompt, model=model, tokenizer=tokenizer, temperature=temperature
        )

        print("\n===== STEP", step_idx, "=====")
        print("Prompt (truncated):")
        print(prompt[:800], "...\n")
        print("LLM:")
        print(raw_output)
        print("\n")

        projected_actions, _valids = discoveryworld_projection([raw_output])
        env_action = projected_actions[0]

        print("Projected env action:", env_action)

        mem.store({"text_obs": [text_obs], "action": [env_action]})

        text_obs, reward, done, info = env.step(env_action)
        cum_reward += float(reward)

        print("Reward:", reward, "Done:", done)

        step_idx += 1

    return step_idx, cum_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local SFT DiscoveryWorld policy in the environment.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sft/models/SFT-Qwen2.5-1.5B-Instruct-merged",
        help=(
            "HF model directory to load (default: merged SFT model). "
            "Run sft/merge_sft_lora.py first if you only have LoRA adapters."
        ),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_env_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario_name", type=str, default="Combinatorial Chemistry")
    parser.add_argument("--difficulty", type=str, default="Easy")
    parser.add_argument(
        "--history_length",
        type=int,
        default=3,
        help=(
            "If > 0, use DISCOVERYWORLD_TEMPLATE with up to this many "
            "past observation-action pairs as history; otherwise use the "
            "no-history template."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    base_model_path = "Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",        # puts model on GPU
        dtype=torch.float16,
        trust_remote_code=True
    )

    env = DiscoveryWorldEnv(
        seed=args.seed,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_steps=args.max_env_steps,
        thread_id=0,
    )

    total_steps = 0
    total_reward = 0.0

    for ep in range(args.episodes):
        print("\n==============================")
        print(f"Starting episode {ep}")
        steps, reward = inference_one_step(
            env,
            model=model,
            tokenizer=tokenizer,
            max_env_steps=args.max_env_steps,
            temperature=args.temperature,
            history_length=args.history_length,
        )
        print(f"Episode {ep} finished: steps={steps}, reward={reward}")
        total_steps += steps
        total_reward += reward

    if args.episodes > 0:
        print(f"Avg steps per episode: {total_steps / args.episodes:.2f}")
        print(f"Avg reward per episode: {total_reward / args.episodes:.2f}")


if __name__ == "__main__":
    main()
