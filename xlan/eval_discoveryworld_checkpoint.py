#!/usr/bin/env python3
"""Paper-style evaluation of a VERL checkpoint on fixed DiscoveryWorld seeds."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import ray
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from agent_system.environments.env_package.discovery.seed import build_ordered_seed_pools_by_amount
from verl.trainer.main_ppo import run_ppo


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def make_eval_parquet(path: Path, size: int) -> None:
    rows = [
        {
            "data_source": "text",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "extra_info": {"split": "test", "index": index},
        }
        for index in range(size)
    ]
    Dataset.from_list(rows).to_parquet(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Maximum --eval-size for the val split with the default "
            "--target-train-fraction=0.8: "
            "N=1: 1 (val seeds [3]); "
            "N=2: 2 (val seeds [8, 9]); "
            "N=3: 4 (val seeds [16, 17, 18, 19]); "
            "N=4: 7 (val seeds [28, 29, 30, 31, 32, 33, 34]). "
            "Changing --target-train-fraction changes these limits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="global_step_N or best_val_success checkpoint directory")
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base HF model used to initialize the checkpoint architecture")
    parser.add_argument(
        "--eval-size",
        "--val-size",
        dest="eval_size",
        type=int,
        default=2,
        help="Number of distinct seeds selected from --eval-split; --val-size is a backward-compatible alias",
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "train"),
        default="val",
        help="Evaluate seeds from the curriculum-disabled validation or training pool",
    )
    parser.add_argument("--rollouts-per-seed", type=int, default=5)
    parser.add_argument("--max-chemical-n", type=int, default=2)
    parser.add_argument("--target-train-fraction", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--env-variant", default="original")
    parser.add_argument("--seed", type=int, default=0, help="Policy sampling seed; environment seeds come from --eval-split")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-cpus-per-env", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path")
    args = parser.parse_args()
    if args.eval_size <= 0 or args.rollouts_per_seed <= 0:
        parser.error("--eval-size and --rollouts-per-seed must be positive")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "actor").is_dir():
        raise FileNotFoundError(f"Checkpoint actor directory not found: {checkpoint / 'actor'}")

    pools = build_ordered_seed_pools_by_amount(
        max_amount=args.max_chemical_n,
        train_fraction=args.target_train_fraction,
    )
    full_eval_pool = pools[args.max_chemical_n][args.eval_split]
    if args.eval_size > len(full_eval_pool):
        raise ValueError(
            f"eval_size={args.eval_size} exceeds the fixed {args.eval_split} pool size "
            f"{len(full_eval_pool)}: {full_eval_pool}"
        )

    selected_seeds = full_eval_pool[: args.eval_size]
    total_episodes = len(selected_seeds) * args.rollouts_per_seed

    config_dir = str((Path(__file__).resolve().parents[1] / "verl/trainer/config").resolve())
    with tempfile.TemporaryDirectory(prefix="dw_checkpoint_eval_") as tmp:
        val_file = Path(tmp) / "val.parquet"
        make_eval_parquet(val_file, total_episodes)
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="ppo_trainer")
        with open_dict(cfg):
            cfg.data.train_files = str(val_file)
            cfg.data.val_files = str(val_file)
            cfg.data.train_batch_size = total_episodes
            cfg.data.val_batch_size = total_episodes
            cfg.data.max_prompt_length = 2048
            cfg.data.max_response_length = 256
            cfg.data.return_raw_chat = True
            cfg.actor_rollout_ref.model.path = args.model_path
            cfg.actor_rollout_ref.model.use_remove_padding = True
            cfg.actor_rollout_ref.actor.strategy = "fsdp"
            cfg.actor_rollout_ref.actor.use_kl_loss = False
            # val_only never updates the actor or computes rollout log-probs,
            # but RayPPOTrainer/worker initialization still validates these.
            cfg.actor_rollout_ref.actor.ppo_mini_batch_size = args.num_gpus
            cfg.actor_rollout_ref.actor.ppo_micro_batch_size = None
            cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
            cfg.actor_rollout_ref.rollout.name = "vllm"
            cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size = None
            cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
            cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = args.num_gpus
            cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
            cfg.actor_rollout_ref.rollout.enable_chunked_prefill = False
            cfg.actor_rollout_ref.rollout.enforce_eager = False
            cfg.actor_rollout_ref.rollout.free_cache_engine = False
            cfg.actor_rollout_ref.rollout.val_kwargs.do_sample = True
            cfg.actor_rollout_ref.rollout.val_kwargs.temperature = args.temperature
            cfg.actor_rollout_ref.rollout.val_kwargs.top_p = args.top_p
            cfg.algorithm.adv_estimator = "grpo"
            cfg.env.env_name = "discoveryworld"
            cfg.env.seed = args.seed
            cfg.env.max_steps = args.max_steps
            cfg.env.rollout.n = 1
            cfg.env.resources_per_worker.num_cpus = args.num_cpus_per_env
            cfg.env.discoveryworld = {
                "scenario_name": "Combinatorial Chemistry",
                "difficulty": "Challenge",
                "max_chemical_n": args.max_chemical_n,
                "target_train_fraction": args.target_train_fraction,
                "eval_seed_pool": selected_seeds,
                "env_variant": args.env_variant,
                "save_frames": False,
            }
            cfg.trainer.logger = ["console"]
            cfg.trainer.n_gpus_per_node = args.num_gpus
            cfg.trainer.nnodes = 1
            cfg.trainer.val_before_train = True
            cfg.trainer.val_only = True
            cfg.trainer.resume_mode = "resume_path"
            cfg.trainer.resume_from_path = str(checkpoint)
            cfg.trainer.save_best_val_success = False
            cfg.trainer.log_llm_steps = False
            cfg.ray_init.address = "auto" if "RAY_ADDRESS" in __import__("os").environ else None

        metrics = run_ppo(cfg)

    rate = float(metrics["val/success_rate"])
    successes = int(round(rate * total_episodes))
    ci_low, ci_high = wilson_interval(successes, total_episodes)
    per_seed = {
        str(seed): float(metrics[f"val/success_rate_by_seed/{seed}"])
        for seed in selected_seeds
    }
    macro_rate = sum(per_seed.values()) / len(per_seed)
    summary = {
        "checkpoint": str(checkpoint),
        "seed_pool": selected_seeds,
        "rollouts_per_seed": args.rollouts_per_seed,
        "episodes": total_episodes,
        "successes": successes,
        "overall_micro_success_rate": rate,
        "macro_success_rate_across_seeds": macro_rate,
        "per_seed_success_rate": per_seed,
        "overall_micro_success_rate_wilson_95_ci": [ci_low, ci_high],
        "eval_parameters": {
            "checkpoint": str(checkpoint),
            "model_path": args.model_path,
            "eval_split": args.eval_split,
            "eval_size": args.eval_size,
            "rollouts_per_seed": args.rollouts_per_seed,
            "selected_eval_seeds": selected_seeds,
            "max_chemical_n": args.max_chemical_n,
            "target_train_fraction": args.target_train_fraction,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "policy_sampling_seed": args.seed,
            "vllm_gpu_memory_utilization": 0.6,
        },
    }
    print(f"\nPaper-style {args.eval_split}-split evaluation summary")
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
