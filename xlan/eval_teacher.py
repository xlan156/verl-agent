from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_package.discovery.curriculum import (  # noqa: E402
    build_stage_pools,
    format_chemical_state,
    plan_stratified_batch,
)
from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv  # noqa: E402
from agent_system.environments.env_package.discovery.rule_based_agent import (  # noqa: E402
    RulebasedAgentSkill,
)


@dataclass
class EpisodeResult:
    split: str
    index: int
    seed: int
    curriculum_state: Tuple[int, ...]
    won: bool
    steps: int
    final_score: float
    final_chemical_dict: Dict[str, int]
    actions: List[Optional[str]]
    teacher_skill_counts: Dict[str, int]
    error: Optional[str] = None


def parse_mix_ratios(raw: str) -> Tuple[float, float, float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--mix-ratios must contain three comma-separated floats")
    return (values[0], values[1], values[2])


def create_env(seed: int, max_steps: int, max_chemical_n: int, save_frames: bool = False) -> DiscoveryWorldEnv:
    return DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Challenge",
        seed=seed,
        max_steps=max_steps,
        max_chemical_n=max_chemical_n,
        save_frames=save_frames,
    )


def plan_states(
    split: str,
    size: int,
    max_chemical_n: int,
    train_fraction: float,
    curriculum_seed: int,
    mix_ratios: Sequence[float],
) -> List[Tuple[int, ...]]:
    if size <= 0:
        return []

    stage_pools = build_stage_pools(
        max_chemical_n=max_chemical_n,
        num_chemicals=4,
        train_fraction=train_fraction,
        seed=curriculum_seed,
    )
    return plan_stratified_batch(
        stage=max_chemical_n,
        batch_size=size,
        stage_pools=stage_pools,
        split=split,
        mix_ratios=mix_ratios,
        seed=curriculum_seed + (0 if split == "train" else 10_000),
    )


def rollout_teacher(
    split: str,
    index: int,
    seed: int,
    curriculum_state: Tuple[int, ...],
    max_steps: int,
    max_chemical_n: int,
    verbose: bool = False,
) -> EpisodeResult:
    random.seed(seed)
    env = create_env(seed=seed, max_steps=max_steps, max_chemical_n=max_chemical_n)
    actions: List[Optional[str]] = []

    try:
        _, info = env.reset({"curriculum_state": curriculum_state})
        teacher = RulebasedAgentSkill(env)
        done = False

        while not done:
            teacher_skill = teacher.select_skill(info)
            actions.append(teacher_skill)
            _, _, done, info = env.step(teacher_skill)

            if verbose:
                print(
                    f"[{split} #{index:04d}] step={env._steps:02d} "
                    f"state={format_chemical_state(curriculum_state)} "
                    f"skill={teacher_skill} won={info.get('won')}",
                    flush=True,
                )

        return EpisodeResult(
            split=split,
            index=index,
            seed=seed,
            curriculum_state=tuple(curriculum_state),
            won=bool(info.get("won", False)),
            steps=int(env._steps),
            final_score=float(info.get("score_normalized", 0.0)),
            final_chemical_dict=dict(info.get("chemical_dict", {})),
            actions=actions,
            teacher_skill_counts=dict(Counter(action for action in actions if action)),
        )
    except Exception as exc:
        return EpisodeResult(
            split=split,
            index=index,
            seed=seed,
            curriculum_state=tuple(curriculum_state),
            won=False,
            steps=int(getattr(env, "_steps", 0)),
            final_score=0.0,
            final_chemical_dict={},
            actions=actions,
            teacher_skill_counts=dict(Counter(action for action in actions if action)),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        env.close()


def summarize(split: str, results: List[EpisodeResult]) -> Dict[str, Any]:
    total = len(results)
    success = sum(1 for result in results if result.won)
    errors = sum(1 for result in results if result.error)
    avg_steps = sum(result.steps for result in results) / total if total else 0.0
    avg_score = sum(result.final_score for result in results) / total if total else 0.0
    return {
        "split": split,
        "total": total,
        "success": success,
        "success_rate": success / total if total else 0.0,
        "errors": errors,
        "avg_steps": avg_steps,
        "avg_final_score": avg_score,
    }


def print_summary(summary: Dict[str, Any], results: List[EpisodeResult], show_failures: int) -> None:
    print(
        f"{summary['split']}: "
        f"{summary['success']}/{summary['total']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"avg_steps={summary['avg_steps']:.2f} "
        f"avg_final_score={summary['avg_final_score']:.3f} "
        f"errors={summary['errors']}",
        flush=True,
    )

    failures = [result for result in results if not result.won]
    for result in failures[:show_failures]:
        error = f" error={result.error}" if result.error else ""
        print(
            f"  failure #{result.index}: seed={result.seed} "
            f"state={format_chemical_state(result.curriculum_state)} "
            f"steps={result.steps} score={result.final_score:.3f}{error}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the DiscoveryWorld rule-based teacher on curriculum.py pools.",
    )
    parser.add_argument("--train_size", "--train-size", type=int, default=100)
    parser.add_argument("--val_size", "--val-size", type=int, default=100)
    parser.add_argument("--max-chemical-n", "--stage", dest="max_chemical_n", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="Base environment and teacher RNG seed.")
    parser.add_argument("--curriculum-seed", type=int, default=None)
    parser.add_argument("--curriculum-train-fraction", type=float, default=0.7)
    parser.add_argument("--mix-ratios", type=parse_mix_ratios, default=(0.7, 0.2, 0.1))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL path for per-episode results.")
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curriculum_seed = args.seed if args.curriculum_seed is None else int(args.curriculum_seed)
    all_results: List[EpisodeResult] = []

    for split, size in (("train", args.train_size), ("val", args.val_size)):
        states = plan_states(
            split=split,
            size=size,
            max_chemical_n=args.max_chemical_n,
            train_fraction=args.curriculum_train_fraction,
            curriculum_seed=curriculum_seed,
            mix_ratios=args.mix_ratios,
        )
        split_results = [
            rollout_teacher(
                split=split,
                index=index,
                seed=args.seed + (0 if split == "train" else 1_000_000) + index,
                curriculum_state=state,
                max_steps=args.max_steps,
                max_chemical_n=args.max_chemical_n,
                verbose=args.verbose,
            )
            for index, state in enumerate(states)
        ]
        all_results.extend(split_results)
        print_summary(summarize(split, split_results), split_results, args.show_failures)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        print(f"Wrote per-episode results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
