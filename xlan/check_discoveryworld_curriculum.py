from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from agent_system.environments.env_package.discovery.curriculum import (
    normalize_chemical_state,
    state_to_dict,
)
from agent_system.environments.env_package.discovery.curriculum import (
    build_stage_pools,
    plan_stratified_batch,
)
from agent_system.environments.env_package.discovery.envs import DiscoveryWorldWorker
from agent_system.environments.env_package.discovery.seed import assign_split_seeds


def _parse_state(values: Sequence[int]) -> Tuple[int, ...]:
    return normalize_chemical_state(list(values), num_chemicals=4)


def _split_seeds(
    base_seed: int,
    train_size: int,
    val_size: int,
    chemical_n: int,
) -> Dict[str, List[int]]:
    return assign_split_seeds(
        base_seed=base_seed,
        train_size=train_size,
        val_size=val_size,
        num_chemicals=4,
        min_chemicals=1,
        min_amount=chemical_n,
        max_amount=chemical_n,
    )


def _is_strict_substate(init_state: Tuple[int, ...], goal_state: Tuple[int, ...]) -> bool:
    if len(init_state) != len(goal_state):
        return False
    leq = all(init <= goal for init, goal in zip(init_state, goal_state))
    strict = any(init < goal for init, goal in zip(init_state, goal_state))
    return leq and strict


def _bucket_name(state: Tuple[int, ...], goal_state: Tuple[int, ...]) -> str:
    state_total = sum(state)
    goal_total = sum(goal_state)
    if state_total == goal_total - 1:
        return "current"
    if state_total == goal_total - 2:
        return "previous"
    return "earlier"


def _print_split_summary(label: str, seeds: List[int]) -> None:
    print(f"[{label}] seeds={seeds}")


def _run_split_probe(
    label: str,
    split: str,
    batch_size: int,
    base_seed: int,
    train_size: int,
    val_size: int,
    scenario_name: str,
    difficulty: str,
    max_chemical_n: int,
) -> bool:
    split_seeds = _split_seeds(
        base_seed=base_seed,
        train_size=train_size,
        val_size=val_size,
        chemical_n=max_chemical_n,
    )
    seeds = split_seeds[split][:batch_size]
    env_kwargs = {
        "scenario_name": scenario_name,
        "difficulty": difficulty,
        "max_steps": 40,
        "train_size": batch_size,
        "val_size": batch_size,
        "chemical_N": max_chemical_n,
        "max_chemical_N": max_chemical_n,
        "curriculum_enabled": True,
        "curriculum_train_fraction": 0.7,
        "curriculum_mix_ratios": (0.7, 0.2, 0.1),
        "curriculum_seed": base_seed,
        "is_train": split == "train",
    }

    print(f"\n[{label}] split={split} batch_size={batch_size}")
    _print_split_summary(label, seeds)

    # Create workers (env objects) but do not prepare their states yet.
    workers = [
        DiscoveryWorldWorker(seed=seed, env_kwargs=env_kwargs, thread_id=idx)
        for idx, seed in enumerate(seeds)
    ]

    # If curriculum enabled, preload each worker's solution_state by initializing
    # the scenario (without preparing jars), then compute available strict-substate
    # pools per-worker and assign buckets to meet batch quotas.
    if env_kwargs.get("curriculum_enabled") and "curriculum_state" not in env_kwargs:
        rng = __import__("random").Random(env_kwargs.get("curriculum_seed", 0))
        # preload scenarios to read their solution dicts
        per_worker_solution = []
        for w in workers:
            try:
                w._env._init_api()
                sol = w._env._get_chemical_solution_state()
            except Exception:
                sol = None
            per_worker_solution.append(sol)

        # Build per-worker pools of strict substates (train/val) using solution-specific pools
        per_worker_buckets = []
        for sol in per_worker_solution:
            if not sol:
                per_worker_buckets.append({"current": [], "previous": [], "earlier": [], "all": []})
                continue
            sol_pools = build_stage_pools(max_chemical_n=max_chemical_n, num_chemicals=4, train_fraction=env_kwargs.get("curriculum_train_fraction", 0.7), seed=env_kwargs.get("curriculum_seed", 0))
            # Instead of stage-level pools, derive from solution substates
            from agent_system.environments.env_package.discovery.curriculum import build_solution_curriculum_pools, enumerate_substates

            sol_states = build_solution_curriculum_pools(solution_state=sol, num_chemicals=4, train_fraction=env_kwargs.get("curriculum_train_fraction", 0.7), seed=env_kwargs.get("curriculum_seed", 0), include_empty=True)
            split_pool = sol_states.get(split, [])
            total_groups = {}
            for s in split_pool:
                total_groups.setdefault(sum(s), []).append(s)
            if total_groups:
                max_total = max(total_groups)
                current = total_groups.get(max_total, [])
                previous = total_groups.get(max_total - 1, [])
                earlier = []
                for tot in sorted(total_groups):
                    if tot < max_total - 1:
                        earlier.extend(total_groups[tot])
            else:
                current = previous = earlier = []
            per_worker_buckets.append({"current": current, "previous": previous, "earlier": earlier, "all": split_pool})

        # Determine desired counts per bucket
        mix = env_kwargs.get("curriculum_mix_ratios", (0.7, 0.2, 0.1))
        desired = [int(round(len(workers) * mix[0])), int(round(len(workers) * mix[1])), int(round(len(workers) * mix[2]))]
        # adjust to match total
        diff = len(workers) - sum(desired)
        if diff != 0:
            desired[0] += diff

        assigned = [None] * len(workers)
        # Greedy assign workers to buckets: current -> previous -> earlier
        def try_assign(bucket_name, quota):
            for idx in rng.sample(list(range(len(workers))), len(workers)):
                if assigned[idx] is None and per_worker_buckets[idx].get(bucket_name):
                    if quota[0] <= 0:
                        break
                    assigned[idx] = bucket_name
                    quota[0] -= 1

        quotas = [desired[0], desired[1], desired[2]]
        try_assign("current", [quotas[0]])
        try_assign("previous", [quotas[1]])
        try_assign("earlier", [quotas[2]])

        # Fill any unassigned workers with any available bucket
        for idx in range(len(workers)):
            if assigned[idx] is None:
                for b in ("current", "previous", "earlier"):
                    if per_worker_buckets[idx].get(b):
                        assigned[idx] = b
                        break
                if assigned[idx] is None and per_worker_buckets[idx].get("all"):
                    assigned[idx] = "all"

        # Sample one state per worker from their assigned bucket
        worker_env_kwargs = []
        for idx, w in enumerate(workers):
            bucket = assigned[idx]
            pools = per_worker_buckets[idx]
            candidate_pool = pools.get(bucket) or pools.get("all") or []
            if not candidate_pool:
                chosen = None
            else:
                chosen = rng.choice(candidate_pool)
            kw = dict(env_kwargs)
            if chosen is not None:
                kw["curriculum_state"] = chosen
            worker_env_kwargs.append(kw)
    else:
        worker_env_kwargs = [dict(env_kwargs) for _ in seeds]

    # Recreate workers with the prepared per-worker kwargs so resets will use the planned states
    for i, w in enumerate(workers):
        w.close()
    workers = [
        DiscoveryWorldWorker(seed=seed, env_kwargs=worker_env_kwargs[idx], thread_id=idx)
        for idx, seed in enumerate(seeds)
    ]

    all_ok = True
    try:
        bucket_counts = Counter()
        for idx, worker in enumerate(workers):
            obs, info = worker.reset()
            debug_after_reset = worker.debug_state()
            goal_state = tuple(info.get("chemical_solution_state") or debug_after_reset.get("chemical_solution_state") or ())
            init_state = tuple(info.get("curriculum_state") or ())
            goal_dict = state_to_dict(goal_state)
            init_dict = state_to_dict(init_state)
            ui_chemical_dict = dict(info.get("chemical_dict") or {})
            bucket = _bucket_name(init_state, goal_state) if goal_state else "unknown"
            bucket_counts[bucket] += 1

            print(f"[{label}] worker={idx}")
            print(f"  seed={debug_after_reset['seed']}")
            print(f"  goal_state={goal_state}")
            print(f"  goal_dict={goal_dict}")
            print(f"  init_state={init_state}")
            print(f"  init_state_text={info.get('curriculum_state_text')}")
            print(f"  init_chemical_dict={ui_chemical_dict}")
            print(f"  prepared_chemical_dict={init_dict}")
            print(f"  init_bucket={bucket}")
            print(f"  goal_state_text={info.get('chemical_solution_state_text')}")
            print(f"  goal_chemical_dict={state_to_dict(goal_state)}")
            print(f"  debug_curriculum_state={debug_after_reset.get('curriculum_state')}")
            print(f"  debug_curriculum_state_text={debug_after_reset.get('curriculum_state_text')}")
            print(f"  debug_goal_state={debug_after_reset.get('chemical_solution_state')}")
            print(f"  debug_goal_state_text={debug_after_reset.get('chemical_solution_state_text')}")

            if not goal_state:
                all_ok = False
            if tuple(info.get("curriculum_state") or ()) != init_state:
                all_ok = False
            if not _is_strict_substate(init_state, goal_state):
                all_ok = False
            if init_state == goal_state:
                all_ok = False
            if tuple(debug_after_reset.get("curriculum_state") or ()) != init_state:
                all_ok = False
            if tuple(debug_after_reset.get("chemical_solution_state") or ()) != goal_state:
                all_ok = False
            if obs is None:
                all_ok = False

        total = max(len(workers), 1)
        bucket_ratios = {key: bucket_counts[key] / total for key in ("current", "previous", "earlier")}
        print(f"[{label}] init_bucket_counts={dict(bucket_counts)}")
        print(f"[{label}] init_bucket_ratios={bucket_ratios}")
        print(f"[{label}] worker_prepare_ok={all_ok}")
    finally:
        for worker in workers:
            worker.close()

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DiscoveryWorld curriculum assignment and init states.")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--env-num", type=int, default=2)
    parser.add_argument("--group-n", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--scenario-name", type=str, default="Combinatorial Chemistry")
    parser.add_argument("--difficulty", type=str, default="Challenge")
    parser.add_argument("--max-chemical-n", type=int, default=3)
    parser.add_argument("--curriculum-stage", type=int, default=3)
    args = parser.parse_args()

    stage = max(1, min(args.curriculum_stage, args.max_chemical_n))
    print("Configuration:")
    print(f"  base_seed={args.base_seed}")
    print(f"  max_chemical_n={args.max_chemical_n}")
    print(f"  curriculum_stage={stage}")
    print(f"  train_size={args.train_size}")
    print(f"  val_size={args.val_size}")
    print(f"  scenario_name={args.scenario_name}")
    print(f"  difficulty={args.difficulty}")

    singleton_ok = _run_split_probe(
        label="singleton",
        split="train",
        batch_size=1,
        base_seed=args.base_seed,
        train_size=1,
        val_size=1,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_chemical_n=args.max_chemical_n,
    ) and _run_split_probe(
        label="singleton",
        split="val",
        batch_size=1,
        base_seed=args.base_seed,
        train_size=1,
        val_size=1,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_chemical_n=args.max_chemical_n,
    )

    multi_train_size = args.train_size if args.train_size != 1 else 3
    multi_val_size = args.val_size if args.val_size != 1 else 2
    multi_ok = _run_split_probe(
        label="multi",
        split="train",
        batch_size=multi_train_size,
        base_seed=args.base_seed,
        train_size=multi_train_size,
        val_size=multi_val_size,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_chemical_n=args.max_chemical_n,
    ) and _run_split_probe(
        label="multi",
        split="val",
        batch_size=multi_val_size,
        base_seed=args.base_seed,
        train_size=multi_train_size,
        val_size=multi_val_size,
        scenario_name=args.scenario_name,
        difficulty=args.difficulty,
        max_chemical_n=args.max_chemical_n,
    )

    split_seeds = _split_seeds(
        base_seed=args.base_seed,
        train_size=multi_train_size,
        val_size=multi_val_size,
        chemical_n=args.max_chemical_n,
    )
    no_cross_ok = not (set(split_seeds["train"]) & set(split_seeds["val"]))
    print("Train/val seed overlap ok:", no_cross_ok)
    print("Singleton probe ok:", singleton_ok)
    print("Multi probe ok:", multi_ok)
    print("Overall ok:", singleton_ok and multi_ok and no_cross_ok)


if __name__ == "__main__":
    main()
