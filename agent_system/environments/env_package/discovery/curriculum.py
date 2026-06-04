from __future__ import annotations

from itertools import product
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


CHEMICAL_ORDER: Tuple[str, ...] = ("A", "B", "C", "D")


def _to_int_list(state: Any) -> List[int]:
    if state is None:
        return []
    if isinstance(state, dict):
        return [int(state.get(name, 0)) for name in CHEMICAL_ORDER]
    if isinstance(state, (list, tuple)):
        return [int(value) for value in state]
    if isinstance(state, str):
        cleaned = state.strip().strip("[]()")
        if not cleaned:
            return []
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        return [int(part) for part in parts]
    return [int(state)]


def normalize_chemical_state(state: Any, num_chemicals: int = 4) -> Tuple[int, ...]:
    values = _to_int_list(state)
    values = [max(0, int(value)) for value in values[:num_chemicals]]
    if len(values) < num_chemicals:
        values.extend([0] * (num_chemicals - len(values)))
    return tuple(values)


def state_to_dict(state: Any, num_chemicals: int = 4) -> Dict[str, int]:
    normalized = normalize_chemical_state(state, num_chemicals=num_chemicals)
    return {name: normalized[idx] for idx, name in enumerate(CHEMICAL_ORDER[:num_chemicals])}


def solution_dict_to_state(solution: Dict[str, int], num_chemicals: int = 4) -> Tuple[int, ...]:
    aliases = {
        "A": ("A", "Chemical A", "Substance A"),
        "B": ("B", "Chemical B", "Substance B"),
        "C": ("C", "Chemical C", "Substance C"),
        "D": ("D", "Chemical D", "Substance D"),
    }
    lookup = {}
    for canonical, names in aliases.items():
        for name in names:
            lookup[name] = canonical

    normalized_solution = {lookup.get(key, key): int(value) for key, value in solution.items()}
    return normalize_chemical_state(
        [int(normalized_solution.get(name, 0)) for name in CHEMICAL_ORDER[:num_chemicals]],
        num_chemicals=num_chemicals,
    )


def format_chemical_state(state: Any, num_chemicals: int = 4) -> str:
    normalized = normalize_chemical_state(state, num_chemicals=num_chemicals)
    pairs = [f"{name}={normalized[idx]}" for idx, name in enumerate(CHEMICAL_ORDER[:num_chemicals])]
    return "[" + ", ".join(pairs) + "]"


def enumerate_substates(
    target_state: Any,
    num_chemicals: int = 4,
    include_target: bool = False,
    include_empty: bool = False,
) -> List[Tuple[int, ...]]:
    """Enumerate all component-wise substates of a target chemical state."""
    target = normalize_chemical_state(target_state, num_chemicals=num_chemicals)
    states = []
    for candidate in product(*(range(value + 1) for value in target)):
        if not include_empty and all(value == 0 for value in candidate):
            continue
        if not include_target and candidate == target:
            continue
        states.append(tuple(int(value) for value in candidate))

    states.sort(key=lambda candidate: (sum(candidate), candidate))
    return states


def _l1_distance(lhs: Sequence[int], rhs: Sequence[int]) -> int:
    return sum(abs(int(left) - int(right)) for left, right in zip(lhs, rhs))


def enumerate_nearby_init_states(
    target_state: Any,
    num_chemicals: int = 4,
    include_empty: bool = False,
    include_same_total_neighbors: bool = True,
) -> List[Tuple[int, ...]]:
    """Enumerate init-state candidates near a target chemical state.

    The pool contains strict substates and, optionally, same-total one-transfer
    neighbors such as (0, 1, 1, 0) for target (1, 1, 0, 0).
    """
    target = normalize_chemical_state(target_state, num_chemicals=num_chemicals)
    states = set(enumerate_substates(target, num_chemicals=num_chemicals, include_target=False, include_empty=include_empty))

    if include_same_total_neighbors:
        target_total = sum(target)
        for candidate in enumerate_compositions(target_total, num_chemicals=num_chemicals):
            if candidate == target:
                continue
            if not include_empty and all(value == 0 for value in candidate):
                continue
            if _l1_distance(candidate, target) == 2:
                states.add(tuple(int(value) for value in candidate))

    nearby_states = sorted(states, key=lambda candidate: (sum(candidate), candidate))
    return nearby_states


def split_states(
    states: Sequence[Tuple[int, ...]],
    train_fraction: float = 0.7,
    seed: int = 0,
) -> Dict[str, List[Tuple[int, ...]]]:
    shuffled = list(states)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    
    if not shuffled:
        return {"train": [], "val": [], "all": []}
    
    split_index = int(round(len(shuffled) * min(max(train_fraction, 0.0), 1.0)))
    train_states = shuffled[:split_index]
    val_states = shuffled[split_index:]
    
    if not train_states:
        train_states = [shuffled[0]]
    if not val_states:
        val_states = [shuffled[-1]]

    return {
        "train": train_states,
        "val": val_states,
        "all": shuffled,
    }


def enumerate_curriculum_levels(
    max_chemical_n: int,
    num_chemicals: int = 4,
    include_empty: bool = False,
) -> List[Tuple[int, ...]]:
    """Enumerate a compact curriculum from easy to hard for a scalar max amount."""
    max_chemical_n = max(0, int(max_chemical_n))
    levels = [tuple(candidate) for candidate in product(range(max_chemical_n + 1), repeat=num_chemicals)]
    if not include_empty:
        levels = [state for state in levels if any(value != 0 for value in state)]
    levels.sort(key=lambda candidate: (sum(candidate), candidate))
    return levels


def enumerate_compositions(total_amount: int, num_chemicals: int = 4) -> List[Tuple[int, ...]]:
    """Enumerate all non-negative compositions of total_amount over num_chemicals slots."""
    total_amount = max(0, int(total_amount))
    num_chemicals = max(1, int(num_chemicals))

    if num_chemicals == 1:
        return [(total_amount,)]

    compositions: List[Tuple[int, ...]] = []

    def _recurse(prefix: List[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            compositions.append(tuple(prefix + [remaining]))
            return
        for amount in range(remaining + 1):
            _recurse(prefix + [amount], remaining - amount, slots_left - 1)

    _recurse([], total_amount, num_chemicals)
    return compositions


def build_stage_pools(
    max_chemical_n: int,
    num_chemicals: int = 4,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> Dict[int, Dict[str, List[Tuple[int, ...]]]]:
    """Build disjoint train/val pools for each curriculum stage.

    Stage is defined by the total chemical amount. The stage pools are deterministic,
    shuffled per stage, and then split into train/val without overlap.
    """
    max_chemical_n = max(1, int(max_chemical_n))
    train_fraction = min(max(train_fraction, 0.0), 1.0)

    pools: Dict[int, Dict[str, List[Tuple[int, ...]]]] = {}
    for stage in range(1, max_chemical_n + 1):
        stage_states = enumerate_compositions(stage, num_chemicals=num_chemicals)
        stage_rng = random.Random(seed + stage)
        stage_rng.shuffle(stage_states)

        split_index = int(round(len(stage_states) * train_fraction))
        split_index = min(max(split_index, 1 if stage_states else 0), len(stage_states))

        train_states = stage_states[:split_index]
        val_states = stage_states[split_index:]
        pools[stage] = {
            "train": train_states,
            "val": val_states,
            "all": stage_states,
        }

    return pools


def build_solution_curriculum_pools(
    solution_state: Any,
    num_chemicals: int = 4,
    train_fraction: float = 0.7,
    seed: int = 0,
    include_empty: bool = True,
    include_same_total_neighbors: bool = True,
) -> Dict[str, List[Tuple[int, ...]]]:
    """Build a disjoint train/val pool of nearby init states for one solution."""
    states = enumerate_nearby_init_states(
        solution_state,
        num_chemicals=num_chemicals,
        include_empty=include_empty,
        include_same_total_neighbors=include_same_total_neighbors,
    )
    return split_states(states, train_fraction=train_fraction, seed=seed)


def sample_solution_curriculum_state(
    solution_state: Any,
    split: str = "train",
    train_fraction: float = 0.7,
    seed: int = 0,
    num_chemicals: int = 4,
    mix_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    include_empty: bool = True,
    include_same_total_neighbors: bool = True,
) -> Tuple[int, ...]:
    """Sample a nearby init state of the given solution.

    The pool is first split into train/val without overlap, then mixed by
    current / previous / earlier difficulty bands based on total chemical count.
    The current bucket includes same-total one-transfer neighbors when enabled.
    """
    pools = build_solution_curriculum_pools(
        solution_state=solution_state,
        num_chemicals=num_chemicals,
        train_fraction=train_fraction,
        seed=seed,
        include_empty=include_empty,
        include_same_total_neighbors=include_same_total_neighbors,
    )
    split_states_pool = pools.get(split, [])
    if not split_states_pool:
        split_states_pool = pools.get("all", [])
    
    if not split_states_pool:
        raise ValueError(f"No curriculum states available for split={split}")

    split_states_pool = list(split_states_pool)
    total_groups: Dict[int, List[Tuple[int, ...]]] = {}
    for state in split_states_pool:
        total_groups.setdefault(sum(state), []).append(state)

    max_total = max(total_groups)
    current_pool = total_groups.get(max_total, [])
    previous_pool = total_groups.get(max_total - 1, [])
    earlier_pool: List[Tuple[int, ...]] = []
    for total in sorted(total_groups):
        if total < max_total - 1:
            earlier_pool.extend(total_groups[total])

    buckets = [current_pool, previous_pool, earlier_pool]
    weights = list(mix_ratios[:3])
    while len(weights) < 3:
        weights.append(0.0)

    rng = random.Random(seed)
    available = [(bucket, weight) for bucket, weight in zip(buckets, weights) if bucket and weight > 0]
    if not available:
        return rng.choice(split_states_pool)

    total_weight = sum(weight for _, weight in available)
    draw = rng.random() * total_weight
    cumulative = 0.0
    for bucket, weight in available:
        cumulative += weight
        if draw <= cumulative:
            return rng.choice(bucket)

    return rng.choice(available[-1][0])


def flatten_pools(stage_pools: Dict[int, Dict[str, List[Tuple[int, ...]]]], split: str) -> List[Tuple[int, ...]]:
    states: List[Tuple[int, ...]] = []
    for stage in sorted(stage_pools):
        states.extend(stage_pools[stage].get(split, []))
    return states


def sample_mixed_curriculum_state(
    stage: int,
    stage_pools: Dict[int, Dict[str, List[Tuple[int, ...]]]],
    split: str = "train",
    mix_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    seed: Optional[int] = None,
) -> Tuple[int, ...]:
    """Sample a curriculum state using current / previous / earlier stage mixing.

    The default ratio is 70% current stage, 20% previous stage, 10% earlier stages.
    If a bucket is empty, its weight is redistributed across available buckets.
    """
    stage = max(1, int(stage))
    rng = random.Random(seed)

    current_pool = list(stage_pools.get(stage, {}).get(split, []))
    previous_pool = list(stage_pools.get(stage - 1, {}).get(split, [])) if stage - 1 >= 1 else []

    earlier_pool: List[Tuple[int, ...]] = []
    for earlier_stage in range(1, stage - 1):
        earlier_pool.extend(stage_pools.get(earlier_stage, {}).get(split, []))

    buckets = [current_pool, previous_pool, earlier_pool]
    weights = list(mix_ratios[:3])
    while len(weights) < 3:
        weights.append(0.0)

    available = [(bucket, weight) for bucket, weight in zip(buckets, weights) if bucket and weight > 0]
    if not available:
        fallback = current_pool or previous_pool or earlier_pool
        if not fallback:
            fallback = list(stage_pools.get(stage, {}).get("all", []))
        if not fallback:
            raise ValueError(f"No curriculum states available for stage={stage}, split={split}")
        return rng.choice(fallback)

    total_weight = sum(weight for _, weight in available)
    draw = rng.random() * total_weight
    cumulative = 0.0
    for bucket, weight in available:
        cumulative += weight
        if draw <= cumulative:
            return rng.choice(bucket)

    return rng.choice(available[-1][0])


def plan_curriculum_batch(
    stage: int,
    batch_size: int,
    stage_pools: Dict[int, Dict[str, List[Tuple[int, ...]]]],
    split: str = "train",
    mix_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    seed: int = 0,
) -> List[Tuple[int, ...]]:
    """Plan a batch of curriculum states for a given stage and split."""
    rng = random.Random(seed)
    plans: List[Tuple[int, ...]] = []
    for batch_idx in range(max(0, int(batch_size))):
        plans.append(
            sample_mixed_curriculum_state(
                stage=stage,
                stage_pools=stage_pools,
                split=split,
                mix_ratios=mix_ratios,
                seed=rng.randint(0, 2**31 - 1),
            )
        )
    return plans


def plan_stratified_batch(
    stage: int,
    batch_size: int,
    stage_pools: Dict[int, Dict[str, List[Tuple[int, ...]]]],
    split: str = "train",
    mix_ratios: Sequence[float] = (0.7, 0.2, 0.1),
    seed: int = 0,
) -> List[Tuple[int, ...]]:
    """Plan a batch with deterministic stratified counts per bucket.

    This computes exact counts for current/previous/earlier according to `mix_ratios`
    (rounded, adjusted to sum to batch_size), then samples from each bucket using
    the provided RNG. If a bucket is empty, its quota is redistributed to remaining
    non-empty buckets.
    """
    rng = random.Random(seed)
    stage = max(1, int(stage))

    current_pool = list(stage_pools.get(stage, {}).get(split, []))
    previous_pool = list(stage_pools.get(stage - 1, {}).get(split, [])) if stage - 1 >= 1 else []
    earlier_pool: List[Tuple[int, ...]] = []
    for earlier_stage in range(1, stage - 1):
        earlier_pool.extend(stage_pools.get(earlier_stage, {}).get(split, []))

    buckets = [current_pool, previous_pool, earlier_pool]
    weights = list(mix_ratios[:3])
    while len(weights) < 3:
        weights.append(0.0)

    # Determine desired counts (rounded)
    raw_counts = [max(0, int(round(batch_size * w))) for w in weights]
    # Adjust sum to match batch_size by adding/subtracting from largest weight index
    total_assigned = sum(raw_counts)
    if total_assigned != batch_size:
        diff = batch_size - total_assigned
        # find index of largest weight to absorb the difference
        largest_idx = int(max(range(len(weights)), key=lambda i: weights[i]))
        raw_counts[largest_idx] += diff

    # If a bucket is empty, set its count to 0 and redistribute its quota
    counts = list(raw_counts)
    available_idxs = [i for i, b in enumerate(buckets) if b]
    if not available_idxs:
        raise ValueError(f"No available buckets to sample for stage={stage}, split={split}")

    # Redistribute counts for empty buckets
    for i in range(len(buckets)):
        if not buckets[i] and counts[i] > 0:
            q = counts[i]
            counts[i] = 0
            # distribute q across available_idxs proportionally to weights
            weight_sum = sum(weights[j] for j in available_idxs)
            for j in available_idxs:
                add = int(round(q * (weights[j] / weight_sum))) if weight_sum > 0 else q // len(available_idxs)
                counts[j] += add

    # Fix any rounding drift to ensure sum == batch_size
    while sum(counts) < batch_size:
        # assign to largest-weight available bucket
        target = max(available_idxs, key=lambda i: weights[i])
        counts[target] += 1
    while sum(counts) > batch_size:
        target = max(available_idxs, key=lambda i: counts[i])
        if counts[target] > 0:
            counts[target] -= 1

    plans: List[Tuple[int, ...]] = []
    # Sample from each bucket (with replacement if needed)
    for idx, cnt in enumerate(counts):
        pool = buckets[idx]
        if not pool:
            continue
        if cnt <= len(pool):
            picks = rng.sample(pool, cnt)
        else:
            # sample with replacement to fill quota
            picks = [rng.choice(pool) for _ in range(cnt)]
        plans.extend(picks)

    # Shuffle the aggregate plans to avoid ordered bias
    rng.shuffle(plans)
    # Ensure exact length
    return plans[:batch_size]
