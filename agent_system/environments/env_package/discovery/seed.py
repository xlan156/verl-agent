from __future__ import annotations

import argparse
from typing import Dict, List
import random

from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.scenarios.storage_shed import (
	mkEnumerateChemicalCombinations,
)


def expand_seedpool(seed_pool: List[int], target_size: int) -> List[int]:
	"""Repeat a seed pool cyclically until reaching target_size."""
	target_size = max(int(target_size), 0)
	if target_size == 0:
		return []
	if not seed_pool:
		raise ValueError("seed_pool must not be empty when target_size > 0")

	out: List[int] = []
	pool_len = len(seed_pool)
	for i in range(target_size):
		out.append(int(seed_pool[i % pool_len]))
	return out


def assign_split_seeds(
	base_seed: int,
	train_size: int,
	val_size: int,
	num_chemicals: int = 4,
	min_chemicals: int = 1,
	min_amount: int = 2,
	max_amount: int = 2,
) -> Dict[str, List[int]]:
	"""Assign deterministic train/val seeds from chemical-combination index space.

	1) Enumerate all valid chemical combinations and use their indices as candidate seeds.
	2) Shuffle deterministically with base_seed.
	3) Randomly partition candidates into train/val pools (no overlap).
	4) Expand each pool to requested size by cyclic repetition when needed.

	Special case: if both train_size and val_size are 0, split the full candidate pool
	into train/val with val taking 1/4 of the combinations and without cyclic reuse.
	"""
	train_size = int(train_size)
	val_size = int(val_size)

	all_combinations = mkEnumerateChemicalCombinations(
		numChemicals=int(num_chemicals),
		minChemicals=int(min_chemicals),
		minAmount=int(min_amount),
		maxAmount=int(max_amount),
	)
	total_combinations = len(all_combinations)
	if total_combinations <= 0:
		raise ValueError("No valid chemical combinations available for seed assignment")
	zero_zero_default = train_size == 0 and val_size == 0
	if zero_zero_default:
		val_size = max(1, total_combinations // 4)
		train_size = total_combinations - val_size
	else:
		train_size = max(train_size, 0)
		val_size = max(val_size, 0)

	candidate_seeds = list(range(total_combinations))
	rng = random.Random(int(base_seed))
	rng.shuffle(candidate_seeds)

	if train_size > 0 and val_size > 0 and total_combinations >= 2:
		train_ratio = train_size / float(train_size + val_size)
		train_pool_size = int(round(total_combinations * train_ratio))
		train_pool_size = max(1, min(total_combinations - 1, train_pool_size))
	elif train_size > 0:
		train_pool_size = total_combinations
	else:
		train_pool_size = 0

	train_pool = candidate_seeds[:train_pool_size]
	val_pool = candidate_seeds[train_pool_size:]

	if train_size > 0 and not train_pool:
		train_pool = candidate_seeds[:1]
	if val_size > 0 and not val_pool:
		if len(candidate_seeds) > 1:
			val_pool = candidate_seeds[-1:]
		else:
			val_pool = candidate_seeds[:1]

	if zero_zero_default:
		train_seeds = train_pool[:train_size]
		val_seeds = val_pool[:val_size]
	else:
		train_seeds = expand_seedpool(train_pool, train_size)
		val_seeds = expand_seedpool(val_pool, val_size)

	return {
		"train": train_seeds,
		"val": val_seeds,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect DiscoveryWorld seed assignment.")
	parser.add_argument("--base-seed", type=int, default=0, help="Base seed used for deterministic shuffling.")
	parser.add_argument("--train-size", type=int, default=0, help="Requested train seed count; 0 means full combination count.")
	parser.add_argument("--val-size", type=int, default=0, help="Requested val seed count; 0 means full combination count.")
	parser.add_argument("--num-chemicals", type=int, default=4, help="Number of available chemicals.")
	parser.add_argument("--min-chemicals", type=int, default=1, help="Minimum distinct chemicals in a combination.")
	parser.add_argument("--min-amount", type=int, default=2, help="Minimum total chemical amount in a combination.")
	parser.add_argument("--max-amount", type=int, default=2, help="Maximum total chemical amount in a combination.")
	args = parser.parse_args()

	seed_split = assign_split_seeds(
		base_seed=args.base_seed,
		train_size=args.train_size,
		val_size=args.val_size,
		num_chemicals=args.num_chemicals,
		min_chemicals=args.min_chemicals,
		min_amount=args.min_amount,
		max_amount=args.max_amount,
	)

	train_pool_size = len(mkEnumerateChemicalCombinations(
		numChemicals=args.num_chemicals,
		minChemicals=args.min_chemicals,
		minAmount=args.min_amount,
		maxAmount=args.max_amount,
	))
	print(f"combination_count={train_pool_size}")
	print(f"train_size={args.train_size} -> {seed_split['train']}")
	print(f"val_size={args.val_size} -> {seed_split['val']}")


if __name__ == "__main__":
	main()

