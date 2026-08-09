from __future__ import annotations

import argparse
from collections import defaultdict
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


def _build_fixed_seed_pool(
	base_seed: int,
	num_chemicals: int = 4,
	min_chemicals: int = 1,
	min_amount: int = 2,
	max_amount: int = 2,
	train_fraction: float = 0.7,
) -> Dict[str, List[int]]:
	"""Build a deterministic train/val seed pool that does not depend on runtime sizes."""
	all_combinations = mkEnumerateChemicalCombinations(
		numChemicals=int(num_chemicals),
		minChemicals=int(min_chemicals),
		minAmount=int(min_amount),
		maxAmount=int(max_amount),
	)
	total_combinations = len(all_combinations)
	if total_combinations <= 0:
		raise ValueError("No valid chemical combinations available for seed assignment")

	candidate_seeds = list(range(total_combinations))
	rng = random.Random(int(base_seed))
	rng.shuffle(candidate_seeds)

	train_fraction = min(max(float(train_fraction), 0.0), 1.0)
	if total_combinations == 1:
		train_pool = candidate_seeds[:1]
		val_pool = candidate_seeds[:1]
	else:
		train_pool_size = int(round(total_combinations * train_fraction))
		train_pool_size = max(1, min(total_combinations - 1, train_pool_size))
		train_pool = candidate_seeds[:train_pool_size]
		val_pool = candidate_seeds[train_pool_size:]
		if not val_pool:
			val_pool = candidate_seeds[-1:]

	return {
		"train": train_pool,
		"val": val_pool,
	}


def build_fixed_seed_pools_by_amount(
	base_seed: int,
	max_amount: int,
	num_chemicals: int = 4,
	min_chemicals: int = 1,
	train_fraction: float = 0.7,
) -> Dict[int, Dict[str, List[int]]]:
	"""Prebuild fixed train/val seed pools for every chemical amount up to max_amount."""
	max_amount = max(1, int(max_amount))
	pools: Dict[int, Dict[str, List[int]]] = {}
	for amount in range(1, max_amount + 1):
		pools[amount] = _build_fixed_seed_pool(
			base_seed=base_seed + amount,
			num_chemicals=num_chemicals,
			min_chemicals=min_chemicals,
			min_amount=amount,
			max_amount=amount,
			train_fraction=train_fraction,
		)
	return pools


def build_ordered_seed_pools_by_amount(
	max_amount: int,
	num_chemicals: int = 4,
	min_chemicals: int = 1,
	train_fraction: float = 0.8,
) -> Dict[int, Dict[str, List[int]]]:
	"""Build an independent, contiguous train/val seed pool for each exact amount."""
	max_amount = max(1, int(max_amount))
	train_fraction = min(max(float(train_fraction), 0.0), 1.0)
	pools: Dict[int, Dict[str, List[int]]] = {}
	for amount in range(1, max_amount + 1):
		all_combinations = mkEnumerateChemicalCombinations(
			numChemicals=int(num_chemicals),
			minChemicals=int(min_chemicals),
			minAmount=int(amount),
			maxAmount=int(amount),
		)
		if not all_combinations:
			raise ValueError("No valid chemical combinations available for seed assignment")

		seeds = list(range(len(all_combinations)))
		if len(seeds) == 1:
			train_pool = list(seeds)
			val_pool = list(seeds)
		else:
			train_size = int(round(len(seeds) * train_fraction))
			train_size = max(1, min(len(seeds) - 1, train_size))
			train_pool = seeds[:train_size]
			val_pool = seeds[train_size:]

		pools[amount] = {
			"train": train_pool,
			"val": val_pool,
		}
	return pools


def assign_split_seeds(
	base_seed: int,
	train_size: int,
	val_size: int,
	num_chemicals: int = 4,
	min_chemicals: int = 1,
	min_amount: int = 2,
	max_amount: int = 2,
) -> Dict[str, List[int]]:
	"""Assign deterministic train/val seeds from a fixed chemical-combination pool.

	The train/val pools are determined only by the chemical combination space and
	base_seed. Requested train_size/val_size only control how many seeds are sampled
	from those pools, using cyclic reuse when necessary.
	"""
	train_size = int(train_size)
	val_size = int(val_size)
	train_size = max(train_size, 0)
	val_size = max(val_size, 0)

	pools = _build_fixed_seed_pool(
		base_seed=base_seed,
		num_chemicals=num_chemicals,
		min_chemicals=min_chemicals,
		min_amount=min_amount,
		max_amount=max_amount,
	)
	train_seeds = expand_seedpool(pools["train"], train_size)
	val_seeds = expand_seedpool(pools["val"], val_size)

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
