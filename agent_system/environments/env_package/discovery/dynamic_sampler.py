"""Adaptive seed sampling for DiscoveryWorld rollouts.

The sampler prioritizes seeds with a Bayesian estimate of whether they will
produce DAPO-useful rollout groups. It deliberately lives on the driver
(inside the vector environment), rather than in individual Ray workers, so all
seed outcomes contribute to one shared curriculum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SeedStats:
    groups: int = 0
    trajectories: int = 0
    accepted_groups: int = 0
    rejected_groups: int = 0
    successes: int = 0
    success_ema: float = 0.0
    # Optimized return, including teacher shaping. Kept for diagnostics only.
    reward_std_ema: float = 0.0
    # Teacher-free task return variation. Kept for diagnostics only.
    task_reward_std_ema: float = 0.0
    learning_progress_ema: float = 0.0
    last_sampled_step: int = -1


class DynamicSeedSampler:
    """Sample seeds by their posterior probability of producing useful groups.

    A useful group is exactly the event consumed by DAPO filtering: the seed's
    rollout group survives the non-zero task-return variance filter.  Each seed
    keeps a Beta posterior over that Bernoulli event with Jeffreys prior
    ``Beta(0.5, 0.5)``; sampling probabilities are normalized posterior means.
    """

    STATE_VERSION = 4
    JEFFREYS_PRIOR = 0.5

    def __init__(
        self,
        seeds: Iterable[int],
        config: Optional[Mapping[str, Any]] = None,
        rng_seed: int = 0,
    ) -> None:
        cfg = dict(config or {})
        self.seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
        if not self.seeds:
            raise ValueError("DynamicSeedSampler requires at least one seed")

        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        replacement_value = cfg.get("sample_with_replacement", True)
        if isinstance(replacement_value, str):
            replacement_value = replacement_value.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.sample_with_replacement = bool(replacement_value)

        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.stats: Dict[int, SeedStats] = {seed: SeedStats() for seed in self.seeds}
        self.total_groups = 0
        self.total_trajectories = 0
        self.total_sample_batches = 0
        self.total_sampled_groups = 0
        self.batches_with_replacement = 0
        self.duplicate_selections = 0
        self._rng = np.random.default_rng(int(rng_seed))
        self._last_probabilities = np.full(len(self.seeds), 1.0 / len(self.seeds))

    @staticmethod
    def _ema(previous: float, value: float, alpha: float, initialized: bool) -> float:
        if not initialized:
            return float(value)
        return float((1.0 - alpha) * previous + alpha * value)

    def _posterior_mean(self, stat: SeedStats) -> float:
        prior = self.JEFFREYS_PRIOR
        return float(
            (prior + stat.accepted_groups)
            / (2.0 * prior + stat.accepted_groups + stat.rejected_groups)
        )

    def probabilities(self) -> np.ndarray:
        """Return the current sampling distribution in seed order."""
        priorities = np.asarray(
            [self._posterior_mean(self.stats[seed]) for seed in self.seeds],
            dtype=np.float64,
        )
        if not np.isfinite(priorities).all() or priorities.sum() <= 0:
            probabilities = np.full(len(self.seeds), 1.0 / len(self.seeds))
        else:
            probabilities = priorities / priorities.sum()
        self._last_probabilities = probabilities
        return probabilities.copy()

    def sample(self, num_groups: int) -> list[int]:
        """Sample group seeds from the posterior sampling distribution."""
        num_groups = int(num_groups)
        if num_groups <= 0:
            return []

        probabilities = self.probabilities()
        replace = self.sample_with_replacement or num_groups > len(self.seeds)
        sampled = self._rng.choice(
            self.seeds,
            size=num_groups,
            replace=replace,
            p=probabilities,
        )
        selected = [int(seed) for seed in np.atleast_1d(sampled)]
        duplicate_count = len(selected) - len(set(selected))
        self.total_sample_batches += 1
        self.total_sampled_groups += len(selected)
        self.batches_with_replacement += int(duplicate_count > 0)
        self.duplicate_selections += duplicate_count
        return selected

    def observe_group(
        self,
        seed: int,
        rewards: Sequence[float],
        successes: Sequence[float],
        accepted: bool,
        global_step: Optional[int] = None,
        task_rewards: Optional[Sequence[float]] = None,
    ) -> None:
        seed = int(seed)
        if seed not in self.stats:
            raise ValueError(f"Observed seed {seed} is outside sampler pool {self.seeds}")
        rewards_array = np.asarray(rewards, dtype=np.float64)
        task_rewards_array = np.asarray(
            rewards if task_rewards is None else task_rewards,
            dtype=np.float64,
        )
        successes_array = np.asarray(successes, dtype=np.float64)
        if (
            rewards_array.size == 0
            or task_rewards_array.size == 0
            or successes_array.size == 0
        ):
            raise ValueError("Cannot observe an empty rollout group")
        if not (
            rewards_array.size
            == task_rewards_array.size
            == successes_array.size
        ):
            raise ValueError(
                "rewards, task_rewards, and successes must have equal lengths"
            )

        stat = self.stats[seed]
        initialized = stat.groups > 0
        group_success = float(successes_array.mean())
        reward_std = float(rewards_array.std())
        task_reward_std = float(task_rewards_array.std())
        previous_success = stat.success_ema

        stat.groups += 1
        stat.trajectories += int(successes_array.size)
        stat.successes += int(np.rint(successes_array).sum())
        stat.accepted_groups += int(bool(accepted))
        stat.rejected_groups += int(not accepted)
        stat.success_ema = self._ema(
            stat.success_ema, group_success, self.ema_alpha, initialized
        )
        stat.reward_std_ema = self._ema(
            stat.reward_std_ema, reward_std, self.ema_alpha, initialized
        )
        stat.task_reward_std_ema = self._ema(
            stat.task_reward_std_ema,
            task_reward_std,
            self.ema_alpha,
            initialized,
        )
        progress = abs(group_success - previous_success) if initialized else 0.0
        stat.learning_progress_ema = self._ema(
            stat.learning_progress_ema, progress, self.ema_alpha, initialized
        )
        stat.last_sampled_step = (
            int(global_step) if global_step is not None else self.total_groups
        )
        self.total_groups += 1
        self.total_trajectories += int(successes_array.size)

    def metrics(self) -> Dict[str, float]:
        probabilities = self.probabilities()
        metrics: Dict[str, float] = {
            "seed_sampler/total_groups": float(self.total_groups),
            "seed_sampler/total_trajectories": float(self.total_trajectories),
            "seed_sampler/probability_entropy": float(
                -(probabilities * np.log(probabilities + 1e-12)).sum()
            ),
            "seed_sampler/replacement_batch_rate": (
                self.batches_with_replacement / self.total_sample_batches
                if self.total_sample_batches
                else 0.0
            ),
            "seed_sampler/duplicate_selection_rate": (
                self.duplicate_selections / self.total_sampled_groups
                if self.total_sampled_groups
                else 0.0
            ),
        }
        total_accepted = 0
        total_rejected = 0
        for index, seed in enumerate(self.seeds):
            stat = self.stats[seed]
            total_accepted += stat.accepted_groups
            total_rejected += stat.rejected_groups
            denominator = stat.accepted_groups + stat.rejected_groups
            metrics[f"seed_sampler/success_ema/{seed}"] = stat.success_ema
            metrics[f"seed_sampler/shaped_reward_std_ema/{seed}"] = stat.reward_std_ema
            # Backward-compatible alias for existing dashboards/checkpoints.
            metrics[f"seed_sampler/reward_std_ema/{seed}"] = stat.reward_std_ema
            metrics[f"seed_sampler/task_reward_std_ema/{seed}"] = (
                stat.task_reward_std_ema
            )
            metrics[f"seed_sampler/probability/{seed}"] = float(probabilities[index])
            metrics[f"seed_sampler/posterior_mean/{seed}"] = self._posterior_mean(stat)
            metrics[f"seed_sampler/groups/{seed}"] = float(stat.groups)
            metrics[f"seed_sampler/accept_rate/{seed}"] = (
                stat.accepted_groups / denominator if denominator else 0.0
            )
        total = total_accepted + total_rejected
        metrics["seed_sampler/accept_rate"] = total_accepted / total if total else 0.0
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "seeds": list(self.seeds),
            "config": {
                "posterior_prior": "jeffreys",
                "ema_alpha": self.ema_alpha,
                "sample_with_replacement": self.sample_with_replacement,
            },
            "stats": {str(seed): asdict(stat) for seed, stat in self.stats.items()},
            "total_groups": self.total_groups,
            "total_trajectories": self.total_trajectories,
            "total_sample_batches": self.total_sample_batches,
            "total_sampled_groups": self.total_sampled_groups,
            "batches_with_replacement": self.batches_with_replacement,
            "duplicate_selections": self.duplicate_selections,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        state_seeds = tuple(int(seed) for seed in state.get("seeds", []))
        if state_seeds != self.seeds:
            raise ValueError(
                f"Sampler checkpoint seeds {state_seeds} do not match configured seeds {self.seeds}"
            )
        for seed in self.seeds:
            saved = dict(state.get("stats", {}).get(str(seed), {}))
            if saved:
                # Version 1 only tracked the teacher-shaped return. Preserve
                # checkpoint compatibility, but do not silently reuse that
                # contaminated statistic as teacher-free task variance.
                saved.setdefault("task_reward_std_ema", 0.0)
                self.stats[seed] = SeedStats(**saved)
        self.total_groups = int(state.get("total_groups", 0))
        self.total_trajectories = int(state.get("total_trajectories", 0))
        self.total_sample_batches = int(state.get("total_sample_batches", 0))
        self.total_sampled_groups = int(state.get("total_sampled_groups", 0))
        self.batches_with_replacement = int(
            state.get("batches_with_replacement", 0)
        )
        self.duplicate_selections = int(state.get("duplicate_selections", 0))
        if "rng_state" in state:
            self._rng.bit_generator.state = dict(state["rng_state"])
