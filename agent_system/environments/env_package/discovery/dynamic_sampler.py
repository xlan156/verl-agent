"""Adaptive seed sampling for DiscoveryWorld rollouts.

The sampler combines uniform replay with a difficulty/informativeness priority.
It deliberately lives on the driver (inside the vector environment), rather
than in individual Ray workers, so all seed outcomes contribute to one shared
curriculum.
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
    # Teacher-free task return. This is the reward signal used for priority.
    task_reward_std_ema: float = 0.0
    learning_progress_ema: float = 0.0
    last_sampled_step: int = -1


class DynamicSeedSampler:
    """Sample hard-but-informative seeds while retaining uniform replay."""

    STATE_VERSION = 2

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

        self.uniform_ratio = float(cfg.get("uniform_ratio", 0.4))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        self.hardness_alpha = float(cfg.get("hardness_alpha", 1.0))
        self.min_attempts_per_seed = int(cfg.get("min_attempts_per_seed", 2))
        self.max_probability_per_seed = float(cfg.get("max_probability_per_seed", 0.2))
        self.min_informativeness = float(cfg.get("min_informativeness", 0.2))
        # DiscoveryWorld scales its monotonic normalized score by 25. A scale
        # of 5 keeps ordinary ~0.08 progress steps distinguishable instead of
        # saturating the informativeness term immediately, while terminal
        # success gaps still saturate as intended.
        self.reward_std_scale = float(cfg.get("reward_std_scale", 5.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.25))

        if not 0.0 <= self.uniform_ratio <= 1.0:
            raise ValueError("uniform_ratio must be in [0, 1]")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.min_attempts_per_seed < 0:
            raise ValueError("min_attempts_per_seed must be non-negative")
        if self.reward_std_scale <= 0:
            raise ValueError("reward_std_scale must be positive")

        minimum_cap = 1.0 / len(self.seeds)
        self.max_probability_per_seed = max(self.max_probability_per_seed, minimum_cap)
        self.stats: Dict[int, SeedStats] = {seed: SeedStats() for seed in self.seeds}
        self.total_groups = 0
        self.total_trajectories = 0
        self._rng = np.random.default_rng(int(rng_seed))
        self._last_probabilities = np.full(len(self.seeds), 1.0 / len(self.seeds))

    @staticmethod
    def _ema(previous: float, value: float, alpha: float, initialized: bool) -> float:
        if not initialized:
            return float(value)
        return float((1.0 - alpha) * previous + alpha * value)

    @staticmethod
    def _cap_probabilities(probabilities: np.ndarray, cap: float) -> np.ndarray:
        """Project probabilities onto the simplex with an upper bound."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        if cap >= 1.0 or probabilities.max() <= cap:
            return probabilities

        result = np.zeros_like(probabilities)
        remaining = np.ones(len(probabilities), dtype=bool)
        remaining_mass = 1.0
        while remaining.any():
            scaled = probabilities[remaining]
            scaled = scaled / scaled.sum() * remaining_mass
            over = scaled > cap
            if not over.any():
                result[remaining] = scaled
                break
            remaining_indices = np.flatnonzero(remaining)
            capped_indices = remaining_indices[over]
            result[capped_indices] = cap
            remaining[capped_indices] = False
            remaining_mass = 1.0 - result.sum()
        return result / result.sum()

    def probabilities(self) -> np.ndarray:
        """Return the current sampling distribution in seed order."""
        uniform = np.full(len(self.seeds), 1.0 / len(self.seeds), dtype=np.float64)
        priorities = []
        for seed in self.seeds:
            stat = self.stats[seed]
            if stat.groups == 0:
                priorities.append(1.0)
                continue
            success = float(np.clip(stat.success_ema, 0.0, 1.0))
            hardness = (1.0 - success) ** self.hardness_alpha
            success_variance = 4.0 * success * (1.0 - success)
            reward_variance = min(
                1.0, stat.task_reward_std_ema / self.reward_std_scale
            )
            informativeness = max(success_variance, reward_variance)
            informativeness = self.min_informativeness + (
                1.0 - self.min_informativeness
            ) * informativeness
            progress = min(1.0, stat.learning_progress_ema)
            exploration = 1.0 / np.sqrt(1.0 + stat.groups)
            priorities.append(
                hardness * informativeness
                + self.progress_weight * progress
                + 0.05 * exploration
            )

        adaptive = np.asarray(priorities, dtype=np.float64)
        if not np.isfinite(adaptive).all() or adaptive.sum() <= 0:
            adaptive = uniform
        else:
            adaptive /= adaptive.sum()
        probabilities = self.uniform_ratio * uniform + (1.0 - self.uniform_ratio) * adaptive
        probabilities = self._cap_probabilities(
            probabilities, self.max_probability_per_seed
        )
        self._last_probabilities = probabilities
        return probabilities.copy()

    def sample(self, num_groups: int) -> list[int]:
        """Sample group seeds, covering under-sampled seeds before prioritization."""
        num_groups = int(num_groups)
        if num_groups <= 0:
            return []

        under_sampled = [
            seed
            for seed in self.seeds
            if self.stats[seed].groups < self.min_attempts_per_seed
        ]
        selected: list[int] = []
        if under_sampled:
            self._rng.shuffle(under_sampled)
            under_sampled.sort(key=lambda seed: self.stats[seed].groups)
            selected.extend(under_sampled[:num_groups])

        remaining = num_groups - len(selected)
        if remaining > 0:
            probabilities = self.probabilities()
            available = [seed for seed in self.seeds if seed not in selected]
            replace = remaining > len(available)
            if available:
                indices = [self.seeds.index(seed) for seed in available]
                available_p = probabilities[indices]
                available_p /= available_p.sum()
                sampled = self._rng.choice(
                    available,
                    size=remaining,
                    replace=replace,
                    p=available_p,
                )
            else:
                sampled = self._rng.choice(
                    self.seeds,
                    size=remaining,
                    replace=True,
                    p=probabilities,
                )
            selected.extend(int(seed) for seed in np.atleast_1d(sampled))
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
                "uniform_ratio": self.uniform_ratio,
                "ema_alpha": self.ema_alpha,
                "hardness_alpha": self.hardness_alpha,
                "min_attempts_per_seed": self.min_attempts_per_seed,
                "max_probability_per_seed": self.max_probability_per_seed,
                "min_informativeness": self.min_informativeness,
                "reward_std_scale": self.reward_std_scale,
                "progress_weight": self.progress_weight,
            },
            "stats": {str(seed): asdict(stat) for seed, stat in self.stats.items()},
            "total_groups": self.total_groups,
            "total_trajectories": self.total_trajectories,
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
        if "rng_state" in state:
            self._rng.bit_generator.state = dict(state["rng_state"])
