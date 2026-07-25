from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agent_system.environments.env_package.discovery.utils import (
    CHEMICAL_NAMES,
    chemical_counts,
    compress_ui_observation,
)


logger = logging.getLogger(__name__)


TASK_COMPLETION_BONUS = 20.0

GAME_PROGRESS_REWARD_SCALE = 25.0

NON_TERMINAL_STEP_COST = -0.1

SUCCESS_REMAINING_STEP_BONUS_SCALE = 2.0

MAX_NON_TERMINAL_REWARD = 10.0


class DiscoveryWorldRewardMixin:
    """Reward shaping helpers for DiscoveryWorldEnv.

    The mixin expects the environment instance to provide the runtime fields
    used below: `_api`, `_prev_score`, `_steps`, `_max_steps`, `teacher`,
    `action_history`, `location_history`, and `_last_info`.
    """

    def _game_progress_reward(self, cur_score: float) -> float:
        """
        Reward based on score increase.
        Original score was normalized, so multiplied by a large number.
        """
        return GAME_PROGRESS_REWARD_SCALE * (cur_score - self._prev_score)

    def _teacher_skill_reward(self, skill_name: Optional[str], info: Optional[Dict[str, Any]]) -> float:
        info = info or {}
        teacher_skill = self.teacher.select_skill(self._last_info or info)
        self._last_teacher_skill = teacher_skill

        if not teacher_skill:
            ui = (info.get("raw_observation") or {}).get("ui", {})
            logger.debug(
                "Teacher could not select a skill. is_key_in_jar=%s used_dispensers=%s observation=%s",
                info.get("is_key_in_jar", False),
                info.get("used_dispensers", {}),
                compress_ui_observation(ui),
            )
            return 0.0

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    @staticmethod
    def _chemical_combination(info: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        counts = chemical_counts((info or {}).get("chemical_dict"))
        return tuple(counts[name] for name in CHEMICAL_NAMES)

    def _target_distance_reward(self, info: Dict) -> float:
        """Reward one-step L1 progress toward the simulator's hidden target."""
        target = getattr(self, "_hidden_chemical_target", None)
        if target is None:
            return 0.0
        previous = self._chemical_combination(self._last_info)
        current = self._chemical_combination(info)
        before_distance = sum(
            abs(value - goal) for value, goal in zip(previous, target)
        )
        after_distance = sum(
            abs(value - goal) for value, goal in zip(current, target)
        )
        return float(before_distance - after_distance)

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -1.0
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -2.0
        return penalty

    def clip_reward(self, reward: float) -> float:
        """Bound anomalous rewards without truncating normalized game progress."""
        return float(np.clip(reward, -2.0, MAX_NON_TERMINAL_REWARD))

    def _compute_step_reward(
        self,
        skill_name: Optional[str],
        info: Dict,
    ) -> Tuple[float, bool]:
        cur_score = float(info.get("score_normalized", 0.0))
        game_progress_reward = self._game_progress_reward(cur_score)
        target_distance_reward = self._target_distance_reward(info)
        teacher_reward = (
            self._teacher_skill_reward_coef
            * self._teacher_skill_reward(skill_name, self._last_info)
        )

        repetition_penalty = self._repetition_penalty()
        task_completed = bool(self._is_task_complete(info))
        step_cost = 0.0 if task_completed else NON_TERMINAL_STEP_COST
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        task_reward = (
            game_progress_reward
            + target_distance_reward
            + repetition_penalty
            + step_cost
        )
        reward = task_reward + teacher_reward

        if task_completed:
            completion_reward = (
                TASK_COMPLETION_BONUS
                + SUCCESS_REMAINING_STEP_BONUS_SCALE
                * max(self._max_steps - self._steps, 0)
            )
            task_reward += completion_reward
            reward += completion_reward
        else:
            completion_reward = 0.0
            task_reward = self.clip_reward(task_reward)
            reward = self.clip_reward(reward)

        info["teacher_skill"] = self._last_teacher_skill
        info["task_reward"] = float(task_reward)
        info["reward_components"] = {
            "game_progress": float(game_progress_reward),
            "target_distance": float(target_distance_reward),
            "teacher": float(reward - task_reward),
            "repetition": float(repetition_penalty),
            "step_cost": float(step_cost),
            "completion": float(completion_reward),
        }
        self._prev_score = cur_score
        return reward, done
