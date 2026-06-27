from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agent_system.environments.env_package.discovery.utils import (
    compress_ui_observation,
    is_dispenser_skill,
    is_remove_skill,
)


logger = logging.getLogger(__name__)


class DiscoveryWorldRewardMixin:
    """Reward shaping helpers for DiscoveryWorldEnv.

    The mixin expects the environment instance to provide the runtime fields
    used below: `_api`, `_prev_score`, `_steps`, `_max_steps`, `teacher`,
    `action_history`, `location_history`, and `_last_info`.
    """

    def _game_progress_reward(self, cur_score: float) -> float:
        """Reward based on score increase."""
        return cur_score - self._prev_score

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

        # In the chemistry phase, the teacher only specifies the action class
        # "add one chemical". Any concrete dispenser A/B/C/D is correct.
        if is_dispenser_skill(teacher_skill):
            return 1.0 if is_dispenser_skill(skill_name) else 0.0

        if is_remove_skill(teacher_skill):
            return 1.0 if is_remove_skill(skill_name) else 0.0

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -0.1
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -0.2
        return penalty

    def _invalid_action_penalty(self, info: Dict[str, Any]) -> float:
        """Penalty for invalid or failed actions."""
        action_status = info.get("action_status")
        if action_status == "invalid_no_skill":
            # Keep this smaller than the terminal task reward. The trainer
            # separately applies is_action_valid's format penalty, so a large
            # duplicate penalty would make early GRPO groups uniformly bad.
            return -0.2
        if action_status == "valid_but_failed":
            return -0.2
        return 0.0

    def _stage_reward(self, last_info: Dict[str, Any], current_info: Dict[str, Any]) -> float:
        """Reward for progress through key task stages."""
        if last_info is None:
            return 0.0

        reward = 0.0
        prev_has_key = last_info.get("has_key", False)
        prev_has_jar = last_info.get("has_jar", False)
        prev_is_key_in_jar = last_info.get("is_key_in_jar", False)

        has_key = current_info.get("has_key", False)
        has_jar = current_info.get("has_jar", False)
        is_key_in_jar = current_info.get("is_key_in_jar", False)

        if has_key and not prev_has_key:
            reward += 0.2
        if has_jar and not prev_has_jar:
            reward += 0.2
        if is_key_in_jar and not prev_is_key_in_jar:
            reward += 0.4
        return reward

    def _no_progress_move_penalty(self, action: Optional[str], cur_score: float, prev_score: float) -> float:
        """Penalty for moving but making no progress."""
        if len(self.location_history) < 4:
            return 0.0

        no_location_change = len(set(self.location_history[-4:])) == 1
        no_score_change = abs(cur_score - prev_score) < 1e-6

        if no_score_change:
            if no_location_change:
                return -0.3
            if len(self.action_history) >= 2 and action == self.action_history[-2]:
                return -0.15
            return -0.1
        return 0.0

    def clip_reward(self, reward: float) -> float:
        """Clip reward to a reasonable range."""
        return float(np.clip(reward, -1.0, 2.0))

    def _compute_step_reward(
        self,
        skill_name: Optional[str],
        info: Dict[str, Any],
        format_score: float = 0.0,
    ) -> Tuple[float, bool]:
        cur_score = float(info.get("score_normalized", 0.0))
        prev_score = self._prev_score
        game_progress_reward = self._game_progress_reward(cur_score)
        teacher_skill_reward = self._teacher_skill_reward(skill_name, self._last_info)
        stage_reward = self._stage_reward(self._last_info, info)

        repetition_penalty = self._repetition_penalty()
        invalid_penalty = self._invalid_action_penalty(info)
        no_progress_penalty = self._no_progress_move_penalty(skill_name, cur_score, prev_score)
        format_reward = float(np.clip(format_score, 0.0, 1.0))
        info["teacher_skill"] = self._last_teacher_skill
        info["reward_components"] = {
            "game_progress": game_progress_reward,
            "teacher_skill": teacher_skill_reward,
            "stage": stage_reward,
            "repetition_penalty": repetition_penalty,
            "invalid_penalty": invalid_penalty,
            "no_progress_penalty": no_progress_penalty,
            "format": format_reward,
        }

        task_completed = bool(self._api.areTasksComplete())
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        reward = 0.0
        reward += 10.0 * game_progress_reward
        reward += teacher_skill_reward
        # Dense schema reward is intentionally independent of task progress.
        # This lets an all-invalid GRPO group rank responses by how close they
        # are to the required <think>...</think><action>...</action> protocol.
        reward += format_reward

        reward += repetition_penalty
        reward += invalid_penalty
        reward += no_progress_penalty

        if not task_completed:
            reward = self.clip_reward(reward)
        else:
            reward += 10.0

        self._prev_score = cur_score
        return reward, done
