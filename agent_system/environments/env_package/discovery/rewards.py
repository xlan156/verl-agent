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

FORMAT_REWARD_SCALE = 0.5
TASK_COMPLETION_BONUS = 20.0


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
            return 0.2 if is_dispenser_skill(skill_name) else 0.0

        if is_remove_skill(teacher_skill):
            return 0.2 if is_remove_skill(skill_name) else 0.0

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -1.0
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -2.0
        return penalty

    def _invalid_action_penalty(self, info: Dict[str, Any]) -> float:
        """Penalty for invalid or failed actions."""
        action_status = info.get("action_status")
        if action_status == "invalid_no_skill":
            # Keep this smaller than the terminal task reward. The trainer
            # separately applies is_action_valid's format penalty, so a large
            # duplicate penalty would make early GRPO groups uniformly bad.
            return -2.0
        if action_status == "valid_but_failed":
            return -1.0
        return 0.0

    def _no_progress_move_penalty(self, action: Optional[str], cur_score: float, prev_score: float) -> float:
        """Penalty for moving but making no progress."""
        if len(self.location_history) < 4:
            return 0.0

        no_location_change = len(set(self.location_history[-4:])) == 1
        no_score_change = abs(cur_score - prev_score) < 1e-6

        if no_score_change:
            if no_location_change:
                return -1.0
            if len(self.action_history) >= 2 and action == self.action_history[-2]:
                return -0.5
            return -0.5
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

        repetition_penalty = self._repetition_penalty()
        #invalid_penalty = self._invalid_action_penalty(info)
        no_progress_penalty = self._no_progress_move_penalty(skill_name, cur_score, prev_score)
        format_reward = FORMAT_REWARD_SCALE * float(np.clip(format_score, 0.0, 1.0))
        info["teacher_skill"] = self._last_teacher_skill
        info["reward_components"] = {
            "game_progress": game_progress_reward,
            "teacher_skill": teacher_skill_reward,
            "repetition_penalty": repetition_penalty,
            "no_progress_penalty": no_progress_penalty,
            "format": format_reward,
        }

        task_completed = bool(self._is_task_complete(info))
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        reward = 0.0
        reward += 20.0 * game_progress_reward
        reward += teacher_skill_reward
        # Keep schema reward small once the policy mostly emits valid actions;
        # otherwise long non-winning episodes can outrank the terminal action.
        reward += format_reward

        reward += repetition_penalty
        reward += no_progress_penalty

        if not task_completed:
            reward = self.clip_reward(reward)
        else:
            reward += TASK_COMPLETION_BONUS

        self._prev_score = cur_score
        return reward, done
