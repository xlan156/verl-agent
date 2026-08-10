from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agent_system.environments.env_package.discovery.common import (
    compress_ui_observation,
)


logger = logging.getLogger(__name__)


TASK_COMPLETION_BONUS = 20.0

GAME_PROGRESS_REWARD_SCALE = 25.0

NON_TERMINAL_STEP_COST = -0.1

VALID_ACTION_FAILURE_PENALTY = -1.0

INVALID_NO_SKILL_PENALTY = -1.0

STATE_REVISIT_PENALTY = -0.5

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

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    def _target_distance_reward(self, info: Dict) -> float:
        return float(self._task_adapter.target_progress_reward(self, info))

    def _task_reward_state_signature(self, info: Optional[Dict[str, Any]]) -> Tuple[Any, ...]:
        return self._task_adapter.state_signature(info)

    def _state_novelty(self, info: Dict[str, Any]) -> Tuple[bool, bool]:
        """Return ``(changed, novel)`` and remember states within this episode."""
        # ``teacher`` is recreated by ``init_reward_shaping`` on every reset,
        # so keeping the set there avoids reward memory leaking across episodes.
        seen = getattr(self.teacher, "_reward_seen_state_signatures", None)
        if seen is None:
            seen = {self._task_reward_state_signature(self._last_info)}
            self.teacher._reward_seen_state_signatures = seen

        previous = self._task_reward_state_signature(self._last_info)
        current = self._task_reward_state_signature(info)
        changed = current != previous
        novel = changed and current not in seen
        seen.add(current)
        return changed, novel

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -1.0
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -2.0
        if (
            len(self.action_history) >= 4
            and self.action_history[-4:-2] == self.action_history[-2:]
        ):
            penalty = min(penalty, -2.0)
        return penalty

    def action_status_penalty(self) -> float:
        """Penalize failed execution and actions lost during projection."""
        action_status = getattr(self, "_last_action_status", None)
        if action_status == "invalid_no_skill":
            return INVALID_NO_SKILL_PENALTY
        if action_status == "valid_but_failed":
            return VALID_ACTION_FAILURE_PENALTY
        return 0.0

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

        state_changed, novel_state = self._state_novelty(info)
        if not novel_state:
            # Potential-based rewards must telescope around a cycle.  Because
            # step rewards are clipped asymmetrically, suppressing only their
            # repeat positive edge prevents a profitable loop while retaining
            # the negative edge that teaches the policy not to undo progress.
            game_progress_reward = min(game_progress_reward, 0.0)
            target_distance_reward = min(target_distance_reward, 0.0)
            teacher_reward = 0.0
        state_revisit_penalty = (
            STATE_REVISIT_PENALTY if state_changed and not novel_state else 0.0
        )
        repetition_penalty = self._repetition_penalty()
        self._last_action_status = info.get("action_status")
        action_status_penalty = self.action_status_penalty()
        task_completed = bool(self._is_task_complete(info))
        step_cost = 0.0 if task_completed else NON_TERMINAL_STEP_COST
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        task_reward = (
            game_progress_reward
            + target_distance_reward
            + state_revisit_penalty
            + repetition_penalty
            + action_status_penalty
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
            "state_revisit": float(state_revisit_penalty),
            "repetition": float(repetition_penalty),
            "action_status": float(action_status_penalty),
            "step_cost": float(step_cost),
            "completion": float(completion_reward),
        }
        self._prev_score = cur_score
        return reward, done
