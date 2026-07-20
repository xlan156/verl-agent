from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from agent_system.environments.env_package.discovery.utils import (
    CHEMICAL_NAMES,
    WASH,
    candidate_targets,
    chemical_counts,
    compress_ui_observation,
    is_dispenser_skill,
    is_remove_skill,
    observable_experiment_evidence,
)


logger = logging.getLogger(__name__)


TASK_COMPLETION_BONUS = 20.0
SUCCESS_REMAINING_STEP_BONUS_SCALE = 2.0
TARGET_CONFIRMATION_BONUS = 2.0
WRONG_EXPLOITATION_DIRECTION_PENALTY = -2.0
NON_TERMINAL_STEP_COST = -0.1
MAX_NON_TERMINAL_REWARD = 6.0


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

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    @staticmethod
    def _chemical_combination(info: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        counts = chemical_counts((info or {}).get("chemical_dict"))
        return tuple(counts[name] for name in CHEMICAL_NAMES)

    def _chemical_belief_reward(
        self,
        skill_name: Optional[str],
        info: Dict[str, Any],
    ) -> Tuple[float, float, float, int, int]:
        """Reward observable belief reduction, target confirmation, and exploitation.

        This deliberately derives the target from accumulated assay evidence. It
        never reads ``_hidden_chemical_target``.
        """
        if not (
            is_dispenser_skill(skill_name)
            or is_remove_skill(skill_name)
            or skill_name == WASH
        ):
            return 0.0, 0.0, 0.0, 0, 0

        if not hasattr(self, "_chemical_evidence"):
            self._chemical_evidence = {}

        required_amount = int(info.get("max_chemical_n", 0) or 0)
        if required_amount <= 0:
            return 0.0, 0.0, 0.0, 0, 0

        candidates_before = candidate_targets(self._chemical_evidence, required_amount)
        evidence = observable_experiment_evidence(self._last_info or {}, info)
        if evidence is not None:
            self._chemical_evidence[self._chemical_combination(info)] = (
                evidence["kind"],
                evidence["label"],
            )
        candidates_after = candidate_targets(self._chemical_evidence, required_amount)

        before_count = len(candidates_before)
        after_count = len(candidates_after)
        information_gain = 0.0
        if before_count > 1 and 0 < after_count < before_count:
            information_gain = math.log(before_count / after_count)

        confirmation = (
            TARGET_CONFIRMATION_BONUS
            if before_count > 1 and after_count == 1
            else 0.0
        )

        exploitation = 0.0
        if before_count == 1:
            target = candidates_before[0]
            previous = self._chemical_combination(self._last_info)
            current = self._chemical_combination(info)
            before_distance = sum(abs(value - goal) for value, goal in zip(previous, target))
            after_distance = sum(abs(value - goal) for value, goal in zip(current, target))
            # Preserve magnitude: wash_jar and future macro actions may change
            # more than one unit of L1 distance in a single transition.
            exploitation = float(before_distance - after_distance)

        return information_gain, confirmation, exploitation, before_count, after_count

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -1.0
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -2.0
        return penalty

    def clip_reward(self, reward: float) -> float:
        """Bound shaping reward without truncating an n=3 one-step resolution."""
        return float(np.clip(reward, -2.0, MAX_NON_TERMINAL_REWARD))

    def _compute_step_reward(
        self,
        skill_name: Optional[str],
        info: Dict[str, Any],
        format_score: float = 0.0,
    ) -> Tuple[float, bool]:
        cur_score = float(info.get("score_normalized", 0.0))
        game_progress_reward = self._game_progress_reward(cur_score)
        teacher_skill_reward = self._teacher_skill_reward(skill_name, self._last_info)
        weighted_teacher_reward = self._teacher_skill_reward_coef * teacher_skill_reward
        (
            belief_information_gain,
            target_confirmation_reward,
            exploitation_reward,
            candidates_before,
            candidates_after,
        ) = self._chemical_belief_reward(skill_name, info)
        wrong_exploitation_direction = (
            candidates_before == 1 and exploitation_reward < 0.0
        )

        repetition_penalty = self._repetition_penalty()
        # Format validity is already handled by the actor's invalid-action
        # penalty. A positive format reward on every step is a living reward
        # and therefore conflicts with finishing quickly.
        format_validity = float(np.clip(format_score, 0.0, 1.0))
        format_reward = 0.0
        task_completed = bool(self._is_task_complete(info))
        step_cost = 0.0 if task_completed else NON_TERMINAL_STEP_COST
        remaining_steps = max(self._max_steps - self._steps, 0)
        remaining_step_bonus = (
            SUCCESS_REMAINING_STEP_BONUS_SCALE * remaining_steps
            if task_completed
            else 0.0
        )
        info["teacher_skill"] = self._last_teacher_skill
        info["reward_components"] = {
            "game_progress": game_progress_reward,
            "teacher_skill": teacher_skill_reward,
            "teacher_skill_weighted": weighted_teacher_reward,
            "belief_information_gain": belief_information_gain,
            "target_confirmation": target_confirmation_reward,
            "exploitation_distance_delta": exploitation_reward,
            "candidate_count_before": candidates_before,
            "candidate_count_after": candidates_after,
            "wrong_exploitation_direction": wrong_exploitation_direction,
            "wrong_exploitation_penalty": (
                WRONG_EXPLOITATION_DIRECTION_PENALTY
                if wrong_exploitation_direction
                else 0.0
            ),
            "repetition_penalty": repetition_penalty,
            "step_cost": step_cost,
            "format_validity": format_validity,
            "format": format_reward,
            "remaining_steps": remaining_steps,
            "remaining_step_bonus": remaining_step_bonus,
        }

        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        if wrong_exploitation_direction:
            # A syntactically valid answer must not erase the learning signal
            # for moving away from an already-confirmed target.
            reward = WRONG_EXPLOITATION_DIRECTION_PENALTY
        else:
            reward = 0.0
            reward += game_progress_reward
            reward += weighted_teacher_reward
            reward += belief_information_gain
            reward += target_confirmation_reward
            reward += exploitation_reward
            reward += format_reward
            reward += repetition_penalty
            reward += step_cost

        if not task_completed:
            reward = self.clip_reward(reward)
        else:
            reward += TASK_COMPLETION_BONUS + remaining_step_bonus

        info["reward_components"]["total"] = reward
        self._prev_score = cur_score
        return reward, done
