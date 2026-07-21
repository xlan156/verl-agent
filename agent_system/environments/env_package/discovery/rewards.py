from __future__ import annotations

import logging
import math
import re
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
# Must not exceed the downstream invalid-action penalty (currently 0.1), or a
# nearly-correct but non-executable response can outrank a valid neutral step.
INVALID_FORMAT_BOOTSTRAP_SCALE = 0.1
THINKING_REWARD_MAX = 1.0
THINKING_REWARD_MIN = -0.25


_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_CAUSAL_WORDS = (
    "because", "so ", "therefore", "since", "thus", "need", "must",
    "not", "no ", "but", "while", "remaining", "candidate", "target",
    "excess", "missing", "accessible", "inventory", "far", "near",
)
_GENERIC_TASK_PHRASES = (
    "find the correct combination",
    "infer the correct combination",
    "derust the key and then open the door",
    "complete the task",
)


def extract_thinking(response: Any) -> str:
    """Extract free-form reasoning without imposing an internal schema."""
    if not isinstance(response, str):
        return ""
    match = _THINK_RE.search(response)
    return match.group(1).strip() if match else ""


def combine_task_and_thinking_reward(
    task_reward: float,
    thinking_score: float,
    reward_components: Optional[Dict[str, Any]] = None,
    coefficient: float = 0.2,
) -> Tuple[float, float, float]:
    """Add bounded CoT shaping without weakening hard action penalties."""
    reward_components = reward_components or {}
    effective = (
        0.0
        if reward_components.get("wrong_exploitation_direction")
        else float(np.clip(thinking_score, THINKING_REWARD_MIN, THINKING_REWARD_MAX))
    )
    weighted = float(coefficient) * effective
    return float(task_reward) + weighted, effective, weighted


def invalid_format_bootstrap_reward(
    skill_name: Optional[str],
    format_score: float,
) -> float:
    """Reward progress toward executable syntax only while still invalid."""
    if skill_name is not None:
        return 0.0
    return INVALID_FORMAT_BOOTSTRAP_SCALE * float(np.clip(format_score, 0.0, 1.0))


def score_thinking(
    response: Any,
    skill_name: Optional[str],
    state: Optional[Dict[str, Any]],
    candidate_targets_before: Optional[list] = None,
    valid_skills: Optional[list] = None,
) -> Dict[str, Any]:
    """Score whether a free-form thought grounds and supports its action.

    This is intentionally a small symbolic verifier, not a reference-answer
    matcher.  It accepts varied prose and only rewards facts that can be
    checked against the observation available before the action.  Singleton
    action states are not polluted with distractors: they can earn grounding
    credit, while the choice-specific bonus is reserved for real choices.
    """
    thought = extract_thinking(response)
    lower = thought.lower()
    details: Dict[str, Any] = {
        "raw": 0.0,
        "grounding": 0.0,
        "action_support": 0.0,
        "choice_support": 0.0,
        "generic_penalty": 0.0,
        "thinking": thought,
    }
    if not thought or not skill_name:
        return details

    state = state or {}
    valid_skills = list(valid_skills or [])
    candidates = [tuple(int(v) for v in target) for target in (candidate_targets_before or [])]
    counts = chemical_counts(state.get("chemical_dict"))
    has_causal_language = any(word in lower for word in _CAUSAL_WORDS)
    grounding = 0.0
    action_support = 0.0
    choice_support = 0.0
    contradiction_penalty = 0.0

    rust_status = str(state.get("key_rust_status") or "unknown").strip().lower()
    if rust_status != "no rust" and any(
        phrase in lower for phrase in ("key is no longer rusted", "key has no rust", "rust-free key")
    ):
        contradiction_penalty -= 0.25

    # Lifecycle/navigation skills: check the decisive observable precondition.
    lifecycle_rules = {
        "move_to_key": ("key", ("far", "not accessible", "reach", "move", "location")),
        "pick_up_key": ("key", ("accessible", "inventory", "pick", "take", "collect", "at the key")),
        "move_to_jar": ("jar", ("far", "not accessible", "reach", "move", "location")),
        "pick_up_jar": ("jar", ("accessible", "inventory", "pick", "take", "collect", "at the jar")),
        "put_key_in_jar": ("jar", ("key", "inside", "put", "place", "apply chemical")),
        "open_door": ("door", ("no rust", "rust-free", "derust", "clean key", "ready")),
    }
    if skill_name in lifecycle_rules:
        subject, evidence_words = lifecycle_rules[skill_name]
        if subject in lower:
            grounding += 0.25
        if any(word in lower for word in evidence_words):
            grounding += 0.35
        if has_causal_language:
            action_support += 0.20

        # Reward agreement with explicit state flags, without requiring their
        # exact names to appear in natural language.
        if skill_name == "pick_up_key" and not bool(state.get("has_key")) and "key" in lower:
            action_support += 0.20
        elif skill_name == "pick_up_jar" and not bool(state.get("has_jar")) and "jar" in lower:
            action_support += 0.20
        elif skill_name == "put_key_in_jar" and not bool(state.get("is_key_in_jar")) and "jar" in lower:
            action_support += 0.20
        elif skill_name == "open_door" and str(state.get("key_rust_status", "")).lower() == "no rust":
            action_support += 0.20
        elif skill_name in {"move_to_key", "move_to_jar"} and any(
            word in lower for word in ("far", "not accessible", "reach", "move")
        ):
            action_support += 0.20

    chemical_match = re.fullmatch(r"(?:use_dispenser|remove_chemical)_([A-D])(?:_on_jar)?", skill_name)
    if chemical_match:
        chemical = chemical_match.group(1)
        index = CHEMICAL_NAMES.index(chemical)
        is_remove = skill_name.startswith("remove_")
        # Bare lower-case "a" is an English article, not Chemical A.  Accept
        # upper-case symbols and explicit "chemical/substance X" mentions.
        named_chemicals = set(re.findall(r"\b[A-D]\b", thought))
        for name in CHEMICAL_NAMES:
            if re.search(rf"\b(?:chemical|substance)\s+{name}\b", thought, re.IGNORECASE):
                named_chemicals.add(name)
        selected_chemical_is_named = chemical in named_chemicals
        operation_words = (
            ("remove", "reduce", "excess", "too much", "not needed", "eliminate")
            if is_remove else
            ("add", "use", "dispense", "test", "try", "missing", "need")
        )
        if selected_chemical_is_named:
            grounding += 0.30
        if any(word in lower for word in operation_words):
            action_support += 0.25
        if has_causal_language:
            action_support += 0.15

        if named_chemicals and chemical not in named_chemicals:
            # The thought explicitly reasons about a different chemical than
            # the action.  Operation words alone must not receive credit.
            action_support = 0.0
            contradiction_penalty = -0.25

        if candidates:
            current = counts[chemical]
            candidate_values = [target[index] for target in candidates]
            direction_supported = (
                current > max(candidate_values) if is_remove
                else current < min(candidate_values)
            )
            # When candidates disagree, exploration may still be reasonable;
            # require an explicit test/candidate rationale rather than treating
            # the hidden target as known.
            exploration_supported = len(set(candidate_values)) > 1 and any(
                word in lower for word in ("test", "try", "candidate", "information", "learn")
            )
            if selected_chemical_is_named and direction_supported and any(
                word in lower for word in ("candidate", "target", "excess", "missing", "need", "too much")
            ):
                choice_support += 0.30
            elif selected_chemical_is_named and exploration_supported:
                choice_support += 0.20

    elif skill_name == WASH:
        if "jar" in lower or "mixture" in lower or "combination" in lower:
            grounding += 0.25
        if any(word in lower for word in ("wash", "wrong", "failed", "different", "restart", "reset")):
            action_support += 0.40
        if has_causal_language:
            action_support += 0.15

    # Only multi-option states receive a bonus for distinguishing this action.
    # Singleton states still learn truthful prerequisite explanations.
    if len(valid_skills) <= 1:
        choice_support = 0.0

    generic_penalty = 0.0
    if any(phrase in lower for phrase in _GENERIC_TASK_PHRASES):
        # Do not punish a longer grounded explanation merely for mentioning the
        # goal; punish it only when it lacks action-specific evidence.
        if grounding < 0.30 or action_support < 0.20:
            generic_penalty = -0.25

    raw = float(np.clip(
        grounding + action_support + choice_support + generic_penalty + contradiction_penalty,
        THINKING_REWARD_MIN,
        THINKING_REWARD_MAX,
    ))
    if generic_penalty and grounding < 0.30:
        # A task restatement can accidentally contain words such as "key" or
        # "need".  Without a decisive state fact it must not earn a bonus.
        raw = min(raw, 0.0)
    details.update({
        "raw": raw,
        "grounding": grounding,
        "action_support": action_support,
        "choice_support": choice_support,
        "generic_penalty": generic_penalty,
        "contradiction_penalty": contradiction_penalty,
        "is_singleton_valid_skill": len(valid_skills) == 1,
    })
    return details


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
        # Formatting failures already receive the actor-level invalid-action
        # penalty.  Treating the manager's "INVALID" sentinel as a repeated
        # task action creates a runaway -2 living penalty and collapses every
        # all-invalid trajectory to the same return before syntax can improve.
        recent_actions = [
            action for action in self.action_history
            if action not in {None, "INVALID"}
        ]
        penalty = 0.0
        if len(recent_actions) >= 3 and len(set(recent_actions[-3:])) == 1:
            penalty = -1.0
        if len(recent_actions) >= 4 and len(set(recent_actions[-4:])) == 1:
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
        # Give malformed responses a small one-step bootstrap gradient toward
        # the required schema.  Executable responses get no format living
        # reward, so this cannot make longer successful trajectories desirable.
        format_validity = float(np.clip(format_score, 0.0, 1.0))
        format_reward = invalid_format_bootstrap_reward(skill_name, format_validity)
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
