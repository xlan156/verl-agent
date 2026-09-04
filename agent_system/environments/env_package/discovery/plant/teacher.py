from __future__ import annotations

from itertools import combinations
import random
from typing import Any

from .skills import LEVELS, NUTRIENTS


Rule = dict[str, Any]
DEFAULT_SUBOPTIMAL_PROBABILITY = 0.6


def normalize_difficulty(value: Any) -> str:
    difficulty = str(value or "Normal").strip().lower()
    if difficulty not in {"easy", "normal", "challenge"}:
        raise ValueError(f"Unsupported Plant Nutrients difficulty: {value!r}")
    return difficulty


def _condition(nutrient: str, level: int) -> dict[str, Any]:
    return {"nutrient": nutrient, "level": int(level)}


def initial_candidates(difficulty: Any) -> list[Rule]:
    """Return every rule expressible by the selected Plant difficulty."""
    difficulty = normalize_difficulty(difficulty)
    if difficulty == "easy":
        return [
            {
                "rule_type": "presence",
                "nutrient": nutrient,
                "level": 3,
                "conditions": [_condition(nutrient, 3)],
            }
            for nutrient in NUTRIENTS
        ]
    if difficulty == "normal":
        return [
            {
                "rule_type": "equal",
                "nutrient": nutrient,
                "level": level,
                "conditions": [_condition(nutrient, level)],
            }
            for nutrient in NUTRIENTS
            for level in LEVELS.values()
        ]

    candidates: list[Rule] = []
    for nutrient1, nutrient2 in combinations(NUTRIENTS, 2):
        for level1 in LEVELS.values():
            for level2 in LEVELS.values():
                conditions = [
                    _condition(nutrient1, level1),
                    _condition(nutrient2, level2),
                ]
                for rule_type in ("xor", "and", "or"):
                    candidates.append(
                        {"rule_type": rule_type, "conditions": conditions}
                    )
    candidates.extend(
        {
            "rule_type": "not",
            "nutrient": nutrient,
            "level": level,
            "conditions": [_condition(nutrient, level)],
        }
        for nutrient in NUTRIENTS
        for level in LEVELS.values()
    )
    return candidates


def candidate_matches(rule: Rule, nutrients: dict[str, Any]) -> bool:
    matches = [
        int(nutrients.get(condition["nutrient"], 0) or 0)
        == int(condition["level"])
        for condition in rule.get("conditions", [])
    ]
    rule_type = rule.get("rule_type")
    if rule_type in {"presence", "equal"}:
        return bool(matches and matches[0])
    if rule_type == "not":
        return bool(matches and not matches[0])
    if rule_type == "and":
        return bool(matches and all(matches))
    if rule_type == "or":
        return any(matches)
    if rule_type == "xor":
        return sum(matches) == 1
    return False


def required_selections(rule: Rule) -> dict[str, int]:
    """Choose one deterministic field configuration that satisfies a rule."""
    conditions = list(rule.get("conditions", []))
    rule_type = rule.get("rule_type")
    if rule_type == "presence":
        return {}
    if rule_type in {"equal", "and", "or"}:
        return {
            str(condition["nutrient"]): int(condition["level"])
            for condition in conditions
        }
    if rule_type == "xor" and len(conditions) == 2:
        first, second = conditions
        forbidden = int(second["level"])
        alternate = next(level for level in LEVELS.values() if level != forbidden)
        return {
            str(first["nutrient"]): int(first["level"]),
            str(second["nutrient"]): alternate,
        }
    if rule_type == "not" and conditions:
        condition = conditions[0]
        forbidden = int(condition["level"])
        alternate = next(level for level in LEVELS.values() if level != forbidden)
        return {str(condition["nutrient"]): alternate}
    return {}


def field_satisfies_rule(rule: Rule, selections: dict[str, Any]) -> bool:
    return candidate_matches(rule, selections)


class PlantRuleBasedTeacher:
    """Observation-only teacher with reproducible, recoverable lapses."""

    def __init__(
        self,
        env: Any,
        suboptimal_probability: float = DEFAULT_SUBOPTIMAL_PROBABILITY,
        rng_seed: int | None = None,
    ) -> None:
        self.env = env
        self.suboptimal_probability = float(suboptimal_probability)
        if not 0.0 <= self.suboptimal_probability <= 1.0:
            raise ValueError("suboptimal_probability must be in [0, 1]")
        if rng_seed is None:
            rng_seed = (
                (int(getattr(env, "_seed", 0)) << 16)
                ^ int(getattr(env, "_thread_id", 0))
                ^ 0x71A9
            )
        self._rng = random.Random(rng_seed)
        self.last_selection_mode = None
        self.last_greedy_skill = None

    def _select(self, greedy: str, suboptimal: list[str] | None = None) -> str:
        self.last_greedy_skill = greedy
        self.last_selection_mode = "greedy"
        choices = [skill for skill in (suboptimal or []) if skill != greedy]
        if choices and self._rng.random() < self.suboptimal_probability:
            self.last_selection_mode = "suboptimal"
            return self._rng.choice(choices)
        return greedy

    @staticmethod
    def candidates(info: dict[str, Any]) -> list[Rule]:
        candidates = initial_candidates(info.get("plant_difficulty", "Normal"))
        for experiment in info.get("plant_experiment_memory", []):
            nutrients = experiment.get("nutrients", {})
            grew = bool(experiment.get("grew"))
            candidates = [
                candidate
                for candidate in candidates
                if candidate_matches(candidate, nutrients) == grew
            ]
        return candidates

    @staticmethod
    def _tools_ready(info: dict[str, Any]) -> bool:
        tools = set(info.get("plant_tools", []))
        if normalize_difficulty(info.get("plant_difficulty")) == "easy":
            return "soil nutrient meter" in tools
        return {"soil nutrient meter", "shovel", "seed jar"}.issubset(tools)

    def select_skill(self, info: dict[str, Any]) -> str | None:
        if bool(info.get("won", False)):
            return None
        difficulty = normalize_difficulty(info.get("plant_difficulty"))
        if not self._tools_ready(info):
            return self._select("collect_plant_tools")

        candidates = self.candidates(info)
        unmeasured = int(info.get("plant_unmeasured_plots", 0) or 0)
        measurements = len(info.get("plant_experiment_memory", []))
        if difficulty == "easy" and measurements < 2 and unmeasured > 0:
            return self._select("measure_next_pilot_plot")
        if len(candidates) != 1 and unmeasured > 0:
            return self._select("measure_next_pilot_plot")
        if len(candidates) != 1:
            return None

        rule = candidates[0]
        if difficulty == "easy":
            return self._select(
                f"select_{rule['nutrient']}",
                [
                    f"select_{nutrient}"
                    for nutrient in NUTRIENTS
                    if nutrient != rule["nutrient"]
                ],
            )

        active = info.get("plant_active_field")
        selections = info.get("plant_field_selections", {})
        if active is not None:
            active_selections = selections.get(str(active), {})
            for nutrient, value in required_selections(rule).items():
                if int(active_selections.get(nutrient, 0) or 0) != value:
                    level = next(
                        name for name, number in LEVELS.items() if number == value
                    )
                    wrong_levels = [
                        f"set_{nutrient}_{name}"
                        for name, number in LEVELS.items()
                        if number != value
                    ]
                    return self._select(f"set_{nutrient}_{level}", wrong_levels)
            return self._select("commit_field_configuration")

        committed = {int(value) for value in info.get("plant_committed_fields", [])}
        if 1 not in committed:
            return self._select("open_field_1_controller")
        if not field_satisfies_rule(rule, selections.get("1", {})):
            return None
        planted = info.get("plant_planted_counts", {}).get("1", 0)
        if planted < 2:
            return self._select("plant_seed_in_field_1", ["wait_for_growth"])
        if not info.get("won", False):
            return self._select("wait_for_growth")
        return None
