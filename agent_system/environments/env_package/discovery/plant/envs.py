from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import LEVELS, NUTRIENTS, STATIC_SKILLS, PlantSkill
from .teacher import (
    PlantRuleBasedTeacher,
    normalize_difficulty,
    required_selections,
)


class TaskAdapter:
    family = "plant"
    skill_names = tuple(STATIC_SKILLS)

    @staticmethod
    def scenario_args(env: Any) -> dict[str, Any]:
        difficulty = normalize_difficulty(env._difficulty)
        if difficulty not in {"easy", "normal", "challenge"}:
            raise ValueError(
                "Plant Nutrients supports difficulty='Easy', 'Normal', or 'Challenge'"
            )
        return {}

    @staticmethod
    def initialize(env: Any) -> None:
        pass

    @staticmethod
    def create_skill_runner(env: Any) -> PlantSkill:
        return PlantSkill(env)

    @staticmethod
    def create_teacher(env: Any) -> PlantRuleBasedTeacher:
        return PlantRuleBasedTeacher(env)

    @staticmethod
    def accepts(skill_name: str) -> bool:
        return skill_name in STATIC_SKILLS

    @staticmethod
    def execute(runner: Any, skill_name: str) -> None:
        runner.skill_mapping[skill_name]()

    @staticmethod
    def enrich_info(env: Any, info: dict[str, Any]) -> None:
        inventory = env._api.world.getUserAgents()[0].getInventory()
        tools = sorted(
            obj.name
            for obj in inventory
            if obj.name in {"soil nutrient meter", "shovel", "seed jar"}
        )
        memory = deepcopy(getattr(env, "plant_experiment_memory", []))
        runner = env._skill_runner
        total_pilot_plots = len(runner._pilot_plots()) if runner is not None else 0
        info.update(
            plant_difficulty=normalize_difficulty(env._difficulty).title(),
            plant_tools=tools,
            plant_experiment_memory=memory,
            plant_unmeasured_plots=max(0, total_pilot_plots - len(memory)),
            plant_selected_nutrient=getattr(
                env, "plant_selected_nutrient", None
            ),
            plant_active_field=getattr(env, "plant_active_field", None),
            plant_field_selections={
                str(key): deepcopy(value)
                for key, value in getattr(env, "plant_field_selections", {}).items()
            },
            plant_committed_fields=sorted(
                getattr(env, "plant_committed_fields", set())
            ),
            plant_planted_counts={
                str(key): value
                for key, value in getattr(env, "plant_planted_counts", {}).items()
            },
            plant_growth_waits=int(getattr(env, "plant_growth_waits", 0)),
        )
        info["plant_rule_candidates"] = PlantRuleBasedTeacher.candidates(info)

    @staticmethod
    def valid_skills(info: dict[str, Any], env: Any) -> list[str]:
        """Expose executable choices appropriate to the observable Plant phase."""
        if bool(info.get("won", False)):
            return []

        difficulty = normalize_difficulty(info.get("plant_difficulty"))
        tools = set(info.get("plant_tools", []))
        required_tools = (
            {"soil nutrient meter"}
            if difficulty == "easy"
            else {"soil nutrient meter", "shovel", "seed jar"}
        )
        if not required_tools.issubset(tools):
            return ["collect_plant_tools"]

        candidates = info.get("plant_rule_candidates", [])
        unmeasured = int(info.get("plant_unmeasured_plots", 0) or 0)
        measurements = len(info.get("plant_experiment_memory", []))
        if difficulty == "easy" and measurements < 2 and unmeasured > 0:
            return ["measure_next_pilot_plot"]
        if len(candidates) != 1:
            return ["measure_next_pilot_plot"] if unmeasured > 0 else []

        if difficulty == "easy":
            # All five answers are executable. The model must infer which one
            # agrees with the observable experiments.
            return [f"select_{nutrient}" for nutrient in NUTRIENTS]

        active = info.get("plant_active_field")
        if active is not None:
            if int(active) != 1:
                return ["cancel_field_configuration"]
            candidate = candidates[0]
            nutrients = tuple(required_selections(candidate))
            # Expose all levels for each rule-relevant nutrient. This keeps a
            # real policy choice for compound Challenge rules while avoiding
            # irrelevant setters for the other nutrients.
            return [
                *(
                    f"set_{nutrient}_{level_name}"
                    for nutrient in nutrients
                    for level_name in LEVELS
                ),
                "commit_field_configuration",
                "cancel_field_configuration",
            ]

        committed = {int(field) for field in info.get("plant_committed_fields", [])}
        planted = {
            int(field): int(count)
            for field, count in info.get("plant_planted_counts", {}).items()
        }
        if 1 not in committed:
            return ["open_field_1_controller"]
        if planted.get(1, 0) < 2:
            return ["plant_seed_in_field_1", "wait_for_growth"]
        return ["wait_for_growth"]

    state_signature = staticmethod(state_signature)
    target_progress_reward = staticmethod(target_progress_reward)

    @staticmethod
    def terminal_reward_adjustment(env: Any, info: dict[str, Any]) -> float:
        from .rewards import terminal_reward_adjustment
        return terminal_reward_adjustment(env, info)

    @staticmethod
    def build_prompt(raw_obs, info, records, config, init=False):
        from .state import build_prompt
        return build_prompt(raw_obs, info, records, config, init)

    @staticmethod
    def build_anchor(raw_obs, info, records, config):
        from .state import build_anchor
        return build_anchor(raw_obs, info, records, config)

    @staticmethod
    def memory_record(previous, current, action):
        from .state import memory_record
        return memory_record(previous, current, action)

    @staticmethod
    def log_state(info):
        from .state import prompt_state
        return prompt_state(info)
