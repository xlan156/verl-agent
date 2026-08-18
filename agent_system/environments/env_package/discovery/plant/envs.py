from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import LEVELS, STATIC_SKILLS, PlantNormalSkill
from .teacher import PlantRuleBasedTeacher


class TaskAdapter:
    family = "plant"
    skill_names = tuple(STATIC_SKILLS)

    @staticmethod
    def scenario_args(env: Any) -> dict[str, Any]:
        if str(env._difficulty or "").strip().lower() != "normal":
            raise ValueError("The Plant MVP currently supports difficulty='Normal' only")
        return {}

    @staticmethod
    def initialize(env: Any) -> None:
        pass

    @staticmethod
    def create_skill_runner(env: Any) -> PlantNormalSkill:
        return PlantNormalSkill(env)

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
            obj.name for obj in inventory
            if obj.name in {"soil nutrient meter", "shovel", "seed jar"}
        )
        memory = deepcopy(getattr(env, "plant_experiment_memory", []))
        from .teacher import PlantRuleBasedTeacher
        info.update(
            plant_tools=tools,
            plant_experiment_memory=memory,
            plant_unmeasured_plots=max(0, 12 - len(memory)),
            plant_active_field=getattr(env, "plant_active_field", None),
            plant_field_selections={
                str(key): deepcopy(value)
                for key, value in getattr(env, "plant_field_selections", {}).items()
            },
            plant_committed_fields=sorted(getattr(env, "plant_committed_fields", set())),
            plant_planted_counts={
                str(key): value
                for key, value in getattr(env, "plant_planted_counts", {}).items()
            },
            plant_growth_waits=int(getattr(env, "plant_growth_waits", 0)),
        )
        info["plant_rule_candidates"] = [
            {"nutrient": nutrient, "level": value}
            for nutrient, value in PlantRuleBasedTeacher.candidates(info)
        ]

    @staticmethod
    def valid_skills(info: dict[str, Any], env: Any) -> list[str]:
        """Expose only actions that advance the observable Plant phase."""
        if bool(info.get("won", False)):
            return []

        tools = set(info.get("plant_tools", []))
        tools_ready = {
            "soil nutrient meter", "shovel", "seed jar"
        }.issubset(tools)
        if not tools_ready:
            return ["collect_plant_tools"]

        candidates = info.get("plant_rule_candidates", [])
        unmeasured = int(info.get("plant_unmeasured_plots", 0) or 0)
        active = info.get("plant_active_field")
        if active is not None:
            # This uses only the sole candidate inferred from observable
            # experiments; no simulator-hidden nutrient target is accessed.
            if len(candidates) != 1 or int(active) != 1:
                return ["cancel_field_configuration"]
            candidate = candidates[0]
            nutrient = str(candidate["nutrient"])
            # Keep a genuine policy decision: the candidate identifies the
            # relevant nutrient, while the model must read its level and the
            # current selection to choose a setter or commit. These are all
            # executable in the currently open controller.
            return [
                *(f"set_{nutrient}_{level_name}" for level_name in LEVELS),
                "commit_field_configuration",
                "cancel_field_configuration",
            ]

        if len(candidates) != 1:
            return ["measure_next_pilot_plot"] if unmeasured > 0 else []

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
