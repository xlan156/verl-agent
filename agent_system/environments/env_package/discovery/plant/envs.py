from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import STATIC_SKILLS, PlantNormalSkill
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
        active = info.get("plant_active_field")
        if active is not None:
            return [
                skill for skill in STATIC_SKILLS
                if skill.startswith("set_") or skill in {
                    "commit_field_configuration", "cancel_field_configuration"
                }
            ]

        valid = []
        tools = set(info.get("plant_tools", []))
        tools_ready = {
            "soil nutrient meter", "shovel", "seed jar"
        }.issubset(tools)
        if not tools_ready:
            valid.append("collect_plant_tools")

        candidates = info.get("plant_rule_candidates", [])
        unmeasured = int(info.get("plant_unmeasured_plots", 0) or 0)
        if tools_ready and len(candidates) != 1 and unmeasured > 0:
            valid.append("measure_next_pilot_plot")

        committed = {int(field) for field in info.get("plant_committed_fields", [])}
        planted = {
            int(field): int(count)
            for field, count in info.get("plant_planted_counts", {}).items()
        }
        for field in (1, 2, 3):
            if field not in committed:
                valid.append(f"open_field_{field}_controller")
            elif planted.get(field, 0) < 2:
                valid.append(f"plant_seed_in_field_{field}")

        if any(count >= 2 for count in planted.values()):
            valid.append("wait_for_growth")

        return valid

    state_signature = staticmethod(state_signature)
    target_progress_reward = staticmethod(target_progress_reward)

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
