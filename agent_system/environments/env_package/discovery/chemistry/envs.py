from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import CombinatorialChemistrySkill
from .teacher import RulebasedAgentSkill
from .utils import (
    SKILL_NAMES,
    extract_detailed_status,
    format_rust_level,
    get_valid_discoveryworld_skills,
    is_dispenser_skill,
    is_remove_skill,
    reaction_signal_for_tuples,
)


class TaskAdapter:
    family = "chemistry"
    skill_names = tuple(SKILL_NAMES)

    @staticmethod
    def scenario_args(env: Any) -> dict[str, Any]:
        return {
            "numChemicals": 4,
            "minChemicals": 1,
            "chemicalMinAmount": env._max_chemical_n,
            "chemicalMaxAmount": env._max_chemical_n,
        }

    @staticmethod
    def initialize(env: Any) -> None:
        env.used_dispensers = dict.fromkeys(("A", "B", "C", "D"), False)

    @staticmethod
    def create_skill_runner(env: Any) -> CombinatorialChemistrySkill:
        return CombinatorialChemistrySkill(env)

    @staticmethod
    def create_teacher(env: Any) -> RulebasedAgentSkill:
        return RulebasedAgentSkill(env)

    @staticmethod
    def accepts(skill_name: str) -> bool:
        return skill_name in SKILL_NAMES

    @staticmethod
    def execute(runner: Any, skill_name: str) -> None:
        runner.skill_mapping[skill_name]()
        if is_dispenser_skill(skill_name) or is_remove_skill(skill_name):
            runner.settle_reactions(max_ticks=1)

    @staticmethod
    def enrich_info(env: Any, info: dict[str, Any]) -> None:
        ui = (info.get("raw_observation") or {}).get("ui", {})
        has_key, has_jar, in_jar, chemicals, rust = extract_detailed_status(ui)
        info.update(
            has_key=has_key,
            has_jar=has_jar,
            is_key_in_jar=in_jar,
            chemical_dict=deepcopy(chemicals),
            key_rust_level=rust,
            key_rust_status=format_rust_level(rust),
            key_is_rusted=None if rust is None else rust != "no rust",
            used_dispensers=dict(env.used_dispensers),
        )
        mixture = tuple(int(chemicals.get(name, 0) or 0) for name in "ABCD")
        info["current_reaction_signal"] = (
            reaction_signal_for_tuples(mixture, env._hidden_chemical_target)
            if in_jar and any(mixture) and env._hidden_chemical_target is not None
            else "not tested"
        )

    @staticmethod
    def valid_skills(info: dict[str, Any], env: Any) -> list[str]:
        return get_valid_discoveryworld_skills(info, env._max_chemical_n)

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
        from .state import log_state
        return log_state(info)
