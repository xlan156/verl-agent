from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import FREQUENCY_SKILL_RE, STATIC_SKILLS, ReactorNormalSkill
from .teacher import ReactorRuleBasedTeacher


class TaskAdapter:
    family = "reactor"
    skill_names = tuple(STATIC_SKILLS)

    @staticmethod
    def scenario_args(env: Any) -> dict[str, Any]:
        if str(env._difficulty or "").strip().lower() != "normal":
            raise ValueError("The Reactor MVP currently supports difficulty='Normal' only")
        return {}

    @staticmethod
    def initialize(env: Any) -> None:
        pass

    @staticmethod
    def create_skill_runner(env: Any) -> ReactorNormalSkill:
        return ReactorNormalSkill(env)

    @staticmethod
    def create_teacher(env: Any) -> ReactorRuleBasedTeacher:
        return ReactorRuleBasedTeacher(env)

    @staticmethod
    def accepts(skill_name: str) -> bool:
        match = FREQUENCY_SKILL_RE.fullmatch(skill_name)
        return skill_name in STATIC_SKILLS or (
            match is not None and 0 <= int(match.group(2)) <= 10000
        )

    @staticmethod
    def execute(runner: ReactorNormalSkill, skill_name: str) -> None:
        runner.execute(skill_name)

    @staticmethod
    def _reactor_number(obj: Any) -> int | None:
        value = obj.attributes.get("reactorNum")
        if value is not None:
            return int(value)
        for token in obj.name.split():
            if token.strip("#").isdigit():
                return int(token.strip("#"))
        return None

    @classmethod
    def enrich_info(cls, env: Any, info: dict[str, Any]) -> None:
        world = env._api.world
        inventory = world.getUserAgents()[0].getInventory()
        instruments = sorted(
            obj.name for obj in inventory
            if str(obj.type).lower().replace(" ", "") in {
                "densitometer", "thermometer", "microscope", "radiationmeter", "spectrometer"
            }
        )
        states = {}
        for reactor in world.getAllWorldObjects():
            if str(getattr(reactor, "type", "")).lower().replace(" ", "") != "crystalreactor":
                continue
            index = cls._reactor_number(reactor)
            states[str(index)] = {
                "has_crystal": any(
                    str(getattr(obj, "type", "")).lower().replace(" ", "") == "quantumcrystal"
                    for obj in reactor.contents
                ),
                "activated": bool(reactor.attributes.get("isActivated", False)),
            }
        memory = deepcopy(getattr(env, "reactor_experiment_memory", {}))
        model = ReactorRuleBasedTeacher.infer_model(memory)
        info.update(
            reactor_instruments=instruments,
            reactor_experiment_memory=memory,
            reactor_states=states,
            reactor_inferred_model=(
                {"dimension": model[0], "slope": model[1], "offset": model[2]}
                if model else None
            ),
        )

    @staticmethod
    def valid_skills(info: dict[str, Any], env: Any) -> list[str]:
        skills = list(STATIC_SKILLS)
        model = info.get("reactor_inferred_model")
        if model:
            for index in (3, 4):
                record = info.get("reactor_experiment_memory", {}).get(str(index))
                if record:
                    value = float(record["readings"][model["dimension"]])
                    target = int(round(model["slope"] * value + model["offset"], 2))
                    skills.append(f"set_reactor_{index}_frequency_{target}")
        return skills

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
