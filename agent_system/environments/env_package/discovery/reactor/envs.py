from __future__ import annotations

from copy import deepcopy
from typing import Any

from .rewards import state_signature, target_progress_reward
from .skills import FREQUENCY_SKILL_RE, STATIC_SKILLS, ReactorSkill
from .teacher import ReactorRuleBasedTeacher


class TaskAdapter:
    family = "reactor"
    skill_names = tuple(STATIC_SKILLS)

    @staticmethod
    def scenario_args(env: Any) -> dict[str, Any]:
        difficulty = str(env._difficulty or "").strip().lower()
        if difficulty not in {"easy", "normal", "challenge"}:
            raise ValueError("Reactor Lab supports difficulty='Easy', 'Normal', or 'Challenge'")
        env._difficulty = difficulty.title()
        return {}

    @staticmethod
    def initialize(env: Any) -> None:
        pass

    @staticmethod
    def create_skill_runner(env: Any) -> ReactorSkill:
        return ReactorSkill(env)

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
    def execute(runner: ReactorSkill, skill_name: str) -> None:
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
                "frequency": reactor.attributes.get("resonanceFreq"),
                "has_crystal": any(
                    str(getattr(obj, "type", "")).lower().replace(" ", "") == "quantumcrystal"
                    for obj in reactor.contents
                ),
                "activated": bool(reactor.attributes.get("isActivated", False)),
            }
        memory = deepcopy(getattr(env, "reactor_experiment_memory", {}))
        model = ReactorRuleBasedTeacher.infer_model(memory)
        info.update(
            reactor_difficulty=str(getattr(env, "_difficulty", "Normal")),
            reactor_instruments=instruments,
            reactor_experiment_memory=memory,
            reactor_states=states,
            reactor_inferred_model=(model if model else None),
            reactor_dynamic_frequency=True,
        )

    @staticmethod
    def valid_skills(info: dict[str, Any], env: Any) -> list[str]:
        difficulty = str(getattr(env, "_difficulty", "Normal")).lower()
        crystal_count = {"easy": 3, "normal": 4, "challenge": 5}[difficulty]
        model = info.get("reactor_inferred_model")
        memory = info.get("reactor_experiment_memory", {})
        target_indices = {"easy": (3,), "normal": (3, 4), "challenge": (4, 5)}[difficulty]

        # Phase-aware, but retain all safe choices within a phase. Measuring
        # any unmeasured crystal is observable and safe, so it remains a real
        # policy choice instead of turning the task into pure imitation.
        required_instruments = 1 if difficulty == "easy" else 5
        if len(info.get("reactor_instruments", [])) < required_instruments:
            return ["collect_reactor_instruments"]
        unmeasured = [index for index in range(1, crystal_count + 1)
                      if str(index) not in memory]
        if unmeasured and model is None:
            return [f"measure_crystal_{index}" for index in unmeasured]

        reactors = info.get("reactor_states", {})
        if model:
            unmeasured_targets = [index for index in target_indices if str(index) not in memory]
            installable = [index for index in target_indices
                           if str(index) in memory
                           and not reactors.get(str(index), {}).get("has_crystal")]
            phase_actions = [*(f"install_crystal_{index}" for index in installable),
                             *(f"measure_crystal_{index}" for index in unmeasured_targets)]
            # Tuning an already installed target is also safe while the other
            # target is being measured, preserving a real policy choice.
            for index in target_indices:
                record = memory.get(str(index))
                reactor = reactors.get(str(index), {})
                if record and reactor.get("has_crystal") and not reactor.get("activated"):
                    value = float(record["readings"][model["dimension"]])
                    target_value = (model["slope"] * value + model["offset"] if model["degree"] == 1
                                    else model["a"] * value * value + model["b"] * value + model["c"])
                    phase_actions.append(f"set_reactor_{index}_frequency_<number>")
            if phase_actions:
                return phase_actions

        uninstalled = [index for index in target_indices
                       if not reactors.get(str(index), {}).get("has_crystal")]
        if uninstalled:
            return [f"install_crystal_{index}" for index in uninstalled]

        actions = []
        if model:
            for index in target_indices:
                record = memory.get(str(index))
                reactor = reactors.get(str(index), {})
                if record and not reactor.get("activated"):
                    value = float(record["readings"][model["dimension"]])
                    target_value = (model["slope"] * value + model["offset"] if model["degree"] == 1
                                    else model["a"] * value * value + model["b"] * value + model["c"])
                    actions.append(f"set_reactor_{index}_frequency_<number>")
        return actions

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
