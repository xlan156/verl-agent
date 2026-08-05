from __future__ import annotations

from typing import Any


DIMENSIONS = ("density", "temperature", "quantum_size", "radiation", "spectrum_channel_4")


class ReactorRuleBasedTeacher:
    def __init__(self, env: Any) -> None:
        self.env = env

    @staticmethod
    def infer_model(memory: dict[str, Any]) -> tuple[str, int, int] | None:
        known = [memory.get("1"), memory.get("2")]
        if any(not item or item.get("known_frequency") is None for item in known):
            return None
        matches = []
        for dimension in DIMENSIONS:
            xs = [float(item["readings"][dimension]) for item in known]
            ys = [float(item["known_frequency"]) for item in known]
            for slope in range(90, 110):
                for offset in range(90, 110):
                    if all(abs(round(slope * x + offset, 2) - y) <= 0.011 for x, y in zip(xs, ys)):
                        matches.append((dimension, slope, offset))
        return matches[0] if len(matches) == 1 else None

    def select_skill(self, info: dict[str, Any]) -> str | None:
        if len(info.get("reactor_instruments", [])) < 5:
            return "collect_reactor_instruments"
        memory = info.get("reactor_experiment_memory", {})
        for index in (1, 2):
            if str(index) not in memory:
                return f"measure_crystal_{index}"
        model = self.infer_model(memory)
        for index in (3, 4):
            if str(index) not in memory:
                return f"measure_crystal_{index}"
            reactor = info.get("reactor_states", {}).get(str(index), {})
            if not reactor.get("has_crystal"):
                return f"install_crystal_{index}"
            if not reactor.get("activated") and model is not None:
                dimension, slope, offset = model
                value = float(memory[str(index)]["readings"][dimension])
                return f"set_reactor_{index}_frequency_{int(round(slope * value + offset, 2))}"
        return None
