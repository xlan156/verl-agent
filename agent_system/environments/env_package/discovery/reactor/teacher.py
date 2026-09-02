from __future__ import annotations

from typing import Any

DIMENSIONS = ("density", "temperature", "quantum_size", "radiation", "spectrum_channel_4")


class ReactorRuleBasedTeacher:
    def __init__(self, env: Any) -> None:
        self.env = env

    @staticmethod
    def infer_model(memory: dict[str, Any]) -> dict[str, Any] | None:
        known = [item for _, item in sorted(memory.items()) if item.get("known_frequency") is not None]
        if len(known) < 2:
            return None
        matches = []
        for dimension in DIMENSIONS:
            xs = [float(item["readings"][dimension]) for item in known]
            ys = [float(item["known_frequency"]) for item in known]
            if len(known) == 2 and abs(xs[1] - xs[0]) > 1e-9:
                slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
                offset = ys[0] - slope * xs[0]
                if 49.5 <= slope <= 110.5 and -0.5 <= offset <= 110.5 and abs(slope - round(slope)) < .01 and abs(offset - round(offset)) < .01 and all(abs(round(slope * x + offset, 2) - y) <= .011 for x, y in zip(xs, ys)):
                    matches.append({"dimension": dimension, "degree": 1, "slope": slope, "offset": offset})
            elif len(known) >= 3:
                x1, x2, x3 = xs[:3]
                y1, y2, y3 = ys[:3]
                denominator = (x1-x2) * (x1-x3) * (x2-x3)
                if abs(denominator) < 1e-9:
                    continue
                a = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denominator
                b = (x3*x3*(y1-y2) + x2*x2*(y3-y1) + x1*x1*(y2-y3)) / denominator
                c = y1 - a*x1*x1 - b*x1
                if 9.5 <= a <= 20.5 and 19.5 <= b <= 40.5 and 19.5 <= c <= 850.5 and abs(a-round(a)) < .2 and abs(b-round(b)) < .2 and abs(c-round(c)) < .2 and all(abs(round(a*x*x + b*x + c, 2) - y) <= .011 for x, y in zip(xs, ys)):
                    matches.append({"dimension": dimension, "degree": 2, "a": a, "b": b, "c": c})
        return matches[0] if len(matches) == 1 else None

    def select_skill(self, info: dict[str, Any]) -> str | None:
        required_instruments = 1 if str(getattr(self.env, "_difficulty", "Normal")).lower() == "easy" else 5
        if len(info.get("reactor_instruments", [])) < required_instruments:
            return "collect_reactor_instruments"
        difficulty = str(getattr(self.env, "_difficulty", "Normal")).lower()
        known_indices = (1, 2, 3) if difficulty == "challenge" else (1, 2)
        target_indices = {"easy": (3,), "normal": (3, 4), "challenge": (4, 5)}[difficulty]
        memory = info.get("reactor_experiment_memory", {})
        for index in known_indices:
            if str(index) not in memory:
                return f"measure_crystal_{index}"
        model = self.infer_model(memory)
        for index in target_indices:
            if str(index) not in memory:
                return f"measure_crystal_{index}"
            reactor = info.get("reactor_states", {}).get(str(index), {})
            if not reactor.get("has_crystal"):
                return f"install_crystal_{index}"
            if not reactor.get("activated") and model is not None:
                value = float(memory[str(index)]["readings"][model["dimension"]])
                target = (model["slope"] * value + model["offset"] if model["degree"] == 1 else
                          model["a"] * value * value + model["b"] * value + model["c"])
                return f"set_reactor_{index}_frequency_{int(round(target, 2))}"
        return None
