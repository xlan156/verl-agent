from __future__ import annotations

from typing import Any

from .skills import LEVELS, NUTRIENTS


class PlantRuleBasedTeacher:
    def __init__(self, env: Any) -> None:
        self.env = env

    @staticmethod
    def candidates(info: dict[str, Any]) -> list[tuple[str, int]]:
        candidates = [(nutrient, value) for nutrient in NUTRIENTS for value in LEVELS.values()]
        for experiment in info.get("plant_experiment_memory", []):
            nutrients = experiment.get("nutrients", {})
            grew = bool(experiment.get("grew"))
            candidates = [
                candidate for candidate in candidates
                if (int(nutrients.get(candidate[0], 0)) == candidate[1]) == grew
            ]
        return candidates

    def select_skill(self, info: dict[str, Any]) -> str | None:
        tools = set(info.get("plant_tools", []))
        if not {"soil nutrient meter", "shovel", "seed jar"}.issubset(tools):
            return "collect_plant_tools"
        active = info.get("plant_active_field")
        candidates = self.candidates(info)
        if active is not None:
            if not candidates:
                return "cancel_field_configuration"
            nutrient, value = candidates[0]
            selections = info.get("plant_field_selections", {}).get(str(active), {})
            if int(selections.get(nutrient, 0)) != value:
                level = next(name for name, number in LEVELS.items() if number == value)
                return f"set_{nutrient}_{level}"
            return "commit_field_configuration"
        if len(candidates) != 1 and info.get("plant_unmeasured_plots", 0) > 0:
            return "measure_next_pilot_plot"
        committed = {int(value) for value in info.get("plant_committed_fields", [])}
        if 1 not in committed:
            return "open_field_1_controller"
        planted = info.get("plant_planted_counts", {}).get("1", 0)
        if planted < 2:
            return "plant_seed_in_field_1"
        if not info.get("won", False):
            return "wait_for_growth"
        return None
