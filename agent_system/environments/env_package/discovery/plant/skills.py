from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..common import ObservableSkillRunner


NUTRIENTS = ("potassium", "titanium", "lithium", "thorium", "barium")
LEVELS = {"low": 1, "medium": 2, "high": 3}
STATIC_SKILLS = (
    "collect_plant_tools",
    "measure_next_pilot_plot",
    *(f"open_field_{field}_controller" for field in (1, 2, 3)),
    *(f"set_{nutrient}_{level}" for nutrient in NUTRIENTS for level in LEVELS),
    "commit_field_configuration",
    "cancel_field_configuration",
    *(f"plant_seed_in_field_{field}" for field in (1, 2, 3)),
    "wait_for_growth",
)


class PlantNormalSkill(ObservableSkillRunner):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        env.plant_experiment_memory = []
        env.plant_measured_plot_uuids = set()
        env.plant_active_field = None
        env.plant_field_selections = {1: {}, 2: {}, 3: {}}
        env.plant_committed_fields = set()
        env.plant_planted_counts = {1: 0, 2: 0, 3: 0}
        env.plant_growth_waits = 0
        self.skill_mapping = {
            "collect_plant_tools": self.collect_plant_tools,
            "measure_next_pilot_plot": self.measure_next_pilot_plot,
            "commit_field_configuration": self.commit_field_configuration,
            "cancel_field_configuration": self.cancel_field_configuration,
            "wait_for_growth": self.wait_for_growth,
        }
        for field in (1, 2, 3):
            self.skill_mapping[f"open_field_{field}_controller"] = (
                lambda field=field: self.open_controller(field)
            )
            self.skill_mapping[f"plant_seed_in_field_{field}"] = (
                lambda field=field: self.plant_seed(field)
            )
        for nutrient in NUTRIENTS:
            for level in LEVELS:
                self.skill_mapping[f"set_{nutrient}_{level}"] = (
                    lambda nutrient=nutrient, level=level: self.set_nutrient(nutrient, level)
                )

    def _one(self, obj_type: str, predicate=lambda obj: True) -> Any | None:
        return next((obj for obj in self.objects(obj_type) if predicate(obj)), None)

    def _controller(self, field: int) -> Any | None:
        return self._one(
            "soilcontroller",
            lambda obj: int(obj.attributes.get("fieldNum", -1)) == field,
        )

    def _pilot_plots(self) -> list[Any]:
        return self.sort_objects(
            obj for obj in self.objects("soil")
            if obj.attributes.get("testField") is False
        )

    def _field_plots(self, field: int) -> list[Any]:
        controller = self._controller(field)
        uuids = set((controller.attributes.get("fieldTileUUIDs") or [])) if controller else set()
        return self.sort_objects(obj for obj in self.objects("soil") if obj.uuid in uuids)

    def _plot_has_plant(self, plot: Any) -> bool:
        location = tuple(plot.getWorldLocation())
        return any(
            tuple(obj.getWorldLocation()) == location
            and str(getattr(obj, "type", "")).lower() in {"mushroom", "plant"}
            for obj in self.world.getAllWorldObjects()
        )

    def collect_plant_tools(self) -> None:
        acquired = []
        for obj_type in ("soilnutrientmeter", "shovel", "jar"):
            obj = self._one(obj_type, lambda value: obj_type != "jar" or "seed" in value.name)
            if obj is not None and self.pickup(obj):
                acquired.append(obj.name)
        self.finish(len(acquired) == 3, f"Collected plant tools: {', '.join(acquired)}")

    def measure_next_pilot_plot(self) -> None:
        meter = self._one("soilnutrientmeter")
        if meter is None or not self._inside_inventory(meter):
            self.finish(False, "Collect the soil nutrient meter first")
            return
        plot = next(
            (obj for obj in self._pilot_plots() if obj.uuid not in self.env.plant_measured_plot_uuids),
            None,
        )
        if plot is None:
            self.finish(False, "All pilot plots have already been measured")
            return
        success = self.teleport(plot) and self.act("USE", meter, plot)
        if success:
            record = {
                "plot": len(self.env.plant_experiment_memory) + 1,
                "nutrients": deepcopy(plot.attributes.get("soilNutrients", {})),
                "grew": self._plot_has_plant(plot),
            }
            self.env.plant_experiment_memory.append(record)
            self.env.plant_measured_plot_uuids.add(plot.uuid)
            self.finish(True, f"Recorded observable pilot experiment: {record}")

    def open_controller(self, field: int) -> None:
        controller = self._controller(field)
        if controller is None:
            self.finish(False, f"Field {field} is unavailable")
            return
        if field in self.env.plant_committed_fields:
            self.finish(False, f"Field {field} was already committed")
            return
        success = self.talk_to(controller)
        if success:
            self.env.plant_active_field = field
            self.finish(True, f"Opened field {field} nutrient controller")

    def set_nutrient(self, nutrient: str, level: str) -> None:
        field = self.env.plant_active_field
        if field is None or not self.agent.isInDialog():
            self.finish(False, "Open a field controller first")
            return
        success = self.choose_matching(f"Set {nutrient.title()} Level")
        success = success and self.choose_matching(f"{nutrient.title()} level: {level}")
        success = success and self.choose_matching("Back to main menu")
        if success:
            self.env.plant_field_selections[field][nutrient] = LEVELS[level]
            self.finish(True, f"Field {field}: {nutrient}={level}")

    def commit_field_configuration(self) -> None:
        field = self.env.plant_active_field
        if field is None or not self.agent.isInDialog():
            self.finish(False, "Open a field controller first")
            return
        success = self.choose_matching("Imprint current selections")
        if success:
            self.env.plant_committed_fields.add(field)
            self.env.plant_active_field = None
            self.finish(True, f"Committed field {field} configuration")

    def cancel_field_configuration(self) -> None:
        field = self.env.plant_active_field
        success = bool(field is not None) and self.choose_matching("Cancel and exit")
        if success:
            self.env.plant_active_field = None
        self.finish(success, "Cancelled controller" if success else "No controller is open")

    def _inventory_seed(self) -> Any | None:
        return next(
            (
                obj for obj in self.world.getAllWorldObjects()
                if "seed" in str(getattr(obj, "type", "")).lower()
                and self._inside_inventory(obj)
            ),
            None,
        )

    def plant_seed(self, field: int) -> None:
        if field not in self.env.plant_committed_fields:
            self.finish(False, f"Commit field {field} before planting")
            return
        shovel = self._one("shovel")
        seed = self._inventory_seed()
        if shovel is None or not self._inside_inventory(shovel) or seed is None:
            self.finish(False, "Collect the shovel and seed jar first")
            return
        plot = next(
            (
                item for item in self._field_plots(field)
                if not item.attributes.get("hasHole", False)
                and not any("seed" in str(getattr(child, "type", "")).lower() for child in item.contents)
                and not self._plot_has_plant(item)
            ),
            None,
        )
        if plot is None:
            self.finish(False, f"No unplanted soil remains in field {field}")
            return
        success = self.teleport(plot) and self.act("USE", shovel, plot)
        dirt = next(
            (obj for obj in self.objects("dirt") if self._inside_inventory(obj)),
            None,
        )
        success = success and self.act("PUT", seed, plot)
        success = success and dirt is not None and self.act("PUT", dirt, plot)
        if success:
            self.env.plant_planted_counts[field] += 1
            self.env.plant_growth_waits = 0
            self.finish(True, f"Planted one seed in field {field}")

    def wait_for_growth(self) -> None:
        for _ in range(5):
            self.api.tick()
        self.env.plant_growth_waits += 1
        self.update_ui()
        self.finish(True, "Waited five world ticks for planted seeds to grow")
