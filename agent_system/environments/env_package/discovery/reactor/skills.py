from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from ..common import ObservableSkillRunner


INSTRUMENT_TYPES = (
    "densitometer", "thermometer", "microscope", "radiationmeter", "spectrometer"
)
STATIC_SKILLS = (
    "collect_reactor_instruments",
    *(f"measure_crystal_{index}" for index in (1, 2, 3, 4)),
    "install_crystal_3",
    "install_crystal_4",
)
FREQUENCY_SKILL_RE = re.compile(r"^set_reactor_([34])_frequency_(\d{1,5})$")


class ReactorNormalSkill(ObservableSkillRunner):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        env.reactor_experiment_memory = {}
        env.reactor_installed = set()
        self.skill_mapping = {
            "collect_reactor_instruments": self.collect_instruments,
            **{
                f"measure_crystal_{index}": lambda index=index: self.measure_crystal(index)
                for index in (1, 2, 3, 4)
            },
            "install_crystal_3": lambda: self.install_crystal(3),
            "install_crystal_4": lambda: self.install_crystal(4),
        }

    def _numbered(self, obj_type: str, index: int) -> Any | None:
        pattern = re.compile(rf"#{index}\b|\b{index}$", re.IGNORECASE)
        return next(
            (
                obj for obj in self.objects(obj_type)
                if int(obj.attributes.get("reactorNum", -1)) == index or pattern.search(obj.name)
            ),
            None,
        )

    def _crystal(self, index: int) -> Any | None:
        return self._numbered("quantumcrystal", index)

    def _reactor(self, index: int) -> Any | None:
        return self._numbered("crystalreactor", index)

    def collect_instruments(self) -> None:
        acquired = []
        for obj_type in INSTRUMENT_TYPES:
            obj = next(iter(self.objects(obj_type)), None)
            if obj is not None and self.pickup(obj):
                acquired.append(obj.name)
        self.finish(
            len(acquired) == len(INSTRUMENT_TYPES),
            f"Collected reactor instruments: {', '.join(acquired)}",
        )

    @staticmethod
    def _public_readings(crystal: Any) -> dict[str, float]:
        material = (crystal.attributes.get("materials") or [{}])[0]
        spectrum = material.get("spectrum") or []
        return {
            "density": float(crystal.attributes.get("density")),
            "temperature": float(crystal.attributes.get("temperatureC")),
            "quantum_size": float(crystal.attributes.get("quantumSize")),
            "radiation": float(material.get("radiationusvh")),
            "spectrum_channel_4": float(spectrum[4]),
        }

    def _containing_reactor(self, crystal: Any) -> Any | None:
        parent = getattr(crystal, "parentContainer", None)
        while parent is not None:
            if str(getattr(parent, "type", "")).lower().replace(" ", "") == "crystalreactor":
                return parent
            parent = getattr(parent, "parentContainer", None)
        return None

    def _observe_reactor_frequency(self, reactor: Any) -> float | None:
        if not self.talk_to(reactor):
            return None
        # This attribute is copied only after opening the control dialog, where
        # the identical value is displayed to the policy/player.
        value = float(reactor.attributes.get("resonanceFreq"))
        self.choose_matching("Exit")
        return value

    def measure_crystal(self, index: int) -> None:
        crystal = self._crystal(index)
        instruments = [next(iter(self.objects(kind)), None) for kind in INSTRUMENT_TYPES]
        if crystal is None or any(item is None or not self._inside_inventory(item) for item in instruments):
            self.finish(False, "Collect all instruments first")
            return
        original_reactor = self._containing_reactor(crystal)
        known_frequency = (
            self._observe_reactor_frequency(original_reactor)
            if original_reactor is not None and index in (1, 2)
            else None
        )
        if not self.pickup(crystal):
            self.finish(False, f"Could not pick up crystal {index}")
            return
        messages = {}
        success = True
        for instrument in instruments:
            success = self.act("USE", instrument, crystal) and success
            messages[instrument.name] = self.env._last_action_result.get("message", "")
        if success:
            record = {
                "readings": self._public_readings(crystal),
                "instrument_messages": messages,
                "known_frequency": known_frequency,
            }
            self.env.reactor_experiment_memory[str(index)] = record
        if original_reactor is not None and index in (1, 2):
            success = self.teleport(original_reactor) and self.act("PUT", crystal, original_reactor) and success
        self.finish(success, f"Recorded all observable readings for crystal {index}")

    def install_crystal(self, index: int) -> None:
        crystal = self._crystal(index)
        reactor = self._reactor(index)
        if crystal is None or reactor is None:
            self.finish(False, f"Crystal/reactor {index} is unavailable")
            return
        if self._containing_reactor(crystal) is reactor:
            self.env.reactor_installed.add(index)
            self.finish(True, f"Crystal {index} is already installed")
            return
        success = self.pickup(crystal) and self.teleport(reactor) and self.act("PUT", crystal, reactor)
        if success:
            self.env.reactor_installed.add(index)
        self.finish(success, f"Installed crystal {index} in reactor {index}")

    def set_frequency(self, reactor_index: int, target: int) -> None:
        if not 0 <= target <= 10000:
            self.finish(False, "Frequency must be between 0 and 10000 Hz")
            return
        reactor = self._reactor(reactor_index)
        if reactor is None or not reactor.contents:
            self.finish(False, f"Install crystal {reactor_index} first")
            return
        if not self.talk_to(reactor):
            return
        success = True
        for _ in range(40):
            if reactor.attributes.get("isActivated", False):
                break
            current = int(float(reactor.attributes.get("resonanceFreq", 0)))
            delta = target - current
            if delta == 0:
                break
            magnitude = next(value for value in (1000, 100, 10, 1) if abs(delta) >= value)
            direction = "Increase" if delta > 0 else "Decrease"
            success = self.choose_matching(f"{direction} frequency by {magnitude} Hz") and success
        activated = bool(reactor.attributes.get("isActivated", False))
        if self.agent.isInDialog():
            self.choose_matching("Exit")
        self.finish(success and activated, f"Reactor {reactor_index} set near {target} Hz; activated={activated}")

    def execute(self, skill_name: str) -> None:
        function = self.skill_mapping.get(skill_name)
        if function is not None:
            function()
            return
        match = FREQUENCY_SKILL_RE.fullmatch(skill_name)
        if match is None:
            raise ValueError(f"Unknown Reactor skill: {skill_name}")
        self.set_frequency(int(match.group(1)), int(match.group(2)))
