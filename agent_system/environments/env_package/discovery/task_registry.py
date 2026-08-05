from __future__ import annotations

from importlib import import_module
from typing import Any


SCENARIO_TO_TASK = {
    "combinatorial chemistry": "chemistry",
    "plant nutrients": "plant",
    "reactor lab": "reactor",
}


def task_family_for_scenario(scenario_name: str | None) -> str:
    normalized = str(scenario_name or "Combinatorial Chemistry").strip().lower()
    try:
        return SCENARIO_TO_TASK[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DiscoveryWorld scenario {scenario_name!r}; expected one of "
            f"{sorted(SCENARIO_TO_TASK)}"
        ) from exc


def get_task_adapter(scenario_name: str | None) -> Any:
    family = task_family_for_scenario(scenario_name)
    return get_task_adapter_by_family(family)


def get_task_adapter_by_family(family: str) -> Any:
    if family not in set(SCENARIO_TO_TASK.values()):
        raise ValueError(f"Unsupported DiscoveryWorld task family: {family!r}")
    module = import_module(
        f"agent_system.environments.env_package.discovery.{family}.envs"
    )
    return module.TaskAdapter()
