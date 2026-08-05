from __future__ import annotations

from typing import Any


def target_progress_reward(env: Any, info: dict[str, Any]) -> float:
    return 0.0


def state_signature(info: dict[str, Any] | None) -> tuple[Any, ...]:
    info = info or {}
    memory = info.get("reactor_experiment_memory", {})
    measured = tuple(
        (index, tuple(sorted(record.get("readings", {}).items())), record.get("known_frequency"))
        for index, record in sorted(memory.items())
    )
    reactors = tuple(
        (index, state.get("frequency"), state.get("has_crystal"), state.get("activated"))
        for index, state in sorted(info.get("reactor_states", {}).items())
    )
    return (
        tuple(info.get("reactor_instruments", [])), measured, reactors,
        bool(info.get("won", False)),
    )
