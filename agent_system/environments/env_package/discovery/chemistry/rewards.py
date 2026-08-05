from __future__ import annotations

from typing import Any

from .utils import CHEMICAL_NAMES, chemical_counts


def chemical_combination(info: dict[str, Any] | None) -> tuple[int, int, int, int]:
    counts = chemical_counts((info or {}).get("chemical_dict"))
    return tuple(counts[name] for name in CHEMICAL_NAMES)


def target_progress_reward(env: Any, info: dict[str, Any]) -> float:
    target = getattr(env, "_hidden_chemical_target", None)
    if target is None:
        return 0.0
    before = chemical_combination(env._last_info)
    after = chemical_combination(info)
    return 2.0 * float(
        sum(abs(value - goal) for value, goal in zip(before, target))
        - sum(abs(value - goal) for value, goal in zip(after, target))
    )


def state_signature(info: dict[str, Any] | None) -> tuple[Any, ...]:
    info = info or {}
    ui = (info.get("raw_observation") or {}).get("ui", {})
    location = ui.get("agentLocation") or {}
    in_jar = bool(info.get("is_key_in_jar", False))
    position = (None, None, None) if in_jar else (
        location.get("x"), location.get("y"), location.get("faceDirection")
    )
    return (
        *position,
        bool(info.get("has_key", False)),
        bool(info.get("has_jar", False)),
        in_jar,
        chemical_combination(info),
        info.get("key_rust_status"),
        bool(info.get("won", False)),
    )
