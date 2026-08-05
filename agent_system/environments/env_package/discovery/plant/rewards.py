from __future__ import annotations

from typing import Any


TOTAL_RULE_CANDIDATES = 15


def _canonical_selections(info: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (str(field), tuple(sorted((values or {}).items())))
        for field, values in sorted(info.get("plant_field_selections", {}).items())
    )


def _unique_rule(info: dict[str, Any]) -> tuple[str, int] | None:
    candidates = info.get("plant_rule_candidates", [])
    if len(candidates) != 1:
        return None
    return str(candidates[0]["nutrient"]), int(candidates[0]["level"])


def _field_matches_rule(info: dict[str, Any], field: str, rule: tuple[str, int] | None) -> bool:
    if rule is None:
        return False
    nutrient, level = rule
    selections = info.get("plant_field_selections", {}).get(str(field), {})
    return int(selections.get(nutrient, 0) or 0) == level


def progress_potential(info: dict[str, Any] | None) -> float:
    """Observable Plant progress potential; no simulator-only nutrient target is used."""
    info = info or {}
    candidates = info.get("plant_rule_candidates", [])
    candidate_count = len(candidates)
    information = (
        (TOTAL_RULE_CANDIDATES - candidate_count) / (TOTAL_RULE_CANDIDATES - 1)
        if 0 < candidate_count <= TOTAL_RULE_CANDIDATES else 0.0
    )
    rule = _unique_rule(info)
    committed = {str(field) for field in info.get("plant_committed_fields", [])}
    planted = {
        str(field): int(count)
        for field, count in info.get("plant_planted_counts", {}).items()
    }
    matching_fields = {
        str(field) for field in info.get("plant_field_selections", {})
        if _field_matches_rule(info, str(field), rule)
    }
    configured = 1.0 if matching_fields else 0.0
    committed_correctly = 1.0 if matching_fields & committed else 0.0
    planted_correctly = 0.5 * min(
        2,
        max((planted.get(field, 0) for field in matching_fields & committed), default=0),
    )
    return float(information + configured + committed_correctly + planted_correctly)


def target_progress_reward(env: Any, info: dict[str, Any]) -> float:
    return progress_potential(info) - progress_potential(env._last_info)


def state_signature(info: dict[str, Any] | None) -> tuple[Any, ...]:
    info = info or {}
    experiments = tuple(
        (item.get("plot"), tuple(sorted(item.get("nutrients", {}).items())), item.get("grew"))
        for item in info.get("plant_experiment_memory", [])
    )
    return (
        tuple(info.get("plant_tools", [])), experiments,
        info.get("plant_active_field"),
        _canonical_selections(info),
        tuple(info.get("plant_committed_fields", [])),
        tuple(sorted(info.get("plant_planted_counts", {}).items())),
        min(int(info.get("plant_growth_waits", 0) or 0), 2),
        bool(info.get("won", False)),
    )
