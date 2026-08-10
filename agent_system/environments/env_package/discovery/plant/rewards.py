from __future__ import annotations

from typing import Any


TOTAL_RULE_CANDIDATES = 15

SUCCESS_POTENTIAL = 20.0
WRONG_FIELD_COMMIT_PENALTY = -4.0
WRONG_FIELD_PLANT_PENALTY = -2.0
UNPRODUCTIVE_WAIT_PENALTY = -1.0


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


def _correct_committed_fields(
    info: dict[str, Any], rule: tuple[str, int] | None
) -> set[str]:
    committed = {str(field) for field in info.get("plant_committed_fields", [])}
    return {field for field in committed if _field_matches_rule(info, field, rule)}


def _newly_committed_fields(
    before: dict[str, Any], after: dict[str, Any]
) -> set[str]:
    previous = {str(field) for field in before.get("plant_committed_fields", [])}
    current = {str(field) for field in after.get("plant_committed_fields", [])}
    return current - previous


def _new_seed_counts(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    previous = {
        str(field): int(count)
        for field, count in before.get("plant_planted_counts", {}).items()
    }
    current = {
        str(field): int(count)
        for field, count in after.get("plant_planted_counts", {}).items()
    }
    return {
        field: count - previous.get(field, 0)
        for field, count in current.items()
        if count > previous.get(field, 0)
    }


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
    planted = {
        str(field): int(count)
        for field, count in info.get("plant_planted_counts", {}).items()
    }
    field_one_matches = _field_matches_rule(info, "1", rule)
    configured = 1.0 if field_one_matches else 0.0
    correctly_committed = "1" in _correct_committed_fields(info, rule)
    committed_correctly = 1.0 if correctly_committed else 0.0
    planted_correctly = (
        0.5 * min(2, planted.get("1", 0)) if correctly_committed else 0.0
    )
    success = SUCCESS_POTENTIAL if bool(info.get("won", False)) else 0.0
    return float(
        information + configured + committed_correctly + planted_correctly + success
    )


def target_progress_reward(env: Any, info: dict[str, Any]) -> float:
    before = env._last_info or {}
    reward = progress_potential(info) - progress_potential(before)
    rule = _unique_rule(info)

    for field in _newly_committed_fields(before, info):
        if field != "1" or not _field_matches_rule(info, field, rule):
            reward += WRONG_FIELD_COMMIT_PENALTY

    correct_fields = _correct_committed_fields(info, rule)
    for field, new_seeds in _new_seed_counts(before, info).items():
        if field != "1" or field not in correct_fields:
            reward += WRONG_FIELD_PLANT_PENALTY * new_seeds

    waits_before = int(before.get("plant_growth_waits", 0) or 0)
    waits_after = int(info.get("plant_growth_waits", 0) or 0)
    if waits_after > waits_before and not bool(info.get("won", False)):
        planted = {
            str(field): int(count)
            for field, count in info.get("plant_planted_counts", {}).items()
        }
        ready_to_grow = "1" in correct_fields and planted.get("1", 0) >= 2
        if not ready_to_grow:
            reward += UNPRODUCTIVE_WAIT_PENALTY

    return float(reward)


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
