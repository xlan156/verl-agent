from __future__ import annotations

import json
from typing import Any

from .prompts import PLANT_TEMPLATE, PLANT_TEMPLATE_NO_HIS


def _visible_candidates(candidates: list[dict[str, Any]]) -> Any:
    if len(candidates) <= 20:
        return candidates
    return f"{len(candidates)} candidates remain; continue observable measurements"


def prompt_state(info: dict[str, Any]) -> dict[str, Any]:
    candidates = info.get("plant_rule_candidates", [])
    return {
        "difficulty": info.get("plant_difficulty", "Normal"),
        "tools": info.get("plant_tools", []),
        "observable_experiment_memory": info.get("plant_experiment_memory", []),
        "rule_candidate_count": len(candidates),
        "rule_candidates": _visible_candidates(candidates),
        "selected_nutrient": info.get("plant_selected_nutrient"),
        "active_field": info.get("plant_active_field"),
        "field_selections": info.get("plant_field_selections", {}),
        "committed_fields": info.get("plant_committed_fields", []),
        "planted_counts": info.get("plant_planted_counts", {}),
        "growth_waits": info.get("plant_growth_waits", 0),
        "score": info.get("score_normalized", 0.0),
        "won": info.get("won", False),
    }


def memory_record(previous, current, action):
    return {
        "action_result": current.get("last_action_result"),
        "experiment_memory": current.get("plant_experiment_memory", []),
        "rule_candidate_count": len(current.get("plant_rule_candidates", [])),
        "rule_candidates": _visible_candidates(
            current.get("plant_rule_candidates", [])
        ),
        "selected_nutrient": current.get("plant_selected_nutrient"),
        "active_field": current.get("plant_active_field"),
        "committed_fields": current.get("plant_committed_fields", []),
        "planted_counts": current.get("plant_planted_counts", {}),
        "growth_waits": current.get("plant_growth_waits", 0),
    }


def _history(records, limit):
    lines = []
    start = max(1, len(records) - limit + 1)
    for index, record in enumerate(records[-limit:], start=start):
        result = (record.get("task_memory") or {}).get("action_result") or {}
        lines.append(f"{index}. {record.get('action')} -> {result.get('message', '')}")
    return "\n".join(lines)


def _instructions(difficulty: str) -> tuple[str, str]:
    difficulty = str(difficulty or "Normal").strip().lower()
    if difficulty == "easy":
        return (
            "Solve Plant Nutrients (Easy) from observable pilot plots. Identify "
            "which nutrient is present on growing plots, then submit that nutrient "
            "to the controller. Nutrient levels are 0=absent and 3=present.",
            "Measure until exactly one rule candidate remains. The five select_* "
            "skills are executable answer choices, so select only the nutrient "
            "supported by the experiments.",
        )
    if difficulty == "challenge":
        return (
            "Solve Plant Nutrients (Challenge) from observable pilot plots. Infer "
            "the unique xor, and, or, or not rule over nutrient levels; configure "
            "field 1 to satisfy it, commit, plant two seeds, and wait for growth. "
            "Levels use 1=low, 2=medium, and 3=high.",
            "Measure until exactly one compound rule remains. In the controller, "
            "compare every rule condition with field_selections. Set all required "
            "levels before commit; for xor satisfy exactly one condition, and for "
            "not choose a level different from the forbidden one.",
        )
    return (
        "Solve Plant Nutrients (Normal) from observable pilot plots. Infer the "
        "single required nutrient and level, configure field 1, commit, plant two "
        "seeds, and wait for growth. Levels use 1=low, 2=medium, and 3=high.",
        "Measure until exactly one rule remains. In the controller choose the "
        "matching setter when the field differs, and commit only when it matches. "
        "After commit, plant two seeds before waiting.",
    )


def build_prompt(raw_obs, info, records, config, init=False):
    history_length = int(getattr(config.env, "history_length", 0) or 0)
    task_instruction, decision_rule = _instructions(
        info.get("plant_difficulty", "Normal")
    )
    args = {
        "task_instruction": task_instruction,
        "decision_rule": decision_rule,
        "step_info": f"Step: {len(records)} / {config.env.max_steps}",
        "state": json.dumps(prompt_state(info), indent=2, sort_keys=True),
        "valid_skills": "\n".join(info.get("valid_skills", [])),
    }
    if init or history_length <= 0 or not records:
        return PLANT_TEMPLATE_NO_HIS.format(**args)
    args["memory_actions"] = _history(records, history_length)
    return PLANT_TEMPLATE.format(**args)


def build_anchor(raw_obs, info, records, config):
    return json.dumps(prompt_state(info), sort_keys=True)
