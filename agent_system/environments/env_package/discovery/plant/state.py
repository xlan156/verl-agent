from __future__ import annotations

from typing import Any
import json

from .prompts import PLANT_TEMPLATE, PLANT_TEMPLATE_NO_HIS


def prompt_state(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "tools": info.get("plant_tools", []),
        "observable_experiment_memory": info.get("plant_experiment_memory", []),
        "rule_candidates": info.get("plant_rule_candidates", []),
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
        "rule_candidates": current.get("plant_rule_candidates", []),
        "active_field": current.get("plant_active_field"),
        "committed_fields": current.get("plant_committed_fields", []),
        "planted_counts": current.get("plant_planted_counts", {}),
        "growth_waits": current.get("plant_growth_waits", 0),
    }


def _history(records, limit):
    lines = []
    for index, record in enumerate(records[-limit:], start=max(1, len(records) - limit + 1)):
        result = (record.get("task_memory") or {}).get("action_result") or {}
        lines.append(f"{index}. {record.get('action')} -> {result.get('message', '')}")
    return "\n".join(lines)


def build_prompt(raw_obs, info, records, config, init=False):
    history_length = int(getattr(config.env, "history_length", 0) or 0)
    args = {
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
