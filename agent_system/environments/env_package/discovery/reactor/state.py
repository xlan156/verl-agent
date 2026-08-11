from __future__ import annotations

from typing import Any
import json

from .prompts import REACTOR_TEMPLATE, REACTOR_TEMPLATE_NO_HIS


def prompt_state(info: dict[str, Any]) -> dict[str, Any]:
    model = info.get("reactor_inferred_model")
    compact_memory = {}
    
    for crystal, record in info.get("reactor_experiment_memory", {}).items():
        known_frequency = record.get("known_frequency")
        crystal_frequency = known_frequency
        
        if crystal_frequency is None and model:
            reading = float(record.get("readings", {}).get(model["dimension"]))
            crystal_frequency = int(round(model["slope"] * reading + model["offset"], 2))
            
        compact_memory[crystal] = {
            "readings": record.get("readings", {}),
            "known_frequency": known_frequency,
            "crystal_frequency": crystal_frequency,
        }
        
    return {
        "instruments": info.get("reactor_instruments", []),
        "crystal_and_instrument_memory": compact_memory,
        "inferred_model": model,
        "reactors": info.get("reactor_states", {}),
        "score": info.get("score_normalized", 0.0),
        "won": info.get("won", False),
    }


def memory_record(previous, current, action):
    return {
        "action_result": current.get("last_action_result"),
        "crystal_and_instrument_memory": current.get("reactor_experiment_memory", {}),
        "inferred_model": current.get("reactor_inferred_model"),
        "reactors": current.get("reactor_states", {}),
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
        return REACTOR_TEMPLATE_NO_HIS.format(**args)
    args["memory_actions"] = _history(records, history_length)
    return REACTOR_TEMPLATE.format(**args)


def build_anchor(raw_obs, info, records, config):
    return json.dumps(prompt_state(info), sort_keys=True)
