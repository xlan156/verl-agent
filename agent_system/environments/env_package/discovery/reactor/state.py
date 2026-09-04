from __future__ import annotations

from typing import Any
import json

from .prompts import REACTOR_TEMPLATE, REACTOR_TEMPLATE_NO_HIS


def prompt_state(info: dict[str, Any]) -> dict[str, Any]:
    compact_memory = {}
    for crystal, record in info.get("reactor_experiment_memory", {}).items():
        compact_memory[crystal] = {
            "readings": record.get("readings", {}),
            # Only manually known frequencies are shown. Target frequencies
            # must be calculated by the model from the observed relationship.
            "known_frequency": record.get("known_frequency"),
        }

    state = {
        "instruments": info.get("reactor_instruments", []),
        "crystal_and_instrument_memory": compact_memory,
        "reactors": info.get("reactor_states", {}),
        "calculation_hint": (
            "Use the calculation worksheet below. Compute the integer frequency yourself; "
            "the environment will validate it. Do not guess 1, 2, or 5000."
        ),
        "score": info.get("score_normalized", 0.0),
        "won": info.get("won", False),
    }

    # Once the two/three reference crystals have been measured, expose the
    # inferred equation and the target input, but never the target answer.
    # This keeps the numeric decision genuinely model-generated while making
    # the task feasible for a small model: it no longer has to discover the
    # relevant dimension and fit the polynomial from a long JSON table.
    model = info.get("reactor_inferred_model")
    if model:
        worksheet = {
            "relationship": "linear" if model.get("degree") == 1 else "quadratic",
            "dimension": model.get("dimension"),
            "targets": {},
        }
        if model.get("degree") == 1:
            worksheet["equation"] = (
                f"frequency = {round(model['slope'])} * {model['dimension']} "
                f"+ {round(model['offset'])}"
            )
        else:
            worksheet["equation"] = (
                f"frequency = {round(model['a'])} * {model['dimension']}^2 "
                f"+ {round(model['b'])} * {model['dimension']} + {round(model['c'])}"
            )
        for crystal, record in sorted(info.get("reactor_experiment_memory", {}).items()):
            if record.get("known_frequency") is None:
                worksheet["targets"][crystal] = {
                    "input": record.get("readings", {}).get(model["dimension"]),
                    "instruction": "substitute this input and round to the nearest integer",
                }
        state["calculation_worksheet"] = worksheet
    return state


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
        "task": _task_text(info.get("reactor_difficulty", "Normal")),
    }
    if init or history_length <= 0 or not records:
        return REACTOR_TEMPLATE_NO_HIS.format(**args)
    args["memory_actions"] = _history(records, history_length)
    return REACTOR_TEMPLATE.format(**args)


def _task_text(difficulty):
    difficulty = str(difficulty).strip().title()
    targets = {"Easy": "reactor 3", "Normal": "reactors 3 and 4", "Challenge": "reactors 4 and 5"}.get(difficulty, "the target reactors")
    relation = "a simple linear relationship" if difficulty in {"Easy", "Normal"} else "a quadratic relationship"
    return (
        f"Solve Reactor Lab ({difficulty}). Collect the required instruments, measure crystals, "
        f"infer {relation}, calculate the target frequency yourself, then install and activate {targets}."
    )


def build_anchor(raw_obs, info, records, config):
    return json.dumps(prompt_state(info), sort_keys=True)
