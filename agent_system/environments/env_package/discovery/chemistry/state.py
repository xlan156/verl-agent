from __future__ import annotations

from typing import Any

from .utils import (
    build_chemical_belief_state,
    build_discoveryworld_anchor_obs,
    build_discoveryworld_text_obs,
    observable_experiment_evidence,
)


def _legacy_records(records):
    legacy = []
    for record in records:
        item = dict(record.get("task_memory") or {})
        item["action"] = record.get("action")
        item["text_obs"] = record.get("text_obs")
        legacy.append(item)
    return legacy


def memory_record(previous, current, action):
    evidence = observable_experiment_evidence(previous, current)
    return {
        "pre_chemical_dict": previous.get("chemical_dict"),
        "post_chemical_dict": current.get("chemical_dict"),
        "pre_rust_level": previous.get("key_rust_level"),
        "pre_rust_status": previous.get("key_rust_status"),
        "pre_reaction_signal": previous.get("current_reaction_signal"),
        "rust_level": current.get("key_rust_level"),
        "rust_status": current.get("key_rust_status"),
        "reaction_signal": current.get("current_reaction_signal"),
        "experiment_evidence_kind": evidence.get("kind") if evidence else None,
        "experiment_evidence_label": evidence.get("label") if evidence else None,
        "action_result": current.get("last_action_result"),
    }


def build_prompt(raw_obs, info, records, config, init=False):
    return build_discoveryworld_text_obs(
        [raw_obs], [info], [_legacy_records(records)], config, init=init
    )[0]


def build_anchor(raw_obs, info, records, config):
    return build_discoveryworld_anchor_obs(
        [raw_obs], [info], [_legacy_records(records)], config
    )[0]


def log_state(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "chemical_dict": info.get("chemical_dict", {}),
        "key_rust_level": info.get("key_rust_level"),
        "key_rust_status": info.get("key_rust_status"),
        "reaction_signal": info.get("current_reaction_signal"),
        "has_key": info.get("has_key", False),
        "has_jar": info.get("has_jar", False),
        "is_key_in_jar": info.get("is_key_in_jar", False),
    }


def belief_log_state(records, info):
    state = log_state(info)
    state["chemical_belief"] = build_chemical_belief_state(_legacy_records(records), info)
    return state
