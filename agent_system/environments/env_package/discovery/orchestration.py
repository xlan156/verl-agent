"""Task-neutral DiscoveryWorld prompt, memory, anchor, and log routing."""

from __future__ import annotations

from typing import Any

from .task_registry import get_task_adapter_by_family


def adapter_for_info(info: dict[str, Any]) -> Any:
    return get_task_adapter_by_family(str(info.get("task_family") or "chemistry"))


def build_discoveryworld_text_obs(text_obs, infos, memory, config, init=False):
    return [
        adapter_for_info(info).build_prompt(
            text_obs[index], info, memory[index] if index < len(memory) else [], config, init
        )
        for index, info in enumerate(infos)
    ]


def build_discoveryworld_anchor_obs(text_obs, infos, memory, config):
    discovery_cfg = getattr(config.env, "discoveryworld", None)
    anchor_mode = str(getattr(discovery_cfg, "anchor_mode", "belief_summary")).lower()
    anchors = []
    for index, info in enumerate(infos):
        raw_obs = text_obs[index]
        if anchor_mode in {"raw", "raw_obs", "text_obs"}:
            anchors.append(raw_obs)
        else:
            records = memory[index] if index < len(memory) else []
            anchors.append(adapter_for_info(info).build_anchor(raw_obs, info, records, config))
    return anchors


def build_memory_record(previous_info, current_info, action, previous_obs):
    adapter = adapter_for_info(current_info)
    return {
        "task_family": adapter.family,
        "text_obs": previous_obs,
        "action": action,
        "task_memory": adapter.memory_record(previous_info, current_info, action),
    }


def build_llm_step_state(info):
    return adapter_for_info(info).log_state(info)
