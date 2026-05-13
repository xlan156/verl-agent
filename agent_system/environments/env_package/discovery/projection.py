from typing import Any, Dict, List, Optional, Tuple
import json
import re
from difflib import SequenceMatcher
from agent_system.environments.env_package.discovery.helpers import *


def similar_string(str1: str, str2: str) -> bool:
    """Check if two strings are similar based on a simple heuristic."""
    str1_clean = re.sub(r"[^a-zA-Z0-9]", "", str1).lower()
    str2_clean = re.sub(r"[^a-zA-Z0-9]", "", str2).lower()
    return str1_clean == str2_clean


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)

def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = _THINK_RE.sub(" ", text)
    m = _ACTION_RE.search(text)
    if m:
        text = m.group(1).strip()
    if "\n" in text:
        text = text.split("\n")[-1]
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower()
    return text


def _string_similarity(a: str, b: str) -> float:
    if similar_string(a, b):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _extract_skill_from_json(text: str) -> Optional[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return None

    if isinstance(payload, dict):
        for key in ("skill", "action"):
            value = payload.get(key)
            if isinstance(value, str) and value in SKILL_NAMES:
                return value
    return None


def _find_matching_skill(action_text: str, info: Dict, threshold: float = 0.6) -> Optional[str]:
    if not action_text:
        return None

    direct = _extract_skill_from_json(action_text)
    if direct:
        return direct

    normalized = _normalize_text(action_text)
    if not normalized:
        return None

    best_skill = None
    best_score = 0.0

    for skill, aliases in SKILL_NAMES.items():
        for alias in aliases + [skill]:
            score = _string_similarity(normalized, _normalize_text(alias))
            if score > best_score:
                best_score = score
                best_skill = skill

    if best_score >= threshold:
        return best_skill
    return None


def discoveryworld_projection(
    actions: List[str],
    infos: List[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    processed: List[str] = []
    valids: List[int] = []

    key_location = (17, 12)
    for i, action in enumerate(actions):
        info = infos[i] if infos else {}
        ui = (info.get("raw_observation") or {}).get("ui", {})
        location = (ui.get("agentLocation").get("x"), ui.get("agentLocation").get("y"))
        #skill = _find_matching_skill(action_text=action, info=info, threshold=0.6)
        skill = action if action in SKILL_NAMES else None
        if skill is None:
            processed.append("Invalid action")
            valids.append(0)
        else:
            processed.append(skill)
            valids.append(1)

    return processed, valids


if __name__ == "__main__":
    from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
    
    env = DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Easy",
        seed=0,
    )
    
    obs, info = env.reset()
    sample_actions = [
        "move_to_key\nput_key_in_jar\nuse_dispenser_A_on_jar"
    ]
    
    projected, valids = discoveryworld_projection(sample_actions, infos=[info])
    for raw, mapped, valid in zip(sample_actions, projected, valids):
        print(f"input={raw!r} -> mapped={mapped!r} valid={valid}")
