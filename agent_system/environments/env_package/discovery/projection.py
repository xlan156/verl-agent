# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple
import json
import re
from difflib import SequenceMatcher


def similar_string(str1: str, str2: str) -> bool:
    """Check if two strings are similar based on a simple heuristic."""
    str1_clean = re.sub(r"[^a-zA-Z0-9]", "", str1).lower()
    str2_clean = re.sub(r"[^a-zA-Z0-9]", "", str2).lower()
    return str1_clean == str2_clean


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)

SKILL_ALIASES: Dict[str, List[str]] = {
    "move_to_key": [
        "move to key",
        "go to key",
        "walk to key",
        "navigate to key",
    ],
    "move_to_jar": [
        "move to jar",
        "go to jar",
        "walk to jar",
        "navigate to jar",
    ],
    "move_to_dispensers_A": [
        "move to dispenser a",
        "go to dispenser a",
        "move to A dispenser",
        "move to dispenser A",
    ],
    "move_to_dispensers_B": [
        "move to dispenser b",
        "go to dispenser b",
        "move to B dispenser",
        "move to dispenser B",
    ],
    "move_to_dispensers_C": [
        "move to dispenser c",
        "go to dispenser c",
        "move to C dispenser",
        "move to dispenser C",
    ],
    "move_to_dispensers_D": [
        "move to dispenser d",
        "go to dispenser d",
        "move to D dispenser",
        "move to dispenser D",
    ],
    "pick_up_key": [
        "pick up key",
        "pickup key",
        "take key",
        "grab key",
    ],
    "put_key_in_jar": [
        "put key in jar",
        "place key in jar",
        "insert key into jar",
    ],
    "pick_up_jar": [
        "pick up jar",
        "pickup jar",
        "take jar",
        "grab jar",
    ],
    "use_dispenser_A_on_jar": [
        "use dispenser A on jar",
        "use A dispenser on jar",
        "dispense A into jar",
        "use the dispenser A to deliver substance A to the jar",
    ],
    "use_dispenser_B_on_jar": [
        "use dispenser B on jar",
        "use B dispenser on jar",
        "dispense B into jar",
        "use the dispenser B to deliver substance B to the jar",
    ],
    "use_dispenser_C_on_jar": [
        "use dispenser C on jar",
        "use C dispenser on jar",
        "dispense C into jar",
        "use the dispenser C to deliver substance C to the jar",
    ],
    "use_dispenser_D_on_jar": [
        "use dispenser D on jar",
        "use D dispenser on jar",
        "dispense D into jar",
        "use the dispenser D to deliver substance D to the jar",
    ],
    "wash_jar": [
        "wash jar",
        "clean jar",
        "rinse jar",
    ],
    "open_door": [
        "open door",
        "open the door",
    ],
}


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
            if isinstance(value, str) and value in SKILL_ALIASES:
                return value
    return None


def _find_best_skill(action_text: str, threshold: float = 0.6) -> Optional[str]:
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

    for skill, aliases in SKILL_ALIASES.items():
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
        skill = _find_best_skill(action)
        if skill is None:
            processed.append(action)
            valids.append(0)
        else:
            if location == key_location and (skill == "move_to_key" or skill == "move_to_jar"):
                processed.append(skill)
                valids.append(0)
            elif location != key_location and skill in ["pick_up_key", "put_key_in_jar", "pick_up_jar"]:
                processed.append(skill)
                valids.append(0)
            elif skill.startswith("use_dispenser") and location[0] < 18:
                processed.append(skill)
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
        "move_to_key\nput_key_in_jar\nuse_dispenser_A"
    ]
    
    projected, valids = discoveryworld_projection(sample_actions, infos=[info])
    for raw, mapped, valid in zip(sample_actions, projected, valids):
        print(f"input={raw!r} -> mapped={mapped!r} valid={valid}")
