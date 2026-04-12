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
from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

AVAILABLE_ACTIONS = DiscoveryWorldAPI.listKnownActionsStatic()

# 用于解析 LLM 输出中的代码块
_CODE_BLOCK_JSON_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_GENERIC_RE = re.compile(r"```(.*?)```", re.DOTALL)

_DEFAULT_SAFE_ACTION = json.dumps({"action": "MOVE_DIRECTION", "arg1": "west"}, separators=(",", ":"))

_DIRECTIONS = {"north", "south", "east", "west"}

_OBJECT_ACTION_PATTERNS = [
    (re.compile(r"\bpick(?:\s+up)?\b|\btake\b", re.IGNORECASE), "PICKUP"),
    (re.compile(r"\bdrop\b|\brelease\b", re.IGNORECASE), "DROP"),
    (re.compile(r"\bopen\b", re.IGNORECASE), "OPEN"),
    (re.compile(r"\bclose\b|\bshut\b", re.IGNORECASE), "CLOSE"),
    (re.compile(r"\bactivate\b|\bturn\s+on\b", re.IGNORECASE), "ACTIVATE"),
    (re.compile(r"\bdeactivate\b|\bturn\s+off\b", re.IGNORECASE), "DEACTIVATE"),
    (re.compile(r"\beat\b|\bconsume\b", re.IGNORECASE), "EAT"),
    (re.compile(r"\bread\b", re.IGNORECASE), "READ"),
    (re.compile(r"\buse\b", re.IGNORECASE), "USE"),
    (re.compile(r"\btalk\b|\bspeak\b", re.IGNORECASE), "TALK"),
]

_SINGLE_OBJECT_ACTIONS = {
    "PICKUP",
    "DROP",
    "OPEN",
    "CLOSE",
    "ACTIVATE",
    "DEACTIVATE",
    "EAT",
    "READ",
    "TALK",
    "TELEPORT_TO_OBJECT",
}

_DOUBLE_OBJECT_ACTIONS = {"USE", "PUT"}


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract a JSON object from LLM output.

    Priority:
      1) Last ```json ... ``` block.
      2) Else last ``` ... ``` block.
      3) Else whole text.
    Returns parsed dict or None.
    """
    candidate: Optional[str] = None

    # 1) ```json ... ```
    matches = _CODE_BLOCK_JSON_RE.findall(text)
    if matches:
        candidate = matches[-1].strip()
    else:
        # 2) 任意 ``` ... ```
        matches = _CODE_BLOCK_GENERIC_RE.findall(text)
        if matches:
            candidate = matches[-1].strip()

    # 3) 没有代码块，就用原文
    if candidate is None:
        candidate = text.strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_last_braced_object(text: str) -> Optional[str]:
    """Return the last balanced {...} substring if present."""
    if "{" not in text or "}" not in text:
        return None

    depth = 0
    end_idx = None
    start_idx = None

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    end_idx = i

    if start_idx is None or end_idx is None:
        return None

    return text[start_idx : end_idx + 1]


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)


def _extract_direction(text: str) -> Optional[str]:
    text_l = text.lower()
    for direction in _DIRECTIONS:
        if re.search(rf"(?<!\w){direction}(?!\w)", text_l):
            return direction
    return None


def _infer_move_or_rotate(text: str) -> Optional[Dict[str, Any]]:
    direction = _extract_direction(text)
    if not direction:
        return None

    text_l = text.lower()
    if re.search(r"\brotate\b|\bturn\b|\bface\b", text_l):
        return {"action": "ROTATE_DIRECTION", "arg1": direction}
    if re.search(r"\bmove\b|\bgo\b|\bwalk\b|\bstep\b", text_l):
        return {"action": "MOVE_DIRECTION", "arg1": direction}
    return None


def _find_uuid_from_text(text: str, object_seen: Dict[str, str]) -> Optional[str]:
    if not object_seen:
        return None

    text_l = text.lower()
    text_norm = re.sub(r"[^a-z0-9]+", " ", text_l).strip()
    text_tokens = set(text_norm.split())

    for name, uuid in object_seen.items():
        if not name:
            continue
        name_l = name.lower()
        if name_l in text_l:
            return str(uuid)
        name_norm = re.sub(r"[^a-z0-9]+", " ", name_l).strip()
        if name_norm and (name_norm in text_norm or text_norm in name_norm):
            return str(uuid)
        name_tokens = set(name_norm.split()) if name_norm else set()
        if name_tokens and name_tokens.issubset(text_tokens):
            return str(uuid)
        if name_tokens and any(tok in text_tokens and len(tok) >= 3 for tok in name_tokens):
            return str(uuid)

    base_map: Dict[str, str] = {}
    for name, uuid in object_seen.items():
        base = re.sub(r"\d+$", "", name).strip().lower()
        if base and base not in base_map:
            base_map[base] = str(uuid)

    for base, uuid in base_map.items():
        if base and base in text_l:
            return uuid
        base_norm = re.sub(r"[^a-z0-9]+", " ", base).strip()
        if base_norm and (base_norm in text_norm or text_norm in base_norm):
            return uuid
        base_tokens = set(base_norm.split()) if base_norm else set()
        if base_tokens and base_tokens.issubset(text_tokens):
            return uuid
        if base_tokens and any(tok in text_tokens and len(tok) >= 3 for tok in base_tokens):
            return uuid

    return None


def _find_two_uuids_from_text(text: str, object_seen: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    if not object_seen:
        return None, None

    text_l = text.lower()
    if " on " in text_l:
        left, right = text_l.split(" on ", 1)
    elif " with " in text_l:
        left, right = text_l.split(" with ", 1)
    elif " into " in text_l:
        left, right = text_l.split(" into ", 1)
    elif " in " in text_l:
        left, right = text_l.split(" in ", 1)
    else:
        return None, None

    left_uuid = _find_uuid_from_text(left, object_seen)
    right_uuid = _find_uuid_from_text(right, object_seen)
    return left_uuid, right_uuid


def _infer_object_action(text: str) -> str:
    text_norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    for pattern, action in _OBJECT_ACTION_PATTERNS:
        if pattern.search(text_norm):
            return action
    return "PICKUP"


def _prefer_jar_as_arg2(
    action: Dict[str, Any],
    text: str,
    object_seen: Dict[str, str],
) -> Dict[str, Any]:
    if action.get("action") not in _DOUBLE_OBJECT_ACTIONS:
        return action

    text_l = text.lower()
    if "jar" not in text_l:
        return action

    jar_uuid = _find_uuid_from_text("jar", object_seen)
    if not jar_uuid:
        return action

    action["arg2"] = jar_uuid

    if action.get("arg1") == jar_uuid or not action.get("arg1"):
        other_text = re.sub(r"jar", " ", text_l)
        other_uuid = _find_uuid_from_text(other_text, object_seen)
        if other_uuid:
            action["arg1"] = other_uuid

    return action
    

def discoveryworld_projection(
    actions: List[str],
    infos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[int]]:

    processed: List[str] = []
    valids: List[int] = [0] * len(actions)
    contain_think_block: List[int] = [0] * len(actions)
    are_json_format: List[int] = [0] * len(actions)

    re_action_block = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)

    object_seen_list: List[Dict[str, str]] = []
    if infos:
        for info in infos:
            seen = info.get("object_seen") or {}
            if isinstance(seen, dict):
                object_seen_list.append({str(k): str(v) for k, v in seen.items()})
            else:
                object_seen_list.append({})
    else:
        object_seen_list = [{} for _ in actions]

    for i, action_str in enumerate(actions):
        action_text = action_str or ""

        # Detect whether the original model output contains a <think>...</think> block.
        # If no block, contain_think_block[i] = 0; else 1.
        contain_think_block[i] = 1 if re.search(
            r"<think>.*?</think>",
            action_text,
            flags=re.IGNORECASE | re.DOTALL,
        ) else 0

        original = _strip_think_blocks(action_text)

        info_i: Dict[str, Any] = infos[i] if (infos is not None and i < len(infos)) else {}
        # Expose debugging flags downstream without changing the return signature.
        if infos is not None and i < len(infos):
            info_i["contain_think_block"] = contain_think_block[i]
            info_i["are_json_format"] = 0

        object_seen = object_seen_list[i] if i < len(object_seen_list) else {}

        valid_uuids = set()
        raw_observation = info_i.get("raw_observation") or {}
        ui = raw_observation.get("ui") or {}
        inventory = ui.get("inventoryObjects") or []
        accessible = ui.get("accessibleEnvironmentObjects") or []
        for obj in list(inventory) + list(accessible):
            if isinstance(obj, dict) and obj.get("name") not in ["wall", "floor", "grass", "path"]:
                valid_uuids.add(str(obj.get("uuid")))
        
        
        m = re_action_block.search(action_text)
        if not m:

            braced = _extract_last_braced_object(action_text)
            if braced is not None:
                candidate_text = braced
            else:
                candidate_text = action_text.strip()[-512:]
        else:
            candidate_text = m.group(1).strip()

        parsed: Optional[Dict[str, Any]] = None
        try:
            tmp = json.loads(candidate_text)
            if isinstance(tmp, dict):
                parsed = tmp
        except Exception:
            parsed = _extract_json_from_text(candidate_text)

        candidate_action: Optional[Dict[str, Any]] = None
        if parsed is not None:
            are_json_format[i] = 1
            if infos is not None and i < len(infos):
                info_i["are_json_format"] = 1
            action_name = parsed.get("action")
            if isinstance(action_name, str) and action_name in AVAILABLE_ACTIONS:
                candidate_action = dict(parsed)
                if "arg1" in candidate_action:
                    arg1_old = candidate_action["arg1"]
                    candidate_action["arg1"] = _find_uuid_from_text(str(arg1_old), object_seen)
                if "arg2" in candidate_action:
                    arg2_old = candidate_action["arg2"]
                    candidate_action["arg2"] = _find_uuid_from_text(str(arg2_old), object_seen)
                if action_name in {"MOVE_DIRECTION", "ROTATE_DIRECTION"}:
                    direction = str(candidate_action.get("arg1", "")).lower()
                    if direction not in _DIRECTIONS:
                        direction = _extract_direction(original or candidate_text)
                        if direction:
                            candidate_action["arg1"] = direction
                elif action_name in _SINGLE_OBJECT_ACTIONS:
                    arg1 = candidate_action.get("arg1")
                    if not arg1:
                        uuid = _find_uuid_from_text(original or candidate_text, object_seen)
                        if uuid:
                            candidate_action["arg1"] = uuid
                elif action_name in _DOUBLE_OBJECT_ACTIONS:
                    arg1 = candidate_action.get("arg1")
                    arg2 = candidate_action.get("arg2")
                    if not arg1 or not arg2:
                        left_uuid, right_uuid = _find_two_uuids_from_text(
                            original or candidate_text,
                            object_seen,
                        )
                        if not arg1 and left_uuid:
                            candidate_action["arg1"] = left_uuid
                        if not arg2 and right_uuid:
                            candidate_action["arg2"] = right_uuid
                    candidate_action = _prefer_jar_as_arg2(
                        candidate_action,
                        original or candidate_text,
                        object_seen,
                    )

        if candidate_action is None:
            candidate_action = _infer_move_or_rotate(original)

        if candidate_action is None:
            action_name = _infer_object_action(original)
            if action_name in _SINGLE_OBJECT_ACTIONS:
                uuid = _find_uuid_from_text(original, object_seen)
                candidate_action = {"action": action_name, "arg1": uuid}

        if candidate_action is None:
            action_name = _infer_object_action(original)
            if action_name in _DOUBLE_OBJECT_ACTIONS:
                uuid2 = _find_uuid_from_text("jar", object_seen)
                uuid1 = _find_uuid_from_text(re.sub(r"jar", " ", original.lower()), object_seen)
                if uuid1 and uuid2:
                    candidate_action = {
                        "action": action_name,
                        "arg1": uuid1,
                        "arg2": uuid2,
                    }

        if candidate_action is None:
            valids[i] = 0
            processed.append(_DEFAULT_SAFE_ACTION)
        else:
            if candidate_action.get("action") in {"MOVE_DIRECTION", "ROTATE_DIRECTION"} and candidate_action.get("arg1") not in _DIRECTIONS:
                valids[i] = 0
                processed.append(_DEFAULT_SAFE_ACTION)
            elif candidate_action.get("action") in _SINGLE_OBJECT_ACTIONS and candidate_action.get("arg1") not in valid_uuids:
                valids[i] = 0
                processed.append(json.dumps(candidate_action))
            elif candidate_action.get("action") in _DOUBLE_OBJECT_ACTIONS and (candidate_action.get("arg1") not in valid_uuids or candidate_action.get("arg2") not in valid_uuids):
                valids[i] = 0
                processed.append(json.dumps(candidate_action))
            else:
                valids[i] = 1
                processed.append(json.dumps(candidate_action, separators=(",", ":")))

    return processed, valids
