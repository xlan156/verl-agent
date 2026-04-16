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
from agent_system.environments.env_package.discovery.actions import all_plausible_action_mapper

AVAILABLE_ACTIONS = DiscoveryWorldAPI.listKnownActionsStatic()

# Regex helpers
JSON_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)
GENERIC_RE = re.compile(r"```(.*?)```", re.DOTALL)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_OBJECT_ACTION_PATTERNS = [
    (re.compile(r"\bwash\b|\brinse\b|\bclean\b", re.IGNORECASE), "USE"),
    (re.compile(r"\bpick(?:\s+up)?\b|\btake\b", re.IGNORECASE), "PICKUP"),
    (re.compile(r"\bopen\b", re.IGNORECASE), "OPEN"),
    (re.compile(r"\buse\b", re.IGNORECASE), "USE"),
    (re.compile(r"\bput\b|\bplace\b", re.IGNORECASE), "PUT"),
    (re.compile(r"\bmove\b|\bgo\b|\bwalk\b|\bstep\b", re.IGNORECASE), "MOVE_DIRECTION"),
    (re.compile(r"\brotate\b|\bturn\b|\bface\b", re.IGNORECASE), "ROTATE_DIRECTION"),
]

_DEFAULT_SAFE_ACTION = json.dumps({"action": "MOVE_DIRECTION", "arg1": "west"}, separators=(",", ":"))


def _normalize_action_choice(text: str) -> str:
    """Normalize an action choice emitted by the model.

    Handles small formatting variations like bullets/numbering and quotes.
    """
    s = (text or "").strip()
    # Strip leading bullets / numbering.
    s = re.sub(r"^\s*(?:[-*]|\d+[\.)])\s+", "", s)
    s = s.strip().strip('"').strip("'").strip()
    return s

_MOVE_ROTATE_ACTIONS = {"MOVE_DIRECTION", "ROTATE_DIRECTION"}
_DIRECTIONS = {"north", "south", "east", "west"}
_SINGLE_OBJECT_ACTIONS = {"PICKUP", "OPEN"}
_DOUBLE_OBJECT_ACTIONS = {"USE", "PUT"}


def _contains_cjk(text: str) -> bool:
    return bool(text and _CJK_RE.search(text))


def _count_top_level_json_objects(text: str) -> int:
    """Count complete top-level {...} JSON objects in text.

    This is a lightweight heuristic used to detect when the model emitted multiple
    JSON action objects inside a single <action>...</action> block.
    """
    if not text:
        return 0

    depth = 0
    in_str = False
    escape = False
    count = 0

    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    count += 1

    return count


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:

    candidate: Optional[str] = None

    matches = JSON_RE.findall(text)
    if matches:
        candidate = matches[-1].strip()
    else:
        matches = GENERIC_RE.findall(text)
        if matches:
            candidate = matches[-1].strip()

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
    text_l = (text or "").lower()
    direction = _extract_direction(text_l)
    if not direction:
        return None

    # Prefer explicit action-name style outputs like:
    #   MOVE_DIRECTION,west
    #   ROTATE_DIRECTION east
    #   move_direction , west
    if re.search(r"\brotate_direction\b|\brotate\s+direction\b", text_l):
        return {"action": "ROTATE_DIRECTION", "arg1": direction}
    if re.search(r"\bmove_direction\b|\bmove\s+direction\b", text_l):
        return {"action": "MOVE_DIRECTION", "arg1": direction}

    # Fallback: natural language phrasing.
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


def _build_object_seen_list(
    infos: Optional[List[Dict[str, Any]]],
    actions: List[str],
) -> List[Dict[str, str]]:
    if not infos:
        return [{} for _ in actions]

    object_seen_list: List[Dict[str, str]] = []
    for info in infos:
        seen = info.get("object_seen") or {}
        if isinstance(seen, dict):
            object_seen_list.append({str(k): str(v) for k, v in seen.items()})
        else:
            object_seen_list.append({})
    return object_seen_list


def _extract_detection_flags(
    action_text: str,
    re_action_block: re.Pattern,
) -> Tuple[int, int, int, int]:
    think_blocks = re.findall(
        r"<think>(.*?)</think>",
        action_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    think_text = "\n".join([t.strip() for t in think_blocks if t is not None])
    think_has_chinese = 1 if _contains_cjk(think_text) else 0

    contain_think_block = 1 if think_blocks else 0
    contain_action_block = 1 if re_action_block.search(action_text) else 0

    action_start_tags = re.findall(r"<action\b", action_text, flags=re.IGNORECASE)
    action_multiple_actions = 1 if len(action_start_tags) > 1 else 0

    action_blocks = re_action_block.findall(action_text)
    if action_multiple_actions == 0 and len(action_blocks) == 1:
        action_payload = (action_blocks[0] or "").strip()
        try:
            parsed_payload = json.loads(action_payload)
            if isinstance(parsed_payload, list) and len(parsed_payload) > 1:
                action_multiple_actions = 1
        except Exception:
            if _count_top_level_json_objects(action_payload) > 1:
                action_multiple_actions = 1

    return contain_think_block, contain_action_block, think_has_chinese, action_multiple_actions


def _update_info_flags(
    info: Dict[str, Any],
    contain_think_block: int,
    contain_action_block: int,
    think_has_chinese: int,
    action_multiple_actions: int,
    are_json_format: int = 0,
) -> None:
    info["contain_think_block"] = contain_think_block
    info["contain_action_block"] = contain_action_block
    info["are_json_format"] = are_json_format
    info["think_has_chinese"] = think_has_chinese
    info["action_multiple_actions"] = action_multiple_actions


def _collect_valid_uuids(info: Dict[str, Any]) -> List[str]:
    raw_observation = info.get("raw_observation") or {}
    ui = raw_observation.get("ui") or {}
    inventory = ui.get("inventoryObjects") or []
    accessible = ui.get("accessibleEnvironmentObjects") or []

    valid_uuids = []
    for obj in list(inventory) + list(accessible):
        if isinstance(obj, dict) and obj.get("name") not in ["wall", "floor", "grass", "path"]:
            valid_uuids.append(str(obj.get("uuid")))
    return valid_uuids


def _select_candidate_text(action_text: str, re_action_block: re.Pattern) -> str:
    m = re_action_block.search(action_text)
    if not m:
        braced = _extract_last_braced_object(action_text)
        if braced is not None:
            return braced
        return action_text.strip()[-512:]
    return m.group(1).strip()


def _resolve_canonical_choice(candidate_text: str) -> Optional[str]:
    normalized_choice = _normalize_action_choice(candidate_text)
    if not normalized_choice:
        return None

    choice_l = normalized_choice.lower()
    for k in all_plausible_action_mapper.keys():
        if choice_l == k.lower():
            return k
    return None


def _parse_candidate_json(candidate_text: str) -> Optional[Dict[str, Any]]:
    try:
        tmp = json.loads(candidate_text)
        if isinstance(tmp, dict):
            return tmp
    except Exception:
        pass
    return _extract_json_from_text(candidate_text)


def _resolve_candidate_action(
    parsed: Optional[Dict[str, Any]],
    original: str,
    candidate_text: str,
    object_seen: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if parsed is None:
        return None

    action_name = parsed.get("action")
    if not isinstance(action_name, str) or action_name not in AVAILABLE_ACTIONS:
        return None

    candidate_action = dict(parsed)
    if "arg1" in candidate_action:
        arg1_old = candidate_action["arg1"]
        candidate_action["arg1"] = _find_uuid_from_text(str(arg1_old), object_seen)
    if "arg2" in candidate_action:
        arg2_old = candidate_action["arg2"]
        candidate_action["arg2"] = _find_uuid_from_text(str(arg2_old), object_seen)

    if action_name in _MOVE_ROTATE_ACTIONS:
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

    return candidate_action


def _fallback_candidate_action(
    original: str,
    object_seen: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    candidate_action = _infer_move_or_rotate(original)
    if candidate_action is not None:
        return candidate_action

    action_name = _infer_object_action(original)
    if action_name in _SINGLE_OBJECT_ACTIONS:
        uuid = _find_uuid_from_text(original, object_seen)
        return {"action": action_name, "arg1": uuid}

    if action_name in _DOUBLE_OBJECT_ACTIONS:
        uuid2 = _find_uuid_from_text("jar", object_seen)
        uuid1 = _find_uuid_from_text(re.sub(r"jar", " ", original.lower()), object_seen)
        if uuid1 and uuid2:
            return {"action": action_name, "arg1": uuid1, "arg2": uuid2}

    return None


def _pack_action_result(
    candidate_action: Optional[Dict[str, Any]],
    valid_uuids: List[str],
) -> Tuple[int, str]:
    if candidate_action is None:
        return 0, _DEFAULT_SAFE_ACTION

    action_name = candidate_action.get("action")
    if action_name in _MOVE_ROTATE_ACTIONS and candidate_action.get("arg1") not in _DIRECTIONS:
        return 0, _DEFAULT_SAFE_ACTION
    if action_name in _SINGLE_OBJECT_ACTIONS and candidate_action.get("arg1") not in valid_uuids:
        return 0, json.dumps(candidate_action)
    if action_name in _DOUBLE_OBJECT_ACTIONS and (
        candidate_action.get("arg1") not in valid_uuids
        or candidate_action.get("arg2") not in valid_uuids
    ):
        return 0, json.dumps(candidate_action)

    return 1, json.dumps(candidate_action, separators=(",", ":"))


def discoveryworld_projection(
    actions: List[str],
    infos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], List[int]]:
    
    # A projection method which is compatible with both text action and JSON action format.
    # Detects think and action blocks with robust heuristics and applies various fallback strategies to extract a valid action.

    processed: List[str] = []
    valids: List[int] = [0] * len(actions)
    contain_think_block: List[int] = [0] * len(actions)
    contain_action_block: List[int] = [0] * len(actions)
    are_json_format: List[int] = [0] * len(actions)

    re_action_block = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)

    object_seen_list = _build_object_seen_list(infos, actions)

    for i, action_str in enumerate(actions):
        action_text = action_str or ""

        (
            contain_think_block[i],
            contain_action_block[i],
            think_has_chinese,
            action_multiple_actions,
        ) = _extract_detection_flags(action_text, re_action_block)

        original = _strip_think_blocks(action_text)

        info_i: Dict[str, Any] = infos[i] if (infos is not None and i < len(infos)) else {}
        if infos is not None and i < len(infos):
            _update_info_flags(
                info_i,
                contain_think_block[i],
                contain_action_block[i],
                think_has_chinese,
                action_multiple_actions,
                are_json_format=0,
            )

        object_seen = object_seen_list[i] if i < len(object_seen_list) else {}

        valid_uuids = _collect_valid_uuids(info_i)

        candidate_text = _select_candidate_text(action_text, re_action_block)

        # If the model chose an action from the provided action-options menu,
        # map that choice back to a canonical string and infer the JSON action.
        # We accept minor formatting noise (bullets, quotes).
        canonical_choice = _resolve_canonical_choice(candidate_text)
        if canonical_choice is not None:
            # Replace `original` with the canonical menu choice so downstream
            # heuristics (_infer_move_or_rotate, _find_uuid_from_text, etc.)
            # can resolve directions / object UUIDs.
            original = canonical_choice
            parsed = None
        else:
            parsed: Optional[Dict[str, Any]] = None
        parsed = _parse_candidate_json(candidate_text)

        candidate_action = _resolve_candidate_action(
            parsed,
            original,
            candidate_text,
            object_seen,
        )
        if candidate_action is None:
            candidate_action = _fallback_candidate_action(original, object_seen)

        if parsed is not None:
            are_json_format[i] = 1
            if infos is not None and i < len(infos):
                info_i["are_json_format"] = 1

        valids[i], processed_action = _pack_action_result(candidate_action, valid_uuids)
        processed.append(processed_action)
        
        if infos is not None and i < len(infos):
            info_i["is_valid"] = valids[i]

    return processed, valids
