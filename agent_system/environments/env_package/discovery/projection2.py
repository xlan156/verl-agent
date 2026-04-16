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
import difflib
from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from agent_system.environments.env_package.discovery.actions import all_plausible_action_mapper

AVAILABLE_ACTIONS = DiscoveryWorldAPI.listKnownActionsStatic()

# 用于解析 LLM 输出中的代码块
JSON_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)
GENERIC_RE = re.compile(r"```(.*?)```", re.DOTALL)

_DEFAULT_SAFE_ACTION = json.dumps({"action": "MOVE_DIRECTION", "arg1": "west"}, separators=(",", ":"))

_OBJECT_ACTION_PATTERNS = [
    (re.compile(r"\bwash\b|\brinse\b|\bclean\b", re.IGNORECASE), "USE"),
    (re.compile(r"\bpick(?:\s+up)?\b|\btake\b", re.IGNORECASE), "PICKUP"),
    (re.compile(r"\bopen\b", re.IGNORECASE), "OPEN"),
    (re.compile(r"\buse\b", re.IGNORECASE), "USE"),
    (re.compile(r"\bput\b|\bplace\b", re.IGNORECASE), "PUT"),
    (re.compile(r"\bmove\b|\bgo\b|\bwalk\b|\bstep\b", re.IGNORECASE), "MOVE_DIRECTION"),
    (re.compile(r"\brotate\b|\bturn\b|\bface\b", re.IGNORECASE), "ROTATE_DIRECTION"),
]


def _normalize_action_choice(text: str) -> str:
    """Normalize an action choice emitted by the model.

    Handles small formatting variations like bullets/numbering and quotes.
    """
    s = (text or "").strip()
    # Strip leading bullets / numbering.
    s = re.sub(r"^\s*(?:[-*]|\d+[\.)])\s+", "", s)
    s = s.strip().strip('"').strip("'").strip()
    return s


def _normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching against action-menu keys."""
    s = (text or "").lower().strip()
    # Keep letters/numbers; collapse the rest to spaces.
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


_ACTION_KEYS: List[str] = list(all_plausible_action_mapper.keys())
_ACTION_KEYS_NORM: List[str] = [_normalize_for_match(k) for k in _ACTION_KEYS]


def _match_action_key(raw_choice: str) -> Optional[str]:
    """Return the best matching action-menu key for a model output string.

    Uses:
      - exact normalized match
      - direction-special-cases ("go west" -> "Move west")
      - fuzzy similarity + token overlap
    """
    choice = _normalize_action_choice(raw_choice)
    if not choice:
        return None

    choice_norm = _normalize_for_match(choice)
    if not choice_norm:
        return None

    # Exact normalized match.
    for k, k_norm in zip(_ACTION_KEYS, _ACTION_KEYS_NORM):
        if choice_norm == k_norm:
            return k

    # Direction-based normalization (covers outputs like "go west" / "turn north").
    direction = _extract_direction(choice_norm)
    if direction:
        if re.search(r"\b(move|go|walk|step)\b", choice_norm):
            candidate = f"Move {direction}"
            if candidate in all_plausible_action_mapper:
                return candidate
        if re.search(r"\b(rotate|turn|face)\b", choice_norm):
            candidate = f"Rotate {direction}"
            if candidate in all_plausible_action_mapper:
                return candidate

    # Fuzzy match: combine SequenceMatcher ratio with token overlap.
    choice_tokens = set(choice_norm.split())
    best_key: Optional[str] = None
    best_score = 0.0

    for k, k_norm in zip(_ACTION_KEYS, _ACTION_KEYS_NORM):
        if not k_norm:
            continue
        k_tokens = set(k_norm.split())
        if not k_tokens:
            continue

        # Substring is a strong signal (e.g., "open door" vs "open the door").
        if choice_norm in k_norm or k_norm in choice_norm:
            score = 1.0
        else:
            seq = difflib.SequenceMatcher(None, choice_norm, k_norm).ratio()
            inter = len(choice_tokens & k_tokens)
            union = len(choice_tokens | k_tokens)
            jaccard = (inter / union) if union else 0.0
            score = max(seq, jaccard)

            # If the model emitted a shortened version (tokens mostly contained
            # within the true key), boost the score.
            if len(choice_tokens) >= 2:
                coverage = inter / len(choice_tokens) if choice_tokens else 0.0
                if choice_tokens.issubset(k_tokens) or coverage >= 0.85:
                    score = max(score, 0.9)

        if score > best_score:
            best_score = score
            best_key = k

    # Conservative threshold to avoid accidental mismaps.
    if best_key is not None and best_score >= 0.72:
        return best_key
    return None

_MOVE_ROTATE_ACTIONS = {"MOVE_DIRECTION", "ROTATE_DIRECTION"}
_DIRECTIONS = {"north", "south", "east", "west"}
_SINGLE_OBJECT_ACTIONS = {
    "PICKUP",
    "OPEN",
}
_DOUBLE_OBJECT_ACTIONS = {"USE", "PUT"}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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

        # --- Detection 1: <think> block contains Chinese (CJK) characters ---
        think_blocks = re.findall(
            r"<think>(.*?)</think>",
            action_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        think_text = "\n".join([t.strip() for t in think_blocks if t is not None])
        think_has_chinese = 1 if _contains_cjk(think_text) else 0

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
                # Not valid JSON; check for multiple top-level {...} objects.
                if _count_top_level_json_objects(action_payload) > 1:
                    action_multiple_actions = 1

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
            info_i["think_has_chinese"] = think_has_chinese
            info_i["action_multiple_actions"] = action_multiple_actions

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

        # Prefer mapping the model's <action> content to an action-menu key.
        # If we cannot confidently match a key, we do NOT try to guess; we will
        # fall back to JSON parsing, and then to default-safe invalid action.
        matched_key = _match_action_key(candidate_text)

        parsed: Optional[Dict[str, Any]] = None
        try:
            tmp = json.loads(candidate_text)
            if isinstance(tmp, dict):
                parsed = tmp
        except Exception:
            parsed = _extract_json_from_text(candidate_text)

        candidate_action: Optional[Dict[str, Any]] = None

        if matched_key is not None:
            candidate_action = dict(all_plausible_action_mapper.get(matched_key, {}))
            if infos is not None and i < len(infos):
                info_i["action_key"] = matched_key

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

        if candidate_action is None:
            valids[i] = 0
            processed.append(_DEFAULT_SAFE_ACTION)
        else:
            if candidate_action.get("action") in _MOVE_ROTATE_ACTIONS and candidate_action.get("arg1") not in _DIRECTIONS:
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
        
        if infos is not None and i < len(infos):
            info_i["is_valid"] = valids[i]

    return processed, valids
