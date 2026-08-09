from typing import Dict, Any, List, Tuple, Optional
import itertools
import json
import re

from .prompts import (
    DISCOVERYWORLD_TEMPLATE,
    DISCOVERYWORLD_TEMPLATE_NO_HIS,
)

from agent_system.environments.env_package.discovery.config import (
    build_frames_dir,
    coerce_max_chemical_n,
    slugify,
)
all_plausible_action_json = [
    {"action": "PICKUP", "arg1": "33120"},
    {"action": "MOVE_DIRECTION", "arg1": "west"},
    {"action": "ROTATE_DIRECTION", "arg1": "north"},
    {"action": "PUT", "arg1": "33120", "arg2": "35632"},
    {"action": "PICKUP", "arg1": "35632"},
    {"action": "ROTATE_DIRECTION", "arg1": "east"},
    {"action": "MOVE_DIRECTION", "arg1": "east"},
    {"action": "ROTATE_DIRECTION", "arg1": "south"},
    {"action": "OPEN", "arg1": "18573"},
    {"action": "MOVE_DIRECTION", "arg1": "south"},
    {"action": "USE", "arg1": "21559", "arg2": "35632"},
    {"action": "USE", "arg1": "57736", "arg2": "35632"},
    {"action": "USE", "arg1": "8549", "arg2": "35632"},
    {"action": "USE", "arg1": "55934", "arg2": "35632"},
    {"action": "USE", "arg1": "51739", "arg2": "35632"},
]

all_plausible_action_mapper = {
    "Move west": {"action": "MOVE_DIRECTION", "arg1": "west"},
    "Move east": {"action": "MOVE_DIRECTION", "arg1": "east"},
    "Move south": {"action": "MOVE_DIRECTION", "arg1": "south"},
    "Rotate north": {"action": "ROTATE_DIRECTION", "arg1": "north"},
    "Rotate east": {"action": "ROTATE_DIRECTION", "arg1": "east"},
    "Rotate south": {"action": "ROTATE_DIRECTION", "arg1": "south"},
    "Rotate west": {"action": "ROTATE_DIRECTION", "arg1": "west"},
    "Pick up the key": {"action": "PICKUP", "arg1": "33120"},
    "Put the key in the jar": {"action": "PUT", "arg1": "33120", "arg2": "35632"},
    "Pick up the jar": {"action": "PICKUP", "arg1": "35632"},
    "Use the dispenser A to deliver substance A to the jar": {"action": "USE", "arg1": "21559", "arg2": "35632"},
    "Use the dispenser B to deliver substance B to the jar": {"action": "USE", "arg1": "57736", "arg2": "35632"},
    "Use the dispenser C to deliver substance C to the jar": {"action": "USE", "arg1": "8549", "arg2": "35632"},
    "Use the dispenser D to deliver substance D to the jar": {"action": "USE", "arg1": "55934", "arg2": "35632"},
    "Wash the jar to clean substances": {"action": "USE", "arg1": "51739", "arg2": "35632"},
    "Open the door": {"action": "OPEN", "arg1": "18573"},
}

all_plausible_action_mapper_no_uuid = {
    "Move west": {"action": "MOVE_DIRECTION", "arg1": "west"},
    "Move east": {"action": "MOVE_DIRECTION", "arg1": "east"},
    "Move south": {"action": "MOVE_DIRECTION", "arg1": "south"},
    "Rotate north": {"action": "ROTATE_DIRECTION", "arg1": "north"},
    "Rotate east": {"action": "ROTATE_DIRECTION", "arg1": "east"},
    "Rotate south": {"action": "ROTATE_DIRECTION", "arg1": "south"},
    "Rotate west": {"action": "ROTATE_DIRECTION", "arg1": "west"},
    "Pick up the key": {"action": "PICKUP", "arg1": "key"},
    "Put the key in the jar": {"action": "PUT", "arg1": "key", "arg2": "jar"},
    "Pick up the jar": {"action": "PICKUP", "arg1": "jar"},
    "Use the dispenser A to deliver substance A to the jar": {"action": "USE", "arg1": "dispenser A", "arg2": "jar"},
    "Use the dispenser B to deliver substance B to the jar": {"action": "USE", "arg1": "dispenser B", "arg2": "jar"},
    "Use the dispenser C to deliver substance C to the jar": {"action": "USE", "arg1": "dispenser C", "arg2": "jar"},
    "Use the dispenser D to deliver substance D to the jar": {"action": "USE", "arg1": "dispenser D", "arg2": "jar"},
    "Wash the jar to clean substances": {"action": "USE", "arg1": "bottle_cleaner", "arg2": "jar"},
    "Open the door": {"action": "OPEN", "arg1": "door"},
}

all_action_abbr = {
    "move_west": {"action": "MOVE_DIRECTION", "arg1": "west"},
    "move_east": {"action": "MOVE_DIRECTION", "arg1": "east"},
    "move_south": {"action": "MOVE_DIRECTION", "arg1": "south"},
    "move_north": {"action": "MOVE_DIRECTION", "arg1": "north"},
    "rotate_north": {"action": "ROTATE_DIRECTION", "arg1": "north"},
    "rotate_east": {"action": "ROTATE_DIRECTION", "arg1": "east"},
    "rotate_south": {"action": "ROTATE_DIRECTION", "arg1": "south"},
    "rotate_west": {"action": "ROTATE_DIRECTION", "arg1": "west"},
    "pickup_key": {"action": "PICKUP", "arg1": "33120"},
    "put_key": {"action": "PUT", "arg1": "33120", "arg2": "35632"},
    "pickup_jar": {"action": "PICKUP", "arg1": "35632"},
    "use_dispenser_A": {"action": "USE", "arg1": "21559", "arg2": "35632"},
    "use_dispenser_B": {"action": "USE", "arg1": "57736", "arg2": "35632"},
    "use_dispenser_C": {"action": "USE", "arg1": "8549", "arg2": "35632"},
    "use_dispenser_D": {"action": "USE", "arg1": "55934", "arg2": "35632"},
    "wash": {"action": "USE", "arg1":"51739", "arg2":"35632"},
    "open_door": {"action": "OPEN", "arg1": "18573"},
}

uuid_to_name = {
    "33120": "Key",
    "35632": "Jar",
    "21559": "Dispenser A",
    "57736": "Dispenser B",
    "8549": "Dispenser C",
    "55934": "Dispenser D",
    "51739": "Bottle Cleaner",
    "18573": "Door",
}

DISPENSER_NAMES = ["Dispenser (Substance A)", "Dispenser (Substance B)", "Dispenser (Substance C)", "Dispenser (Substance D)"]
RUSTED_KEY = "rusted key (heavily rusted)"
RUSTED_KEY_2 = "rusted key (moderately rusted)"
RUSTED_KEY_3 = "rusted key (lightly rusted)"
KEY_NO_RUST = "key (no rust)"
JAR = "jar"
DOOR = "door"
TABLE = "table"
OTHER_OBJECTS = ["wall", "floor", "path", "grass", "table", "agent"]

KEY_NAME_TO_RUST_LABEL = {
    RUSTED_KEY: "heavily rusted",
    RUSTED_KEY_2: "moderately rusted",
    RUSTED_KEY_3: "lightly rusted",
    KEY_NO_RUST: "no rust",
}

RUST_LABEL_TO_LEVEL = {
    "no rust": 0,
    "lightly rusted": 1,
    "moderately rusted": 2,
    "heavily rusted": 3,
}

MOVE_TO_KEY = "move_to_key"
MOVE_TO_JAR = "move_to_jar"
PICK_UP_KEY = "pick_up_key"
PICK_UP_JAR = "pick_up_jar"
PUT_KEY_IN_JAR = "put_key_in_jar"
USE_DISPENSER_A = "use_dispenser_A_on_jar"
USE_DISPENSER_B = "use_dispenser_B_on_jar"
USE_DISPENSER_C = "use_dispenser_C_on_jar"
USE_DISPENSER_D = "use_dispenser_D_on_jar"
REMOVE_CHEMICAL_A = "remove_chemical_A"
REMOVE_CHEMICAL_B = "remove_chemical_B"
REMOVE_CHEMICAL_C = "remove_chemical_C"
REMOVE_CHEMICAL_D = "remove_chemical_D"
WASH = "wash_jar"
OPEN_DOOR = "open_door"

ORDERED_SKILL_NAMES = (
    MOVE_TO_KEY,
    MOVE_TO_JAR,
    PICK_UP_KEY,
    PICK_UP_JAR,
    PUT_KEY_IN_JAR,
    USE_DISPENSER_A,
    REMOVE_CHEMICAL_A,
    USE_DISPENSER_B,
    REMOVE_CHEMICAL_B,
    USE_DISPENSER_C,
    REMOVE_CHEMICAL_C,
    USE_DISPENSER_D,
    REMOVE_CHEMICAL_D,
    WASH,
    OPEN_DOOR,
)

SKILL_NAMES = {
    MOVE_TO_KEY,
    MOVE_TO_JAR,
    PICK_UP_KEY,
    PICK_UP_JAR,
    PUT_KEY_IN_JAR,
    USE_DISPENSER_A,
    USE_DISPENSER_B,
    USE_DISPENSER_C,
    USE_DISPENSER_D,
    REMOVE_CHEMICAL_A,
    REMOVE_CHEMICAL_B,
    REMOVE_CHEMICAL_C,
    REMOVE_CHEMICAL_D,
    WASH,
    OPEN_DOOR
}


def _object_names_by_location(ui: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    inventory = ui.get("inventoryObjects", []) or []
    accessible = ui.get("accessibleEnvironmentObjects", []) or []
    inv_objects = {
        obj.get("name"): obj
        for obj in inventory
        if obj.get("name") and obj.get("name") not in OTHER_OBJECTS
    }
    accessible_objects = {
        obj.get("name"): obj
        for obj in accessible
        if obj.get("name") and obj.get("name") not in OTHER_OBJECTS
    }
    return inv_objects, accessible_objects


def get_valid_discoveryworld_skills(info: Optional[Dict[str, Any]], max_chemical_n: int = 2) -> List[str]:
    """Return physically valid skills without belief-based target filtering."""
    info = info or {}
    ui = (info.get("raw_observation") or {}).get("ui", {})
    inv_objects, accessible_objects = _object_names_by_location(ui)
    location_info = ui.get("agentLocation", {}) or {}
    location = (location_info.get("x"), location_info.get("y"))

    rusted_key_names = (RUSTED_KEY, RUSTED_KEY_2, RUSTED_KEY_3)
    rusted_key_in_hand = any(name in inv_objects for name in rusted_key_names)
    rusted_key_accessible = any(name in accessible_objects for name in rusted_key_names)
    clean_key_in_hand = KEY_NO_RUST in inv_objects
    clean_key_accessible = KEY_NO_RUST in accessible_objects

    has_key = bool(info.get("has_key")) or rusted_key_in_hand or clean_key_in_hand
    has_jar = bool(info.get("has_jar")) or JAR in inv_objects
    is_key_in_jar = bool(info.get("is_key_in_jar"))
    rust_status = str(info.get("key_rust_status") or "").strip().lower()

    if clean_key_accessible and not clean_key_in_hand:
        return [PICK_UP_KEY]

    if clean_key_in_hand or (has_key and rust_status == "no rust"):
        return [OPEN_DOOR]

    if not has_key and not is_key_in_jar:
        if rusted_key_accessible or location == (17, 12):
            return [PICK_UP_KEY]
        return [MOVE_TO_KEY]

    if (has_key or is_key_in_jar) and not has_jar:
        if JAR in accessible_objects or location == (17, 12):
            return [PICK_UP_JAR]
        return [MOVE_TO_JAR]

    if has_key and has_jar and not is_key_in_jar:
        return [PUT_KEY_IN_JAR]

    if is_key_in_jar and has_jar:
        chemical_dict = info.get("chemical_dict") or {}
        chemical_total = sum(int(chemical_dict.get(name, 0) or 0) for name in ("A", "B", "C", "D"))
        max_chemical_n = max(1, int(max_chemical_n or 1))

        if rust_status == "no rust":
            return [OPEN_DOOR]

        if chemical_total < max_chemical_n:
            return [USE_DISPENSER_A, USE_DISPENSER_B, USE_DISPENSER_C, USE_DISPENSER_D]

        # Removing an absent chemical cannot change the state. Mask only those
        # physically impossible no-ops; all present chemicals remain equally
        # available, without belief- or teacher-based recommendations.
        return [
            skill
            for name, skill in zip(
                CHEMICAL_NAMES,
                (
                    REMOVE_CHEMICAL_A,
                    REMOVE_CHEMICAL_B,
                    REMOVE_CHEMICAL_C,
                    REMOVE_CHEMICAL_D,
                ),
            )
            if int(chemical_dict.get(name, 0) or 0) > 0
        ]

    return list(ORDERED_SKILL_NAMES)


def key_name_to_rust_level_label(key_name: Optional[str]) -> Optional[str]:
    if not isinstance(key_name, str):
        return None

    normalized = key_name.strip()
    if not normalized:
        return None

    if normalized in KEY_NAME_TO_RUST_LABEL:
        return KEY_NAME_TO_RUST_LABEL[normalized]

    for known_name, rust_label in KEY_NAME_TO_RUST_LABEL.items():
        if normalized.startswith(known_name):
            return rust_label

    return None


def format_rust_level(rust_level: Any) -> str:
    if rust_level is None:
        return "unknown"

    if isinstance(rust_level, str):
        # Accept either full key names (e.g. "rusted key (heavily rusted)")
        # or direct labels (e.g. "heavily rusted").
        from_key_name = key_name_to_rust_level_label(rust_level)
        if from_key_name is not None:
            return from_key_name

        normalized = rust_level.strip().lower()
        return normalized if normalized in RUST_LABEL_TO_LEVEL else "unknown"

    try:
        level_value = int(rust_level)
    except (TypeError, ValueError):
        return "unknown"

    labels = {
        0: "no rust",
        1: "lightly rusted",
        2: "moderately rusted",
        3: "heavily rusted",
    }
    return labels.get(level_value, "unknown")


def format_rust_update(previous_level: Any, current_level: Any) -> str:
    current_label = format_rust_level(current_level)
    if current_label == "unknown":
        return "Rust level update: unknown"

    if previous_level is None:
        return f"Rust level update: {current_label}"

    previous_label = format_rust_level(previous_level)
    if previous_label == "unknown":
        return f"Rust level update: {current_label}"

    previous_value = RUST_LABEL_TO_LEVEL.get(previous_label)
    current_value = RUST_LABEL_TO_LEVEL.get(current_label)
    if previous_value is None or current_value is None:
        return f"Rust level update: {current_label}"

    if current_value < previous_value:
        change = "improved"
    elif current_value > previous_value:
        change = "worsened"
    else:
        change = "no change"

    return f"Rust level update: {current_label} ({change})"


def extract_detailed_status(ui: Dict):
    inventory = ui.get("inventoryObjects", [])
    accessible = ui.get("accessibleEnvironmentObjects", [])

    def add_chemical_counts_from_name(name: str, chemical_dict: Dict[str, int]) -> None:
        if not name:
            return

        normalized_name = name.strip()

        pure_match = re.match(
            r"^Substance\s+([A-D])(?:\s*\((\d+)\s+measures?\))?$",
            normalized_name,
            flags=re.IGNORECASE,
        )
        if pure_match:
            chemical = pure_match.group(1).upper()
            count = int(pure_match.group(2) or 1)
            chemical_dict[chemical] = chemical_dict.get(chemical, 0) + count
            return

        mixture_match = re.match(r"^mixture\s*\((.*)\)$", normalized_name, flags=re.IGNORECASE)
        if mixture_match:
            parts = [part.strip() for part in mixture_match.group(1).split(",") if part.strip()]
            for part in parts:
                part_match = re.match(
                    r"^(\d+)\s+parts?\s+Substance\s+([A-D])$",
                    part,
                    flags=re.IGNORECASE,
                )
                if part_match:
                    count = int(part_match.group(1))
                    chemical = part_match.group(2).upper()
                    chemical_dict[chemical] = chemical_dict.get(chemical, 0) + count

    is_key_in_jar = False
    for obj in inventory + accessible:
        if key_name_to_rust_level_label(obj.get("name")) is not None:
            description = obj.get("description", "")
            if "in jar" in description:
                is_key_in_jar = True
                break

    inv_objects = {obj.get("name") for obj in inventory or []}
    has_key = any(key_name_to_rust_level_label(obj_name) is not None for obj_name in inv_objects)
    has_jar = any(obj_name == JAR or (isinstance(obj_name, str) and obj_name.startswith("jar")) for obj_name in inv_objects)

    key_rust_level = None
    for obj in inventory or []:
        key_rust_level = key_name_to_rust_level_label(obj.get("name"))
        if key_rust_level is not None:
            break
    
    chemical_dict = {"A": 0, "B": 0, "C": 0, "D": 0}
    for name in inv_objects:
        add_chemical_counts_from_name(name or "", chemical_dict)

    return has_key, has_jar, is_key_in_jar, chemical_dict, key_rust_level


def compress_ui_observation(ui_obs: dict) -> str:
    """
    Compress UI observation from ~4000 tokens to <500 tokens.
    Convert verbose UI JSON into a compact structural text representation.
    """
        
    def _strip_uuid(text: str) -> str:
        if not text:
            return text
        return re.sub(r"\s*\[uuid:\s*[^\]]+\]", "", text).strip()

    lines = []
    
    # 1. Agent Location, facing direction
    loc = ui_obs.get("agentLocation", {})
    if loc:
        facing = loc.get("faceDirection", "unknown")
        can_move = ", ".join(loc.get("directions_you_can_move", []))
        blocked = ", ".join(loc.get("directions_blocked", []))
        lines.append(f"Location: ({loc.get('x', '?')}, {loc.get('y', '?')}), facing {facing}")
        if can_move:
            lines.append(f"Can move: {can_move}")
        if blocked:
            lines.append(f"Blocked: {blocked}")
    
    # 2. Inventory
    inventory = ui_obs.get("inventoryObjects", [])
    if inventory:
        items = [_strip_uuid(f"{obj.get('description', '')}")
                    for obj in inventory if obj.get("name") not in OTHER_OBJECTS]
        lines.append(f"Inventory: {', '.join(items)}")
    else:
        lines.append("Inventory: empty")
    
    # 3. Accessible Objects
    accessible = ui_obs.get("accessibleEnvironmentObjects", [])
    accessible_objects = [_strip_uuid(f"{obj.get('description', '')}")
                            for obj in accessible if obj.get("name") not in OTHER_OBJECTS]
    if accessible_objects:
        lines.append(f"Accessible (facing): {', '.join(accessible_objects)}")
    else:
        lines.append("Accessible (facing): no object is accessible in current location and facing direction")
    
    # 4. Nearby Objects (only interesting objects within certain steps, grouped by direction)
    nearby = ui_obs.get("nearbyObjects", {}).get("objects", {})
    lines.append("Nearby objects:")
    for direction, objects in nearby.items():
        for obj in objects:
            distance = obj.get("distance", 99)
            if 0 < distance <= 2 and obj.get("name") not in OTHER_OBJECTS:
                desc = _strip_uuid(f"{obj.get('description', '')}")
                lines.append(f"- {direction} ({distance} tile(s) away): {desc}")
    
    # 5. Action message
    #last_msg = ui_obs.get("lastActionMessage", "")
    #extended_msg = ui_obs.get("extended_action_message", "")
    #if last_msg:
    #    lines.append(f"\nLast action message: {last_msg}")
    #if extended_msg:
    #    lines.append(f"Extended info: {extended_msg}")
    
    # 6. Task completion
    task_progress = ui_obs.get("taskProgress", [])[0] if ui_obs.get("taskProgress") else {}
    success = task_progress.get("completed", False)
    lines.append(f"\nTask completed: {success}")
    
    return "\n".join(lines)


def is_dispenser_skill(skill_name: Optional[str]) -> bool:
    """Whether a skill belongs to the interchangeable "add chemical" class."""
    return bool(
        isinstance(skill_name, str)
        and skill_name.startswith("use_dispenser_")
        and skill_name.endswith("_on_jar")
    )

def is_remove_skill(skill_name: Optional[str]) -> bool:
    """Whether a skill belongs to the interchangeable "remove chemical" class."""
    return bool(
        isinstance(skill_name, str)
        and skill_name.startswith("remove_chemical_")
    )


CHEMICAL_NAMES = ("A", "B", "C", "D")


def canonical_chemical_belief(
    current_mixture: Tuple[int, ...],
    current_reaction_signal: str,
    candidate_targets: List[Tuple[int, ...]],
) -> Tuple[Tuple[int, ...], str, Tuple[Tuple[int, ...], ...]]:
    """Return the exact structured key used by the chemical-stage anchor."""
    return (
        tuple(int(value) for value in current_mixture),
        str(current_reaction_signal or "not tested").strip().lower(),
        tuple(sorted(tuple(int(value) for value in target) for target in candidate_targets)),
    )


def chemical_counts(chemical_dict: Optional[Dict[str, Any]]) -> Dict[str, int]:
    chemical_dict = chemical_dict or {}
    return {name: int(chemical_dict.get(name, 0) or 0) for name in CHEMICAL_NAMES}


def chemical_tuple_text(chemical_dict: Optional[Dict[str, Any]]) -> str:
    counts = chemical_counts(chemical_dict)
    return "(" + ",".join(str(counts[name]) for name in CHEMICAL_NAMES) + ")"


def similarity_bucket_for_tuples(
    mixture: Tuple[int, int, int, int],
    target: Tuple[int, int, int, int],
) -> Optional[str]:
    mixture_norm = sum(value * value for value in mixture) ** 0.5
    target_norm = sum(value * value for value in target) ** 0.5
    if mixture_norm == 0 or target_norm == 0:
        return None
    similarity = sum(left * right for left, right in zip(mixture, target)) / (
        mixture_norm * target_norm
    )
    if similarity >= 0.99:
        return "no rust"
    if similarity >= 0.66:
        return "lightly rusted"
    if similarity >= 0.33:
        return "moderately rusted"
    return "heavily rusted"


RUST_TO_REACTION_SIGNAL = {
    "no rust": "successful",
    "lightly rusted": "strong",
    "moderately rusted": "weak",
    "heavily rusted": "none",
}
REACTION_SIGNAL_TO_RUST = {value: key for key, value in RUST_TO_REACTION_SIGNAL.items()}


def reaction_signal_for_tuples(
    mixture: Tuple[int, int, int, int],
    target: Tuple[int, int, int, int],
) -> str:
    """Return a reversible, coarse assay reading for the current mixture."""
    rust_bucket = similarity_bucket_for_tuples(mixture, target)
    return RUST_TO_REACTION_SIGNAL.get(rust_bucket, "not tested")


def candidate_targets(
    combo_results: Dict[Tuple[int, int, int, int], Tuple[str, str]],
    required_amount: int,
) -> List[Tuple[int, int, int, int]]:
    candidates = (
        tuple(values)
        for values in itertools.product(range(required_amount + 1), repeat=4)
        if sum(values) == required_amount
    )
    return [
        candidate
        for candidate in candidates
        if all(
            _candidate_matches_evidence(mixture, candidate, kind, label)
            for mixture, (kind, label) in combo_results.items()
        )
    ]


def _candidate_matches_evidence(
    mixture: Tuple[int, int, int, int],
    candidate: Tuple[int, int, int, int],
    kind: str,
    label: str,
) -> bool:
    label = REACTION_SIGNAL_TO_RUST.get(label, label)
    predicted = similarity_bucket_for_tuples(mixture, candidate)
    if predicted is None or label not in RUST_LABEL_TO_LEVEL:
        return False
    if kind == "observed":
        return predicted == label
    if kind == "not_better_than":
        return RUST_LABEL_TO_LEVEL[predicted] >= RUST_LABEL_TO_LEVEL[label]
    return False


def observable_experiment_evidence(
    previous_info: Dict[str, Any],
    current_info: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Build exact evidence from the reversible, agent-visible reaction assay."""
    current_combo = chemical_counts(current_info.get("chemical_dict"))
    combo = tuple(current_combo[name] for name in CHEMICAL_NAMES)
    if not current_info.get("is_key_in_jar") or not any(combo):
        return None

    previous_combo = chemical_counts(previous_info.get("chemical_dict"))
    previous_tuple = tuple(previous_combo[name] for name in CHEMICAL_NAMES)
    newly_tested = not previous_info.get("is_key_in_jar") or combo != previous_tuple
    if not newly_tested:
        return None

    current_signal = str(current_info.get("current_reaction_signal") or "").strip().lower()
    if current_signal not in REACTION_SIGNAL_TO_RUST:
        return None
    return {"kind": "observed", "label": current_signal}


def collect_experiment_evidence(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[int, int, int, int], Tuple[str, str]]:
    results: Dict[Tuple[int, int, int, int], Tuple[str, str]] = {}
    for record in records:
        kind = str(record.get("experiment_evidence_kind") or "")
        label = str(record.get("experiment_evidence_label") or "").strip().lower()
        if label in RUST_LABEL_TO_LEVEL:  # Backward-compatible old rollout records.
            label = RUST_TO_REACTION_SIGNAL[label]
        if kind not in {"observed", "not_better_than"} or label not in REACTION_SIGNAL_TO_RUST:
            continue
        combo_counts = chemical_counts(record.get("post_chemical_dict"))
        combo = tuple(combo_counts[name] for name in CHEMICAL_NAMES)
        results.pop(combo, None)
        results[combo] = (kind, label)
    return results


def build_chemical_belief_state(
    records: List[Dict[str, Any]],
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a structured belief solely from the complete visible experiment log."""
    evidence = collect_experiment_evidence(records)
    required_amount = int(info.get("max_chemical_n", 0) or 0)
    remaining = candidate_targets(evidence, required_amount) if required_amount > 0 else []

    experiments = []
    for record in records:
        kind = str(record.get("experiment_evidence_kind") or "")
        label = str(record.get("experiment_evidence_label") or "").strip().lower()
        if label in RUST_LABEL_TO_LEVEL:
            label = RUST_TO_REACTION_SIGNAL[label]
        if kind not in {"observed", "not_better_than"} or label not in REACTION_SIGNAL_TO_RUST:
            continue
        combo_counts = chemical_counts(record.get("post_chemical_dict"))
        experiments.append({
            "mixture": [combo_counts[name] for name in CHEMICAL_NAMES],
            "reaction_signal": label,
        })
    return {
        "experiments": experiments,
        "candidate_count": len(remaining),
        "candidate_targets": [list(target) for target in remaining],
    }


def build_chemical_belief(
    records: List[Dict[str, Any]],
    info: Dict[str, Any],
    max_records: Optional[int] = None,
) -> str:
    """Format the complete structured belief for the policy prompt."""
    belief = build_chemical_belief_state(records, info)

    sections = []
    if belief["experiments"]:
        lines = [
            f"({','.join(map(str, experiment['mixture']))}) -> {experiment['reaction_signal']}"
            for experiment in belief["experiments"]
        ]
        sections.append("Experiments:\n" + "\n".join(lines))
    else:
        sections.append("No chemical combination has been observed yet.")

    remaining_text = ", ".join(
        f"({','.join(map(str, target))})" for target in belief["candidate_targets"]
    )
    #sections.append(
    #    f"Remaining candidate targets ({belief['candidate_count']}): "
    #    + (remaining_text or "none")
    #)
    return "\n".join(sections)


def build_discoveryworld_anchor_obs(
    text_obs: List[str],
    infos: List[Dict[str, Any]],
    memory: Any,
    config: Any,
) -> List[str]:
    discovery_cfg = getattr(config.env, "discoveryworld", None)
    anchor_mode = str(getattr(discovery_cfg, "anchor_mode", "belief_summary")).strip().lower()
    anchors = []
    for i, obs in enumerate(text_obs):
        if anchor_mode in {"raw", "raw_obs", "text_obs"}:
            anchors.append(obs)
            continue
        info = infos[i]
        records = memory[i] if i < len(memory) else []
        if info.get("is_key_in_jar"):
            belief = build_chemical_belief_state(records, info)
            counts = chemical_counts(info.get("chemical_dict"))
            canonical = canonical_chemical_belief(
                [counts[name] for name in CHEMICAL_NAMES],
                info.get("current_reaction_signal", "not tested"),
                belief["candidate_targets"],
            )
            anchors.append(json.dumps({
                "current_mixture": list(canonical[0]),
                "current_reaction_signal": canonical[1],
                "candidate_targets": [list(target) for target in canonical[2]],
            }, sort_keys=True))
            continue
        state_summary = {
            "state_obs": obs,
            "chemical_dict": chemical_counts(info.get("chemical_dict")),
            "key_rust_status": str(info.get("key_rust_status") or "unknown").strip().lower(),
            "current_reaction_signal": str(
                info.get("current_reaction_signal") or "not tested"
            ).strip().lower(),
            "has_key": bool(info.get("has_key")),
            "has_jar": bool(info.get("has_jar")),
            "is_key_in_jar": bool(info.get("is_key_in_jar")),
            "used_dispensers": {
                key: bool(value)
                for key, value in sorted((info.get("used_dispensers") or {}).items())
            },
        }
        if anchor_mode not in {"state", "state_summary"}:
            state_summary["chemical_belief"] = build_chemical_belief_state(records, info)
        anchors.append(json.dumps(state_summary, sort_keys=True))
    return anchors


def build_discoveryworld_text_obs(
    text_obs: List[str],
    infos: List[Dict[str, Any]],
    memory: Any,
    config: Any,
    init: bool = False,
) -> List[str]:
    discovery_cfg = getattr(config.env, "discoveryworld", None)
    max_chemical_n = getattr(
        discovery_cfg,
        "max_chemical_n",
        getattr(discovery_cfg, "max_chemical_N", getattr(discovery_cfg, "chemical_N", 2)),
    )
    results = []
    for i, _raw_obs in enumerate(text_obs):
        records = memory[i] if i < len(memory) else []
        info = infos[i]
        counts = chemical_counts(info.get("chemical_dict"))
        mixture = tuple(counts[name] for name in CHEMICAL_NAMES)
        chemical_amount = sum(mixture)
        rust_status = str(info.get("key_rust_status") or "unknown").strip().lower()
        reaction_signal = str(
            info.get("current_reaction_signal") or "not tested"
        ).strip().lower()

        if rust_status == "no rust":
            phase = "open_door"
        elif info.get("is_key_in_jar"):
            phase = "chemical_experiment"
        elif info.get("has_key") and info.get("has_jar"):
            phase = "prepare_experiment"
        elif info.get("has_key"):
            phase = "collect_jar"
        else:
            phase = "find_key"

        state_obs = "\n".join((
            f"Step: {len(records)} / {config.env.max_steps}",
            f"Phase: {phase}",
            f"Key: {rust_status}",
            f"Chemical mixture: ({','.join(map(str, mixture))}), "
            f"chemical amount: {chemical_amount}/{max_chemical_n}",
            f"Reaction signal: {reaction_signal}",
        ))
        template_args = {
            "max_chemical_n": max_chemical_n,
            "state_obs": state_obs,
            "chemical_belief": build_chemical_belief(records, info),
        }
        valid_skill_names = get_valid_discoveryworld_skills(
            info, max_chemical_n=max_chemical_n
        )
        info["valid_skills"] = list(valid_skill_names)
        template_args["valid_skills"] = "\n".join(valid_skill_names)
        in_chemical_stage = bool(info.get("is_key_in_jar"))
        if init or in_chemical_stage or config.env.history_length <= 0 or not records:
            obs = DISCOVERYWORLD_TEMPLATE_NO_HIS.format(**template_args)
        else:
            recent_records = records[-config.env.history_length:]
            history_start = len(records) - len(recent_records)
            previous_rust_level = records[history_start - 1].get("rust_level") if history_start > 0 else None
            memory_lines = []
            for step_offset, record in enumerate(recent_records):
                action = record.get("action")
                rust_level = record.get("rust_level")
                if not action:
                    continue
                memory_lines.append(
                    f"{step_offset + 1}. {action} -> {format_rust_update(previous_rust_level, rust_level)}"
                )
                previous_rust_level = rust_level
            template_args["memory_actions"] = "\n".join(memory_lines)
            obs = DISCOVERYWORLD_TEMPLATE.format(**template_args)
        results.append(obs)
    return results
