from typing import Dict, Any, List, Tuple, Optional
import re
import os
import time


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
OTHER_OBJECTS = ["wall", "floor", "path", "grass", "table"]

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


def slugify(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def build_frames_dir(env_kwargs: Dict[str, Any], seed: int, is_train: bool) -> str:
    scenario = slugify(env_kwargs.get("scenario_name"))
    difficulty = slugify(env_kwargs.get("difficulty"))
    model_name = env_kwargs.get("model_name") or os.environ.get("MODEL_NAME")
    job_id = slugify(os.environ.get("SLURM_JOB_ID"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    split = "train" if is_train else "eval"
    return os.path.join(
        "outputs",
        "discoveryworld_frames",
        f"{model_name}__seed{seed}__{job_id}__{timestamp}__{split}",
    )


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
        lines.append(f"Accessible: {', '.join(accessible_objects)}")
    else:
        lines.append("Accessible: no object is accessible in current location and facing direction")
    
    # 4. Nearby Objects (only interesting objects within certain steps, grouped by direction)
    nearby = ui_obs.get("nearbyObjects", {}).get("objects", {})
    lines.append("Nearby objects:")
    for direction, objects in nearby.items():
        for obj in objects:
            distance = obj.get("distance", 99)
            if distance <= 2 and obj.get("name") not in OTHER_OBJECTS:
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


def coerce_max_chemical_n(env_kwargs: Dict[str, Any], default: int = 2) -> int:
    """Read the canonical chemical amount while accepting legacy config keys."""
    return int(
        env_kwargs.get(
            "max_chemical_n",
            env_kwargs.get("max_chemical_N", env_kwargs.get("chemical_N", default)),
        )
    )