from typing import Dict, Any, List, Tuple, Optional
import re

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

def extract_detailed_status(ui: Dict):
    inventory = ui.get("inventoryObjects", [])
    accessible = ui.get("accessibleEnvironmentObjects", [])

    def _add_chemical_counts_from_name(name: str, chemical_dict: Dict[str, int]) -> None:
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
        if any(obj.get("name") == key for key in [RUSTED_KEY, RUSTED_KEY_2, RUSTED_KEY_3, KEY_NO_RUST]):
            description = obj.get("description", "")
            if "in jar" in description:
                is_key_in_jar = True
                break

    inv_objects = {obj.get("name") for obj in inventory or []}
    has_key = any(obj_name and any(obj_name == key or obj_name.startswith(key) for key in [RUSTED_KEY, RUSTED_KEY_2, RUSTED_KEY_3, KEY_NO_RUST]) for obj_name in inv_objects)
    has_jar = any(obj_name == JAR or (isinstance(obj_name, str) and obj_name.startswith("jar")) for obj_name in inv_objects)
    
    chemical_dict = {"A": 0, "B": 0, "C": 0, "D": 0}
    for name in inv_objects:
        _add_chemical_counts_from_name(name or "", chemical_dict)

    return has_key, has_jar, is_key_in_jar, chemical_dict


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
    last_msg = ui_obs.get("lastActionMessage", "")
    extended_msg = ui_obs.get("extended_action_message", "")
    if last_msg:
        lines.append(f"\nLast action message: {last_msg}")
    if extended_msg:
        lines.append(f"Extended info: {extended_msg}")
    
    # 6. Task completion
    task_progress = ui_obs.get("taskProgress", [])[0] if ui_obs.get("taskProgress") else {}
    success = task_progress.get("completed", False)
    lines.append(f"\nTask completed: {success}")
    
    return "\n".join(lines)