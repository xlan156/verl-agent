from __future__ import annotations

from typing import Any, Iterable
import re


IGNORED_UI_OBJECTS = {"wall", "floor", "path", "grass", "table", "agent"}


def compress_ui_observation(ui_obs: dict[str, Any]) -> str:
    """Render the task-independent, observable portion of a DiscoveryWorld UI."""
    strip_uuid = lambda text: re.sub(r"\s*\[uuid:\s*[^\]]+\]", "", text or "").strip()
    lines = []
    location = ui_obs.get("agentLocation", {})
    if location:
        lines.append(
            f"Location: ({location.get('x', '?')}, {location.get('y', '?')}), "
            f"facing {location.get('faceDirection', 'unknown')}"
        )
        if location.get("directions_you_can_move"):
            lines.append("Can move: " + ", ".join(location["directions_you_can_move"]))
        if location.get("directions_blocked"):
            lines.append("Blocked: " + ", ".join(location["directions_blocked"]))
    inventory = [
        strip_uuid(obj.get("description", ""))
        for obj in ui_obs.get("inventoryObjects", [])
        if obj.get("name") not in IGNORED_UI_OBJECTS
    ]
    lines.append("Inventory: " + (", ".join(inventory) if inventory else "empty"))
    accessible = [
        strip_uuid(obj.get("description", ""))
        for obj in ui_obs.get("accessibleEnvironmentObjects", [])
        if obj.get("name") not in IGNORED_UI_OBJECTS
    ]
    lines.append(
        "Accessible (facing): "
        + (", ".join(accessible) if accessible else "no object is accessible in current location and facing direction")
    )
    lines.append("Nearby objects:")
    for direction, objects in ui_obs.get("nearbyObjects", {}).get("objects", {}).items():
        for obj in objects:
            distance = obj.get("distance", 99)
            if 0 < distance <= 2 and obj.get("name") not in IGNORED_UI_OBJECTS:
                lines.append(f"- {direction} ({distance} tile(s) away): {strip_uuid(obj.get('description', ''))}")
    progress = ui_obs.get("taskProgress", [])
    lines.append(f"\nTask completed: {bool(progress and progress[0].get('completed', False))}")
    return "\n".join(lines)


class ObservableSkillRunner:
    """Small helper for legal, UUID-based DiscoveryWorld macro actions."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.api = env._api
        self.world = self.api.world
        self.agent = self.world.getUserAgents()[0]
        self.ui = {}
        self.update_ui()

    def update_ui(self) -> dict[str, Any]:
        self.ui = self.api.getAgentObservation(agentIdx=0).get("ui", {})
        return self.ui

    def act(self, action: str, arg1: Any = None, arg2: Any = None) -> bool:
        payload = {"action": action}
        if arg1 is not None:
            payload["arg1"] = getattr(arg1, "uuid", arg1)
        if arg2 is not None:
            payload["arg2"] = getattr(arg2, "uuid", arg2)
        result = self.api.performAgentAction(agentIdx=0, actionJSON=payload)
        self.api.tick()
        self.update_ui()
        message = str(self.ui.get("lastActionMessage") or "")
        self.env._last_action_result = {
            "success": bool(result.get("success", False)),
            "message": message or "; ".join(result.get("errors", [])),
        }
        return bool(result.get("success", False))

    def finish(self, success: bool, message: str) -> None:
        self.env._last_action_result = {"success": bool(success), "message": message}

    def objects(self, *types: str) -> list[Any]:
        wanted = {re.sub(r"[^a-z0-9]", "", value.lower()) for value in types}
        return [
            obj
            for obj in self.world.getAllWorldObjects()
            if re.sub(r"[^a-z0-9]", "", str(getattr(obj, "type", "")).lower()) in wanted
        ]

    @staticmethod
    def sort_objects(objects: Iterable[Any]) -> list[Any]:
        return sorted(
            objects,
            key=lambda obj: (
                str(getattr(obj, "name", "")),
                tuple(obj.getWorldLocation()),
                int(getattr(obj, "uuid", 0)),
            ),
        )

    def teleport(self, obj: Any) -> bool:
        return self.act("TELEPORT_TO_OBJECT", obj)

    def pickup(self, obj: Any) -> bool:
        if obj.parentContainer is self.agent or self._inside_inventory(obj):
            return True
        return self.teleport(obj) and self.act("PICKUP", obj)

    def _inside_inventory(self, obj: Any) -> bool:
        parent = getattr(obj, "parentContainer", None)
        while parent is not None:
            if parent is self.agent:
                return True
            parent = getattr(parent, "parentContainer", None)
        return False

    def choose_dialog(self, text: str) -> bool:
        options = self.dialog_options()
        if text not in options:
            return False
        result = self.api.performAgentAction(
            agentIdx=0,
            actionJSON={"chosen_dialog_option_int": options.index(text) + 1},
        )
        self.api.tick()
        self.update_ui()
        self.env._last_action_result = {
            "success": bool(result.get("success", False)),
            "message": str(self.ui.get("lastActionMessage") or ""),
        }

        return bool(result.get("success", False))
    def dialog_options(self) -> list[str]:
        if not self.agent.isInDialog():
            return []
        partner = self.agent.getAgentInDialogWith()
        _, options = partner.dialogTree.getCurrentDialog()
        return list(options)

    def choose_matching(self, fragment: str) -> bool:
        match = next(
            (option for option in self.dialog_options() if fragment.lower() in option.lower()),
            None,
        )
        return bool(match) and self.choose_dialog(match)

    def talk_to(self, obj: Any) -> bool:
        return self.teleport(obj) and self.act("TALK", obj)
