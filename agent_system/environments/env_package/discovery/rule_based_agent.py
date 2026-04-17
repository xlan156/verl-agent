from agent_system.environments.env_package.discovery.actions import all_action_abbr

DISPENSER_NAMES = ["Dispenser (Substance A)", "Dispenser (Substance B)", "Dispenser (Substance C)", "Dispenser (Substance D)"]
RUSTED_KEY = "rusted key (heavily rusted)"
KEY_NO_RUST = "key (no rust)"
JAR = "jar"
DOOR = "door"
TABLE = "table"
OTHER_OBJECTS = ["wall", "floor", "path", "grass"]

class RulebasedAgent:
    """
    A simple rule-based agent for the Discovery World environment. It uses hardcoded rules to decide which action to take based on the current observation.
    The rules are designed to solve the "Combinatorial Chemistry Easy" scenario.
    """
    
    def __init__(self, env):
        self.env = env
        self.action_space = all_action_abbr
        self.door_opened = False
    
    def select_action(self, info):

        ui = (info.get("raw_observation") or {}).get("ui", {})
        inventory = ui.get("inventoryObjects", [])
        accessible = ui.get("accessibleEnvironmentObjects", [])
        
        inv_objects = {}
        if inventory:
            inv_objects = {obj.get("name"): obj for obj in inventory if obj.get("name") not in OTHER_OBJECTS}
        
        accessible_objects = {}
        if accessible:
            accessible_objects = {obj.get("name"): obj for obj in accessible if obj.get("name") not in OTHER_OBJECTS}
        
        location = (ui.get("agentLocation").get("x"), ui.get("agentLocation").get("y"))
        facing = ui.get("agentLocation").get("faceDirection")
        
        if accessible_objects:
            if RUSTED_KEY in accessible_objects and not JAR in accessible_objects:
                return self.action_space["pickup_key"]
            
            if JAR in accessible_objects and RUSTED_KEY in inv_objects:
                return self.action_space["put_key"]
            
            if KEY_NO_RUST in accessible_objects:
                return self.action_space["pickup_key"]
            
            if TABLE in accessible_objects and RUSTED_KEY in accessible_objects and JAR in accessible_objects:
                return self.action_space["pickup_jar"]
            
            if RUSTED_KEY in inv_objects and JAR in inv_objects and not DISPENSER_NAMES[1] in accessible_objects:
                return self.action_space["move_east"]
            
            if DISPENSER_NAMES[1] in accessible_objects and JAR in inv_objects and RUSTED_KEY in inv_objects:
                return self.action_space["use_dispenser_B"]
            
            if DISPENSER_NAMES[1] in accessible_objects and KEY_NO_RUST in inv_objects:
                return self.action_space["move_east"]
            
            if KEY_NO_RUST in inv_objects and facing == "south" and not self.door_opened:
                self.door_opened = True
                return self.action_space["open_door"]
            
            if KEY_NO_RUST in inv_objects and facing == "south" and self.door_opened:
                return self.action_space["move_south"]
            
            if KEY_NO_RUST in inv_objects and location == (21, 12):
                return self.action_space["move_south"]

        elif inv_objects and not accessible_objects:
            
            if RUSTED_KEY in inv_objects and not JAR in inv_objects and location == (18, 12):
                return self.action_space["move_west"]
            
            elif RUSTED_KEY in inv_objects and location == (17, 12) and not facing == "north":
                return self.action_space["rotate_north"]
            
            elif RUSTED_KEY in inv_objects and JAR in inv_objects:
                if location == (18, 12):
                    return self.action_space["move_east"]
                elif location == (19, 12):
                    return self.action_space["rotate_north"]
                
            elif KEY_NO_RUST in inv_objects and location == (20, 12):
                if facing != "south":
                    return self.action_space["rotate_south"]

            
        elif not inv_objects and not accessible_objects:
            if location == (17, 12):
                return self.action_space["rotate_north"]
            elif location == (18, 12):
                return self.action_space["move_west"]
            elif location == (19, 12):
                return self.action_space["move_west"]
            elif location == (20, 12):
                return self.action_space["move_west"]
            elif location == (21, 12):
                return self.action_space["move_west"]
