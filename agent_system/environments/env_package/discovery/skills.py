import random

from agent_system.environments.env_package.discovery.helpers import all_action_abbr

DISPENSER_NAMES = ["Dispenser (Substance A)", "Dispenser (Substance B)", "Dispenser (Substance C)", "Dispenser (Substance D)"]
RUSTED_KEY = "rusted key (heavily rusted)"
KEY_NO_RUST = "key (no rust)"
JAR = "jar"
DOOR = "door"
TABLE = "table"
OTHER_OBJECTS = ["wall", "floor", "path", "grass"]

class CombinatorialChemistryEasySkill():
    def __init__(self, env):
        self.env = env
        self.ui = self.env._api.getAgentObservation(agentIdx=0).get("ui")
        self.location = (self.ui.get("agentLocation").get("x"), self.ui.get("agentLocation").get("y"))
        self.action_space = all_action_abbr
        self.skill_mapping = {
            "move_to_key": self.move_to_key,
            "move_to_jar": self.move_to_jar,
            "pick_up_key": self.pick_up_key,
            "put_key_in_jar": self.put_key_in_jar,
            "pick_up_jar": self.pick_up_jar,
            "wash_jar": self.wash_jar,
            "open_door": self.open_door,
        }
        
        for i in ["A", "B", "C", "D"]:
            self.skill_mapping[f"move_to_dispenser_{i}"] = (
                lambda d=i: self.move_to_dispenser(d)
            )
            self.skill_mapping[f"use_dispenser_{i}_on_jar"] = (
                lambda d=i: self.use_dispenser_on_jar(d)
            )
        self.skill_names = list(self.skill_mapping.keys())
    
    def update_ui_and_location(self):
        self.ui = self.env._api.getAgentObservation(agentIdx=0).get("ui")
        self.location = (self.ui.get("agentLocation").get("x"), self.ui.get("agentLocation").get("y"))

    def sample_random_skill(self):
        return random.choice(self.skill_names)
    
    def perform_action(self, action):
        result = self.env._api.performAgentAction(agentIdx=0, actionJSON=action)
        self.env._last_action_result = result
        self.env._api.tick()
        self.update_ui_and_location()
        
    def get_inv_objects(self):
        inventory = self.ui.get("inventoryObjects", [])
        inv_objects = [obj.get("name") for obj in inventory]
        return inv_objects
    
    def get_accessible_objects(self):
        accessible = self.ui.get("accessibleEnvironmentObjects", [])
        accessible_objects = [obj.get("name") for obj in accessible]
        return accessible_objects
        
    def move_to_key(self):
        while self.location[1] == 12 and self.location[0] > 17:
            self.perform_action(self.action_space["move_west"])
        if self.ui.get("agentLocation").get("faceDirection") != "north":
            self.perform_action(self.action_space["rotate_north"])
    
    def move_to_jar(self):
        while self.location[1] == 12 and self.location[0] > 17:
            self.perform_action(self.action_space["move_west"])

        while self.ui.get("agentLocation").get("faceDirection") != "north":
            self.perform_action(self.action_space["rotate_north"])
    
    def move_to_dispenser(self, dispenser_id):
        id_to_location = {
            "A": (18, 12),
            "B": (19, 12),
            "C": (20, 12),
            "D": (21, 12),
        }
        target_location = id_to_location.get(dispenser_id)
        while self.location[1] == 12 and self.location[0] < target_location[0]:
            self.perform_action(self.action_space["move_east"])
        
        while self.location[1] == 12 and self.location[0] > target_location[0]:
            self.perform_action(self.action_space["move_west"])
        
        while self.ui.get("agentLocation").get("faceDirection") != "north":
            self.perform_action(self.action_space["rotate_north"])
    
    def pick_up_key(self):
        self.perform_action(self.action_space["pickup_key"])

    def put_key_in_jar(self):
        if self.ui.get("agentLocation").get("faceDirection") != "north":
            self.perform_action(self.action_space["rotate_north"])
        self.perform_action(self.action_space["put_key"])
             
    def pick_up_jar(self):
        if self.ui.get("agentLocation").get("faceDirection") != "north":
            self.perform_action(self.action_space["rotate_north"])
        self.perform_action(self.action_space["pickup_jar"])

    def use_dispenser_on_jar(self, dispenser_id):
        dispenser_action_map = {
            "A": "use_dispenser_A",
            "B": "use_dispenser_B",
            "C": "use_dispenser_C",
            "D": "use_dispenser_D",
        }
        action_key = dispenser_action_map.get(dispenser_id)
        id_to_location = {
            "A": (18, 12),
            "B": (19, 12),
            "C": (20, 12),
            "D": (21, 12),
        }
        target_location = id_to_location.get(dispenser_id)
        if self.location[0] == target_location[0]:
            self.env.used_dispensers[dispenser_id] = True
            self.perform_action(self.action_space[action_key])
        else:
            self.move_to_dispenser(dispenser_id)
            self.env.used_dispensers[dispenser_id] = True
            self.perform_action(self.action_space[action_key])
            
    
    def wash_jar(self):
        while self.location[0] < 22 and self.location[1] == 12:
            self.perform_action(self.action_space["move_east"])
        
        while self.location[0] > 22 and self.location[1] == 12:
            self.perform_action(self.action_space["move_west"])
        
        self.perform_action(self.action_space["rotate_north"])
        self.perform_action(self.action_space["wash"])
        self.env.used_dispensers["A"] = False
        self.env.used_dispensers["B"] = False
        self.env.used_dispensers["C"] = False
        self.env.used_dispensers["D"] = False
    
    def open_door(self):
        while self.location[1] == 12 and self.location[0] < 20:
            self.perform_action(self.action_space["move_east"])
        
        while self.location[1] == 12 and self.location[0] > 20:
            self.perform_action(self.action_space["move_west"])
        
        self.perform_action(self.action_space["rotate_south"])
        self.perform_action(self.action_space["open_door"])
        self.perform_action(self.action_space["move_south"])
        self.perform_action(self.action_space["move_south"])
    
    
        
        
        