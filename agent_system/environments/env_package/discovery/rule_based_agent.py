from agent_system.environments.env_package.discovery.helpers import all_action_abbr
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistryEasySkill

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
        self.is_key_in_jar = False
    
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
                self.is_key_in_jar = True
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
            
            if not inv_objects and location in [(18, 12), (19, 12), (20, 12), (21, 12), (22, 12)]:
                return self.action_space["move_west"]

        elif inv_objects and not accessible_objects:
            
            if RUSTED_KEY in inv_objects and not JAR in inv_objects:
                if location == (17, 12):
                    if facing != "north":
                        return self.action_space["rotate_north"]
                    else:
                        return self.action_space["pickup_jar"]
                elif location in [(18, 12), (19, 12), (20, 12), (21, 12), (22, 12)]:
                    return self.action_space["move_west"]
            
            elif JAR in inv_objects and not RUSTED_KEY in inv_objects and not KEY_NO_RUST in inv_objects:
                if location == (17, 12):
                    return self.action_space["pickup_key"]
                elif location in [(18, 12), (19, 12), (20, 12), (21, 12), (22, 12)]:
                    return self.action_space["move_west"]
            
            elif RUSTED_KEY in inv_objects and JAR in inv_objects and not self.is_key_in_jar:
                self.is_key_in_jar = True
                return self.action_space["put_key"]
            
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
            elif location in [(18, 12), (19, 12), (20, 12), (21, 12), (22, 12)]:
                return self.action_space["move_west"]


class RulebasedAgentSkill:
    def __init__(self, env):
        self.env = env
        self.skill_counter = {}

    def skill(self, skill_name):
        self.skill_counter[skill_name] = self.skill_counter.get(skill_name, 0) + 1
        return skill_name
    
    def select_skill(self, info):
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
        
        if RUSTED_KEY not in inv_objects and KEY_NO_RUST not in inv_objects and location != (17, 12):
            return self.skill("move_to_key")
        
        if location == (17, 12) and RUSTED_KEY in accessible_objects:
            if JAR not in inv_objects and self.env.is_key_in_jar:
                return self.skill("pick_up_jar")
            if not self.env.is_key_in_jar:
                return self.skill("pick_up_key")
        
        if RUSTED_KEY in inv_objects and JAR not in inv_objects:
            if location != (17, 12):
                return self.skill("move_to_jar")
            else:
                return self.skill("pick_up_jar")
        
        if RUSTED_KEY in inv_objects and JAR in inv_objects and not self.env.is_key_in_jar:
            return self.skill("put_key_in_jar")
        
        used_other_dispensers = any(self.env.used_dispensers[x] for x in ["A", "C", "D"])
        
        if self.env.is_key_in_jar and JAR in inv_objects and location[0] != 19 and not used_other_dispensers:
            return self.skill("move_to_dispensers_B")
        
        if self.env.is_key_in_jar and JAR in inv_objects and location[0] == 19 and not self.env.used_dispensers["B"]:
            return self.skill("use_dispenser_B_on_jar")
        
        if self.env.is_key_in_jar and JAR in inv_objects and used_other_dispensers:
            return self.skill("wash_jar")
        
        if KEY_NO_RUST in inv_objects:
            return self.skill("open_door")
    

if __name__ == "__main__":
    from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
    env = DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Easy",
        seed=0,
        max_steps=50,
    )
    agent = RulebasedAgentSkill(env)

    obs, info = env.reset()
    done = False
    while not done:
        action = agent.select_skill(info)
        obs, reward, done, info = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")