from agent_system.environments.env_package.discovery.utils import *


class RulebasedAgent:
    """
    A simple rule-based agent for the Discovery World environment. It uses hardcoded rules to decide which action to take based on the current observation.
    The rules are designed to solve the "Combinatorial Chemistry Easy" scenario.
    """
    
    def __init__(self, env):
        self.env = env
        self.seed = env._seed
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
        self.observed_combinations = set()
        self.previous_combination = None
        
    def skill(self, skill_name):
        self.skill_counter[skill_name] = self.skill_counter.get(skill_name, 0) + 1
        return skill_name

    @staticmethod
    def get_curr_combination(info):
        chemical_dict = info.get("chemical_dict", {}) or {}
        return tuple(int(chemical_dict.get(chemical, 0) or 0) for chemical in ("A", "B", "C", "D"))

    def select_use_or_remove(self, info):
        """Choose a deterministic, previously untested one-step combination."""
        current = self.get_curr_combination(info)
        self.observed_combinations.add(current)
        max_amount = int(self.env._max_chemical_n)
        total = sum(current)
        chemicals = ("A", "B", "C", "D")
        offset = int(getattr(self.env, "_seed", 0)) % len(chemicals)
        chemical_order = chemicals[offset:] + chemicals[:offset]

        candidates = []
        if total < max_amount:
            for chemical in chemical_order:
                index = chemicals.index(chemical)
                next_combo = list(current)
                next_combo[index] += 1
                candidates.append((f"use_dispenser_{chemical}_on_jar", tuple(next_combo)))
        else:
            for chemical in chemical_order:
                index = chemicals.index(chemical)
                if current[index] <= 0:
                    continue
                next_combo = list(current)
                next_combo[index] -= 1
                candidates.append((f"remove_chemical_{chemical}", tuple(next_combo)))

        # Prefer an unobserved state and never immediately undo the preceding
        # transition when another unexplored option exists.
        unexplored = [
            candidate
            for candidate in candidates
            if candidate[1] not in self.observed_combinations
            and candidate[1] != self.previous_combination
        ]
        if not unexplored:
            unexplored = [
                candidate
                for candidate in candidates
                if candidate[1] not in self.observed_combinations
            ]
        selected = (unexplored or candidates)[0] if candidates else None
        self.previous_combination = current
        return self.skill(selected[0]) if selected else None
    
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
        rusted_key_in_hand = any(key in inv_objects for key in [RUSTED_KEY, RUSTED_KEY_2, RUSTED_KEY_3])
        rusted_key_accessible = any(key in accessible_objects for key in [RUSTED_KEY, RUSTED_KEY_2, RUSTED_KEY_3])
        clean_key_in_hand = KEY_NO_RUST in inv_objects
        clean_key_accessible = KEY_NO_RUST in accessible_objects
        
        is_key_in_jar = info.get("is_key_in_jar", False)
        if not rusted_key_in_hand and not clean_key_in_hand and location != (17, 12):
            return self.skill("move_to_key")
        
        if location == (17, 12) and rusted_key_accessible:
            if JAR not in inv_objects and is_key_in_jar:
                return self.skill("pick_up_jar")
            if not is_key_in_jar:
                return self.skill("pick_up_key")
        
        if rusted_key_in_hand and JAR not in inv_objects:
            if location != (17, 12):
                return self.skill("move_to_jar")
            else:
                return self.skill("pick_up_jar")
        
        if rusted_key_in_hand and JAR in inv_objects and not is_key_in_jar:
            return self.skill("put_key_in_jar")
        
        if clean_key_in_hand or info.get("key_rust_status") == "no rust":
            return self.skill("open_door")

        if is_key_in_jar and rusted_key_in_hand and JAR in inv_objects:
            return self.select_use_or_remove(info)
        
        return None
    

if __name__ == "__main__":
    from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
    env = DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Challenge",
        seed=17,
        max_steps=70,
        max_chemical_n=3,
    )
    agent = RulebasedAgentSkill(env)

    obs, info = env.reset()
    done = False
    while not done:
        action = agent.select_skill(info)
        obs, reward, done, info = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")
