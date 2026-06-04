import random
import re
from typing import Any, Tuple

from agent_system.environments.env_package.discovery.curriculum import (
    CHEMICAL_ORDER,
    normalize_chemical_state,
)
from agent_system.environments.env_package.discovery.utils import *

class CombinatorialChemistrySkill():
    def __init__(self, env):
        self.env = env
        self.ui = self.env._api.getAgentObservation(agentIdx=0).get("ui")
        self.location = (self.ui.get("agentLocation").get("x"), self.ui.get("agentLocation").get("y"))
        self.action_space = all_action_abbr
        self.chemical_dict = {"A": 0, "B": 0, "C": 0, "D": 0}
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
                lambda x=i: self.move_to_dispenser(x)
            )
            self.skill_mapping[f"use_dispenser_{i}_on_jar"] = (
                lambda x=i: self.use_dispenser_on_jar(x)
            )
            self.skill_mapping[f"remove_chemical_{i}"] = (
                lambda x=i: self.remove_one_chemical(x)
            )
        self.skill_names = list(self.skill_mapping.keys())

    def _ensure_key_and_jar_ready(self):
        self.update_ui_and_location()
        has_key, has_jar, is_key_in_jar, _ = extract_detailed_status(self.ui)

        if not has_key and not is_key_in_jar:
            self.move_to_key()
            self.pick_up_key()
            self.update_ui_and_location()
            has_key, has_jar, is_key_in_jar, _ = extract_detailed_status(self.ui)

        if has_key and not has_jar:
            self.move_to_jar()
            self.pick_up_jar()
            self.update_ui_and_location()
            has_key, has_jar, is_key_in_jar, _ = extract_detailed_status(self.ui)

        if has_key and has_jar and not is_key_in_jar:
            self.put_key_in_jar()
            self.update_ui_and_location()

    def prepare_chemical_state(self, target_state: Any, rebuild_from_scratch: bool = True) -> Tuple[int, ...]:
        """Prepare a curriculum start state before the agent begins acting."""
        target = normalize_chemical_state(target_state, num_chemicals=len(CHEMICAL_ORDER))

        self._ensure_key_and_jar_ready()
        if rebuild_from_scratch:
            self.wash_jar()
            self._ensure_key_and_jar_ready()

        for chemical_name, count in zip(CHEMICAL_ORDER, target):
            for _ in range(count):
                self.use_dispenser_on_jar(chemical_name)

        self.update_ui_and_location()
        return target
    
    def update_ui_and_location(self):
        self.ui = self.env._api.getAgentObservation(agentIdx=0).get("ui")
        self.location = (self.ui.get("agentLocation").get("x"), self.ui.get("agentLocation").get("y"))
        _, _, _, self.chemical_dict = extract_detailed_status(self.ui)

    def sample_random_skill(self):
        return random.choice(self.skill_names)
    
    def perform_action(self, action):
        result = self.env._api.performAgentAction(agentIdx=0, actionJSON=action)
        self.env._last_action_result = result
        self.env._api.tick()
        self.update_ui_and_location()
        
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
       
    def remove_one_chemical(self, chemical):    
        target_dict = dict(self.chemical_dict)
        if target_dict.get(chemical, 0) == 0:
            return
        target_dict[chemical] = target_dict[chemical] - 1
        self.wash_jar()
        for chem_id, count in target_dict.items():
            for _ in range(count):
                self.use_dispenser_on_jar(chem_id)
        
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


if __name__ == "__main__":
    from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
    env = DiscoveryWorldEnv(scenario_name="Combinatorial Chemistry", difficulty="Challenge", seed=13, max_steps=50, max_chemical_n=2)
    _, info = env.reset()
    skill_agent = CombinatorialChemistrySkill(env)
    skill_agent.move_to_key()
    skill_agent.pick_up_key()
    skill_agent.put_key_in_jar()
    skill_agent.pick_up_jar()
    skill_agent.use_dispenser_on_jar("A")
    skill_agent.use_dispenser_on_jar("B")
    skill_agent.use_dispenser_on_jar("C")
    skill_agent.remove_one_chemical("B")
    skill_agent.use_dispenser_on_jar("D")
    
    
        
        
        