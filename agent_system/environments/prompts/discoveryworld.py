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

# --------------------- DiscoveryWorld --------------------- #

DISCOVERYWORLD_TEMPLATE_NO_HIS = """
[GOAL]
You are an expert agent in a room with a rusted key and a locked door.
You need to 
1. Find the key, pick it up and put it in the jar.
2. Take the jar to the dispensers to derust the key.
3. Total amount of chemicals in the jar must reach {chemical_N}.
4. You can remove chemicals from the jar if you have excessive amount of chemicals.
5. Curriculum start state: {curriculum_state}.
6. When the chemical combination matches, the key will be derusted. Open the door and exit.

[STATE]
{step_info}
{state_obs}

[OUTPUT]
Return EXACTLY one of:
move_to_key
pick_up_key
pick_up_jar
put_key_in_jar
use_dispenser_A
use_dispenser_B
use_dispenser_C
use_dispenser_D
remove_chemical_A
remove_chemical_B
remove_chemical_C
remove_chemical_D
wash_jar
open_door

Do not output multiple actions or anything else.
"""


DISCOVERYWORLD_TEMPLATE = """
[GOAL]
You are an expert agent in a room with a rusted key and a locked door.
You need to 
1. Find the key, pick it up and put it in the jar.
2. Take the jar to the dispensers to derust the key.
3. Total amount of chemicals in the jar must reach {chemical_N}.
4. You can remove chemicals from the jar if you have excessive amount of chemicals.
5. Curriculum start state: {curriculum_state}.
6. When the chemical combination matches, the key will be derusted. Open the door and exit.

[STATE]
{step_info}
{state_obs}

[MEMORY]
You have taken the following actions in the 3 past steps:
{memory_actions}
Try some different actions from the past 3 steps. Focus on next subgoals and avoid repeating the same actions.

[OUTPUT]
Return EXACTLY one of:
move_to_key
pick_up_key
pick_up_jar
put_key_in_jar
use_dispenser_A
use_dispenser_B
use_dispenser_C
use_dispenser_D
remove_chemical_A
remove_chemical_B
remove_chemical_C
remove_chemical_D
wash_jar
open_door

Do not output multiple actions or anything else.
"""
