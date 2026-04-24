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
1. Pick up the key and put it in the jar.
2. Use dispensers to derust the key.
3. Open the door and exit.

[STATE]
{step_info}
{state_obs}

[OUTPUT]
Return EXACTLY one of:
move_to_key
move_to_jar
move_to_dispenser_A
move_to_dispenser_B
move_to_dispenser_C
move_to_dispenser_D
pick_up_key
put_key_in_jar
use_dispenser_A
use_dispenser_B
use_dispenser_C
use_dispenser_D
wash_jar
open_door

Do not output anything else.
"""


DISCOVERYWORLD_TEMPLATE = """
[GOAL]
You are an expert agent in a room with a rusted key and a locked door.
You need to 
1. Pick up the key and put it in the jar.
2. Use dispensers to derust the key.
3. Open the door and exit.

[STATE]
{step_info}
{state_obs}

[MEMORY]
{memory_actions}

[OUTPUT]
Return EXACTLY one of:
move_to_key
move_to_jar
move_to_dispenser_A
move_to_dispenser_B
move_to_dispenser_C
move_to_dispenser_D
pick_up_key
put_key_in_jar
use_dispenser_A
use_dispenser_B
use_dispenser_C
use_dispenser_D
wash_jar
open_door

Do not output anything else.
"""
