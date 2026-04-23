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

[HINT]
In this task, try one chemical at a time:
1. Move to the key, pick it up, and put it in the jar
2. Move to a dispenser, use it on the jar, and check the result
3. If the key is still rusted, clean the jar and try another chemical
4. Once the key is derusted, move to the door and open it.

[STATE]
{step_info}
{state_obs}

[VALID ACTIONS]
MOVE_DIRECTION: {{east, west, north, south}}
ROTATE_DIRECTION: {{east, west, north, south}}
PICKUP: {{key, jar}}
PUT: (key -> jar)
USE: (Dispenser A | B | C | D -> jar), (Bottle Cleaner -> jar)
OPEN: door

[OUTPUT FORMAT]
Select the best action using the format:
<action> {{"action": "...", "arg1": "...", "arg2": "..."}} </action>
"""


DISCOVERYWORLD_TEMPLATE = """
[GOAL]
You are an expert agent in a room with a rusted key and a locked door.
You need to 
1. Pick up the key and put it in the jar.
2. Use dispensers to derust the key.
3. Open the door and exit.

[HINT]
In this task, try one chemical at a time:
1. Move to the key, pick it up, and put it in the jar
2. Move to a dispenser, use it on the jar, and check the result
3. If the key is still rusted, clean the jar and try another chemical
4. Once the key is derusted, move to the door and open it.

[STATE]
{step_info}
{state_obs}

[MEMORY]
{memory_context}

[VALID ACTIONS]
MOVE_DIRECTION: {{east, west, north, south}}
ROTATE_DIRECTION: {{east, west, north, south}}
PICKUP: {{key, jar}}
PUT: (key -> jar)
USE: (Dispenser A | B | C | D -> jar), (Bottle Cleaner -> jar)
OPEN: door

[OUTPUT FORMAT]
Select the best action using the format:
<action> {{"action": "...", "arg1": "...", "arg2": "..."}} </action>
"""
