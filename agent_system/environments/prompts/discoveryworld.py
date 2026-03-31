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
You are an expert autonomous agent operating in the DiscoveryWorld environment.

Your task is:
{task_description}

Below is the current UI state of the world, from the agent's perspective:
{ui_json}

You can issue actions in this environment by returning a JSON dictionary with keys like
"action", "arg1", "arg2", etc.

The list of available actions and their arguments is:

PICKUP(arg1: object): pick up an object.
DROP(arg1: object): drop an object from inventory.
PUT(arg1: object, arg2: target) : put object on/in another object, or give it to another agent.
OPEN(arg1: object) : open a container/object.
CLOSE(arg1: object) : close a container/object.
ACTIVATE(arg1: object) : turn on / activate an object.
DEACTIVATE(arg1: object) : turn off / deactivate an object.
TALK(arg1: agent) : talk to another agent.
EAT(arg1: object) : eat / consume an object.
READ(arg1: object) : read an object (e.g., document, sign).
USE(arg1: tool, arg2: target) : use a tool/object on a target object.
MOVE_DIRECTION(arg1: north|east|south|west) : move one step in the given direction.
ROTATE_DIRECTION(arg1: north|east|south|west) : rotate to face the given direction.
TELEPORT_TO_LOCATION(arg1: location_name) : teleport to a named location (see list of valid locations).
TELEPORT_TO_OBJECT(arg1: object_uuid) : teleport next to a specific object by UUID.
DISCOVERY_FEED_GET_UPDATES() : read the latest status updates from the discovery feed.
DISCOVERY_FEED_GET_POST_BY_ID(arg1: post_id:int) : read a specific feed post by integer ID.

Teleporting is supported via the TELEPORT_TO_LOCATION action. Valid teleport locations (arg1)
for the current scenario are:
{teleport_locations}

The result of your last action (if any) is:
{last_action_result}

Now it's your turn to choose the next action.


1. First, think carefully step-by-step about the current state, task, and which action
   best advances towards completing all tasks. This reasoning MUST be enclosed within
   <think> </think> tags.
2. After finishing your reasoning, output a single JSON action inside <action> </action>
   tags. Do not output natural language instructions or multiple actions.

For example:

<think> ... your reasoning here ... </think>
<action>{{"action": "MOVE_DIRECTION", "arg1": "north"}}</action>
"""


DISCOVERYWORLD_TEMPLATE = """
You are an expert autonomous agent operating in the DiscoveryWorld environment.

Your task is:
{task_description}

So far you have taken a total of {step_count} step(s). Below are the most recent
{history_length} observation–action pairs that summarize your interaction history:

{action_history}

You are now at step {current_step}. Below is the current UI state of the world, from
the agent's perspective, as a JSON structure:
```json
{ui_json}
```

The list of available actions and their arguments is:
PICKUP(arg1: object): pick up an object.
DROP(arg1: object): drop an object from inventory.
PUT(arg1: object, arg2: target) : put object on/in another object, or give it to another agent.
OPEN(arg1: object) : open a container/object.
CLOSE(arg1: object) : close a container/object.
ACTIVATE(arg1: object) : turn on / activate an object.
DEACTIVATE(arg1: object) : turn off / deactivate an object.
TALK(arg1: agent) : talk to another agent.
EAT(arg1: object) : eat / consume an object.
READ(arg1: object) : read an object (e.g., document, sign).
USE(arg1: tool, arg2: target) : use a tool/object on a target object.
MOVE_DIRECTION(arg1: north|east|south|west) : move one step in the given direction.
ROTATE_DIRECTION(arg1: north|east|south|west) : rotate to face the given direction.
TELEPORT_TO_LOCATION(arg1: location_name) : teleport to a named location (see list of valid locations).
TELEPORT_TO_OBJECT(arg1: object_uuid) : teleport next to a specific object by UUID.
DISCOVERY_FEED_GET_UPDATES() : read the latest status updates from the discovery feed.
DISCOVERY_FEED_GET_POST_BY_ID(arg1: post_id:int) : read a specific feed post by integer ID.

Teleporting is supported via the TELEPORT_TO_LOCATION action. Valid teleport locations
for the current scenario are:
```json
{teleport_locations}
```

The result of your last action (if any) is:
```json
{last_action_result}
```

Now it's your turn to choose the next action.

1. First, think carefully step-by-step about the current state, your past actions,
   and which action best advances towards completing all tasks. This reasoning MUST
   be enclosed within <think> </think> tags.
2. After finishing your reasoning, output a single JSON action inside <action> </action>
   tags. Do not output natural language instructions or multiple actions.

For example:

<think> ... your reasoning here ... </think>
<action>{{"action": "USE", "arg1": 5, "arg2": 12}}</action>
"""
