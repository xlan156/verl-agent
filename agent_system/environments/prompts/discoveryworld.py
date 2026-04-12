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

Below is the current state of the world from the agent's perspective:
{ui_json}

You can now take your action using the following JSON format:
{{"action": action_name, "arg1": item, "arg2": item}}

NOTE: action_name can be one of MOVE_DIRECTION, ROTATE_DIRECTION, PICKUP, DROP, PUT, OPEN, CLOSE, ACTIVATE, DEACTIVATE, TALK, EAT, READ, USE. Action USE and PUT needs two arguments, others only need one.
NOTE: When you want to move or rotate, you can use MOVE_DIRECTION or ROTATE_DIRECTION, and provide the direction (north, east, south, west) as arg1.
NOTE: Only accessible objects can be interacted with (pickup, use, put, etc.). If there is no accessible object, you can only move or rotate.

Now it's your turn to choose the next action.

1. First, think step by step about the current state, task, and which action will be required to reach some intermediate goals and then final goal. This reasoning MUST be enclosed within <think> </think> tags.
   
2. After finishing your reasoning, output a single action inside <action> </action>
   tags. Do not output multiple actions.
"""


DISCOVERYWORLD_TEMPLATE = """
You are an expert autonomous agent operating in the DiscoveryWorld environment.

Your task is:
{task_description}

You are now at step {current_step}.
Below is the current state of the world from the agent's perspective:
{ui_json}

So far you have taken a total of {step_count} step(s). Below are the most recent
{history_length} observation–action pairs that summarize your interaction history:

{action_history}

You can now take your action using the following JSON format:
{{"action": action_name, "arg1": item, "arg2": item}}

NOTE: action_name can be one of MOVE_DIRECTION, ROTATE_DIRECTION, PICKUP, DROP, PUT, OPEN, CLOSE, ACTIVATE, DEACTIVATE, TALK, EAT, READ, USE. Action USE and PUT needs two arguments, others only need one.
NOTE: When you want to move or rotate, you can use MOVE_DIRECTION or ROTATE_DIRECTION, and provide the direction (north, east, south, west) as arg1.
NOTE: Only accessible objects can be interacted with (pickup, use, put, etc.). If there is no accessible object, you can only move or rotate.

Now it's your turn to choose the next action.

1. First, think step by step about the current state, task, and which action will be required to reach some intermediate goals and then final goal. This reasoning MUST be enclosed within <think> </think> tags.
   
2. After finishing your reasoning, output a single action inside <action> </action>
   tags. Do not output multiple actions.
"""
