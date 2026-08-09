def format_empty_chemical_belief():
    return "No chemical experiment has been observed yet."


DISCOVERYWORLD_TEMPLATE_NO_HIS = """
[TASK]
You need to find the rusted key, use a jar to apply chemicals to derust it, and then open the door.
There are 4 types of chemicals: A, B, C, D. A hidden target chemical combination is required to derust the key.
You need to infer the correct combination of chemicals, by analyzing whether the rust level reduces from current chemical combination.
Use the dispenser to add chemicals to the jar, and remove chemicals if necessary.
When key is no rust, you can choose to open the door.

[STATE]
{state_obs}

[CHEMICAL MEMORY]
{chemical_belief}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason explaining the chosen action; do not merely restate the task
</think>
<action>
one valid skill name
</action>
"""


DISCOVERYWORLD_TEMPLATE = """
[TASK]
You need to find the rusted key, use a jar to apply chemicals to derust it, and then open the door.
There are 4 types of chemicals: A, B, C, D. A hidden target chemical combination is required to derust the key.
You need to infer the correct combination of chemicals, by analyzing whether the rust level reduces from current chemical combination.
Use the dispenser to add chemicals to the jar, and remove chemicals if necessary.
When key is no rust, you can choose to open the door.

[STATE]
{state_obs}

[RECENT ACTIONS]
{memory_actions}

[CHEMICAL MEMORY]
{chemical_belief}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason explaining the chosen action; do not merely restate the task
</think>
<action>
one valid skill name
</action>
"""
