def format_current_chemicals(chemical_dict, max_chemical_n):
    chemical_dict = chemical_dict or {}
    chemicals = ["A", "B", "C", "D"]
    counts = {chemical: int(chemical_dict.get(chemical, 0) or 0) for chemical in chemicals}
    total = sum(counts.values())
    current_chemicals = ", ".join(f"{chemical}={counts[chemical]}" for chemical in chemicals)
    return (
        f"Chemical amount in jar: {total} / {max_chemical_n}\n"
        f"Current chemicals: {current_chemicals}"
    )


DISCOVERYWORLD_TEMPLATE_NO_HIS = """
[GOAL]
You are an expert agent in a room with a rusted key and a locked door.
You need to 
1. Find the key, pick it up and put it in the jar.
2. Take the jar to the dispensers to derust the key.
3. Total amount of chemicals in the jar must reach {max_chemical_n}.
4. You can remove chemicals from the jar if you have excessive amount of chemicals.
5. When the chemical combination matches, the key will be derusted. Open the door and exit.

[STATE]
{step_info}
{chemical_state}
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
3. Total amount of chemicals in the jar must reach {max_chemical_n}.
4. You can remove chemicals from the jar if you have excessive amount of chemicals.
5. When the chemical combination matches, the key will be derusted. Open the door and exit.

[STATE]
{step_info}
{chemical_state}
{state_obs}

[MEMORY]
You have taken the following actions in the 3 past steps, along with the key's rust level after each step:
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
