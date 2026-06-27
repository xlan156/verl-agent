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

[OUTPUT FORMAT — STRICT]
Your entire reply must contain exactly two XML-style blocks, in this exact
order: `<think>` followed by `<action>`. The first character of your reply
must be `<` in `<think>`, and the final characters must be `</action>`.

Use this exact shape (replace the example text with your own reasoning and
chosen action):
<think>
The key has not been collected yet, so I should move to it first.
</think>
<action>
move_to_key
</action>

Formatting rules:
- Put exactly one short, state-grounded sentence (at most 20 words) only
  inside `<think>...</think>`; do not make a plan, list, or add action names.
- Put exactly one skill name only inside `<action>...</action>`.
- Do not use Markdown, code fences, labels, bullets, extra tags, or any text
  before, between, or after these two blocks.
- Never place the action in `<think>` or the reasoning in `<action>`.

`one_skill_name` must be EXACTLY one of:
move_to_key
move_to_jar
pick_up_key
pick_up_jar
put_key_in_jar
use_dispenser_A_on_jar
use_dispenser_B_on_jar
use_dispenser_C_on_jar
use_dispenser_D_on_jar
remove_chemical_A
remove_chemical_B
remove_chemical_C
remove_chemical_D
wash_jar
open_door

Before replying, check that the reply begins with `<think>` and ends with
`</action>`. Begin now with `<think>`.
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

[OUTPUT FORMAT — STRICT]
Your entire reply must contain exactly two XML-style blocks, in this exact
order: `<think>` followed by `<action>`. The first character of your reply
must be `<` in `<think>`, and the final characters must be `</action>`.

Use this exact shape (replace the example text with your own reasoning and
chosen action):
<think>
The key has not been collected yet, so I should move to it first.
</think>
<action>
move_to_key
</action>

Formatting rules:
- Put exactly one short, state-grounded sentence (at most 20 words) only
  inside `<think>...</think>`; do not make a plan, list, or add action names.
- Put exactly one skill name only inside `<action>...</action>`.
- Do not use Markdown, code fences, labels, bullets, extra tags, or any text
  before, between, or after these two blocks.
- Never place the action in `<think>` or the reasoning in `<action>`.

`one_skill_name` must be EXACTLY one of:
move_to_key
move_to_jar
pick_up_key
pick_up_jar
put_key_in_jar
use_dispenser_A_on_jar
use_dispenser_B_on_jar
use_dispenser_C_on_jar
use_dispenser_D_on_jar
remove_chemical_A
remove_chemical_B
remove_chemical_C
remove_chemical_D
wash_jar
open_door

Before replying, check that the reply begins with `<think>` and ends with
`</action>`. Begin now with `<think>`.
"""
