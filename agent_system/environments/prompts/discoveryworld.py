def format_current_chemicals(chemical_dict, max_chemical_n):
    chemical_dict = chemical_dict or {}
    chemicals = ["A", "B", "C", "D"]
    counts = {chemical: int(chemical_dict.get(chemical, 0) or 0) for chemical in chemicals}
    total = sum(counts.values())
    current_chemicals = ", ".join(f"{chemical}={counts[chemical]}" for chemical in chemicals)
    return (
        f"Chemical amount in jar / Required chemical amount: {total} / {max_chemical_n}\n"
        f"Current chemicals: {current_chemicals}"
    )


def format_key_status(key_rust_status):
    rust_status = str(key_rust_status or "unknown").strip().lower()
    ready_to_open = rust_status == "no rust"
    return f"Rust level: {rust_status}\nReady to open: {ready_to_open}"


DISCOVERYWORLD_TEMPLATE_NO_HIS = """
[TASK]
You need to find the rusted key, use a jar to apply chemicals to derust it, and then open the door.
There are 4 types of chemicals: A, B, C, D. A hidden target chemical combination is required to derust the key.
You need to infer the correct combination of chemicals, by analyzing whether the rust level reduces from current chemical combination.
Use the dispenser to add chemicals to the jar, and remove chemicals if necessary.
When key is no rust, you can choose to open the door.

[STATE]
{step_info}
{chemical_state}
{key_state}
{state_obs}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason
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
{step_info}
{chemical_state}
{key_state}
{state_obs}

[RECENT ACTIONS]
{memory_actions}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason
</think>
<action>
one valid skill name
</action>
"""
