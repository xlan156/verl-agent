REACTOR_TEMPLATE_NO_HIS = """
[TASK]
Solve Reactor Lab (Normal): measure crystals with observable instruments, infer whether the relationship is linear or quadratic, calculate target frequencies yourself, then tune and activate reactors 3 and 4. Output the calculated frequency in set_reactor_<index>_frequency_<Hz>. Do not assume hidden simulator state.

[STATE]
{step_info}
{state}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason explaining the chosen action
</think>
<action>
one valid skill name
</action>
"""


REACTOR_TEMPLATE = """
[TASK]
Solve Reactor Lab (Normal): measure crystals with observable instruments, infer whether the relationship is linear or quadratic, calculate target frequencies yourself, then tune and activate reactors 3 and 4. Output the calculated frequency in set_reactor_<index>_frequency_<Hz>. Do not assume hidden simulator state.

[STATE]
{step_info}
{state}

[RECENT ACTIONS]
{memory_actions}

[VALID SKILLS]
{valid_skills}

[RESPONSE]
Return exactly this format and no other text:
<think>
one short state-based reason explaining the chosen action
</think>
<action>
one valid skill name
</action>
"""
