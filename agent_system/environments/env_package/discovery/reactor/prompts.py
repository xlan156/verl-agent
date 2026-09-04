REACTOR_TEMPLATE_NO_HIS = """
[TASK]
{task} Output the calculated integer frequency in set_reactor_<index>_frequency_<Hz>. Do not invent a skill name.

[STATE]
{step_info}
{state}

[VALID SKILLS]
{valid_skills}

Choose exactly one action from the list above. If the list contains only one skill, choose that skill.

[RESPONSE]
Return exactly this format and no other text. Use the literal short reason `ok`; never explain calculations:
<think>
ok
</think>
<action>
one valid skill name
</action>
"""


REACTOR_TEMPLATE = """
[TASK]
{task} Output the calculated integer frequency in set_reactor_<index>_frequency_<Hz>. Do not invent a skill name.

[STATE]
{step_info}
{state}

[RECENT ACTIONS]
{memory_actions}

[VALID SKILLS]
{valid_skills}

Choose exactly one action from the list above. If the list contains only one skill, choose that skill.

[RESPONSE]
Return exactly this format and no other text. Use the literal short reason `ok`; never explain calculations:
<think>
ok
</think>
<action>
one valid skill name
</action>
"""
