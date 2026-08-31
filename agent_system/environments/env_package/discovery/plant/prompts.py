PLANT_TEMPLATE_NO_HIS = """
[TASK]
{task_instruction}

[DECISION RULE]
{decision_rule}

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


PLANT_TEMPLATE = """
[TASK]
{task_instruction}

[DECISION RULE]
{decision_rule}

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
