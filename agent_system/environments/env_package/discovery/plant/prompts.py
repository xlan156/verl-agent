PLANT_TEMPLATE_NO_HIS = """
[TASK]
Discover the nutrient rule for Plant Nutrients (Normal) through observable plot experiments. Nutrient levels use 1=low, 2=medium, and 3=high. Once exactly one rule candidate remains, configure field 1 to that nutrient and level, commit it, plant two seeds in field 1, then wait for growth. Do not assume hidden simulator state.

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
Discover the nutrient rule for Plant Nutrients (Normal) through observable plot experiments. Nutrient levels use 1=low, 2=medium, and 3=high. Once exactly one rule candidate remains, configure field 1 to that nutrient and level, commit it, plant two seeds in field 1, then wait for growth. Do not assume hidden simulator state.

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
