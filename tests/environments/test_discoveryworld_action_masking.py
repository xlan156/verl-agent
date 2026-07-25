from types import SimpleNamespace

from agent_system.environments.env_package.discovery.projection import (
    discoveryworld_projection,
)
from agent_system.environments.env_package.discovery.skills import (
    CombinatorialChemistrySkill,
)
from agent_system.environments.env_package.discovery.utils import (
    REMOVE_CHEMICAL_A,
    REMOVE_CHEMICAL_B,
    REMOVE_CHEMICAL_C,
    REMOVE_CHEMICAL_D,
    USE_DISPENSER_A,
    USE_DISPENSER_B,
    USE_DISPENSER_C,
    USE_DISPENSER_D,
    get_valid_discoveryworld_skills,
)


USES = [USE_DISPENSER_A, USE_DISPENSER_B, USE_DISPENSER_C, USE_DISPENSER_D]
REMOVES = [
    REMOVE_CHEMICAL_A,
    REMOVE_CHEMICAL_B,
    REMOVE_CHEMICAL_C,
    REMOVE_CHEMICAL_D,
]


def chemical_info(chemical_dict):
    return {
        "has_key": True,
        "has_jar": True,
        "is_key_in_jar": True,
        "key_rust_status": "heavily rusted",
        "chemical_dict": chemical_dict,
        "max_chemical_n": 3,
        "raw_observation": {"ui": {}},
    }


def response(skill):
    return f"<think>Test this phase-valid action.</think><action>{skill}</action>"


def test_chemical_stage_exposes_all_four_use_actions_below_target_amount():
    assert get_valid_discoveryworld_skills(
        chemical_info({"A": 1, "B": 1, "C": 0, "D": 0}),
        max_chemical_n=3,
    ) == USES


def test_chemical_stage_exposes_all_four_remove_actions_at_target_amount():
    assert get_valid_discoveryworld_skills(
        chemical_info({"A": 3, "B": 0, "C": 0, "D": 0}),
        max_chemical_n=3,
    ) == REMOVES


def test_projection_allows_removing_an_absent_chemical_as_learnable_noop():
    info = chemical_info({"A": 3, "B": 0, "C": 0, "D": 0})
    info["valid_skills"] = get_valid_discoveryworld_skills(info, 3)

    actions, valid = discoveryworld_projection(
        [response(REMOVE_CHEMICAL_D)],
        [info],
    )

    assert actions == [REMOVE_CHEMICAL_D]
    assert valid == [1]


def test_projection_still_rejects_skill_from_wrong_physical_phase():
    info = chemical_info({"A": 3, "B": 0, "C": 0, "D": 0})
    info["valid_skills"] = get_valid_discoveryworld_skills(info, 3)

    actions, valid = discoveryworld_projection([response(USE_DISPENSER_A)], [info])

    assert actions == [None]
    assert valid == [0]


def test_removing_absent_chemical_records_failed_noop():
    skill = CombinatorialChemistrySkill.__new__(CombinatorialChemistrySkill)
    skill.chemical_dict = {"A": 3, "B": 0, "C": 0, "D": 0}
    skill.env = SimpleNamespace(
        _last_action_result={"success": True, "message": "stale"}
    )

    skill.remove_one_chemical("B")

    assert skill.chemical_dict == {"A": 3, "B": 0, "C": 0, "D": 0}
    assert skill.env._last_action_result["success"] is False
