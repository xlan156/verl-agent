from types import SimpleNamespace

from agent_system.environments.env_package.discovery.rule_based_agent import (
    RulebasedAgentSkill,
)


def make_agent(seed=0):
    return RulebasedAgentSkill(
        SimpleNamespace(_seed=seed, _max_chemical_n=3)
    )


def chemical_info(combination, signal):
    return {
        "chemical_dict": dict(zip(("A", "B", "C", "D"), combination)),
        "current_reaction_signal": signal,
        "is_key_in_jar": True,
        "max_chemical_n": 3,
    }


def test_teacher_updates_belief_from_visible_reaction_signal():
    agent = make_agent()

    targets = agent.update_chemical_belief(
        chemical_info((2, 1, 0, 0), "successful")
    )

    assert targets == [(2, 1, 0, 0)]
    assert agent.chemical_evidence[(2, 1, 0, 0)] == ("observed", "successful")


def test_teacher_removes_chemical_that_moves_toward_resolved_target():
    agent = make_agent()
    agent.chemical_evidence[(2, 1, 0, 0)] = ("observed", "successful")

    action = agent.select_use_or_remove(
        chemical_info((1, 1, 1, 0), "strong")
    )

    assert action == "remove_chemical_C"


def test_teacher_uses_seed_only_as_deterministic_tiebreaker():
    actions = []
    for seed in range(4):
        agent = make_agent(seed)
        actions.append(
            agent.select_use_or_remove(chemical_info((0, 0, 0, 0), "not tested"))
        )

    assert actions == [
        "use_dispenser_A_on_jar",
        "use_dispenser_B_on_jar",
        "use_dispenser_C_on_jar",
        "use_dispenser_D_on_jar",
    ]
