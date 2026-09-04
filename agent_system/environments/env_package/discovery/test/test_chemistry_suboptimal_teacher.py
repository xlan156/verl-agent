from __future__ import annotations

import unittest

from agent_system.environments.env_package.discovery.chemistry.teacher import (
    RulebasedAgentSkill,
)
from agent_system.environments.env_package.discovery.chemistry.utils import (
    get_valid_discoveryworld_skills,
)


class _TeacherEnv:
    _seed = 7
    _max_chemical_n = 2

    @property
    def _hidden_chemical_target(self):
        raise AssertionError("The non-oracle teacher accessed the hidden target")


class ChemistrySuboptimalTeacherTest(unittest.TestCase):
    def test_n1_full_jar_retains_remove_and_wash_choices(self):
        skills = get_valid_discoveryworld_skills(
            {
                "has_jar": True,
                "is_key_in_jar": True,
                "key_rust_status": "heavily rusted",
                "chemical_dict": {"A": 1, "B": 0, "C": 0, "D": 0},
            },
            max_chemical_n=1,
        )

        self.assertEqual(skills, ["remove_chemical_A", "wash_jar"])

    def test_n1_lapse_can_choose_wash_instead_of_forced_remove(self):
        class N1Env(_TeacherEnv):
            _max_chemical_n = 1

        teacher = RulebasedAgentSkill(
            N1Env(), suboptimal_probability=1.0, rng_seed=0
        )
        teacher.update_chemical_belief = lambda _info: [(0, 1, 0, 0)]

        action = teacher.select_use_or_remove(
            {"chemical_dict": {"A": 1, "B": 0, "C": 0, "D": 0}}
        )

        self.assertEqual(teacher.last_greedy_skill, "remove_chemical_A")
        self.assertEqual(action, "wash_jar")
        self.assertEqual(teacher.last_selection_mode, "suboptimal")

    def test_teacher_uses_observations_without_hidden_target(self):
        teacher = RulebasedAgentSkill(_TeacherEnv())
        action = teacher.select_use_or_remove(
            {
                "chemical_dict": {"A": 0, "B": 0, "C": 0, "D": 0},
                "is_key_in_jar": True,
                "current_reaction_signal": "not tested",
                "max_chemical_n": 2,
            }
        )
        self.assertIn(action, {
            "use_dispenser_A_on_jar",
            "use_dispenser_B_on_jar",
            "use_dispenser_C_on_jar",
            "use_dispenser_D_on_jar",
        })

    def test_lapse_selects_a_strictly_suboptimal_action(self):
        teacher = RulebasedAgentSkill(_TeacherEnv(), suboptimal_probability=1.0)
        teacher.update_chemical_belief = lambda _info: [(2, 0, 0, 0)]

        action = teacher.select_use_or_remove(
            {"chemical_dict": {"A": 0, "B": 0, "C": 0, "D": 0}}
        )

        self.assertEqual(teacher.last_greedy_skill, "use_dispenser_A_on_jar")
        self.assertNotEqual(action, teacher.last_greedy_skill)
        self.assertEqual(teacher.last_selection_mode, "suboptimal")

    def test_zero_lapse_recovers_the_belief_greedy_teacher(self):
        teacher = RulebasedAgentSkill(_TeacherEnv(), suboptimal_probability=0.0)
        teacher.update_chemical_belief = lambda _info: [(2, 0, 0, 0)]

        action = teacher.select_use_or_remove(
            {"chemical_dict": {"A": 0, "B": 0, "C": 0, "D": 0}}
        )

        self.assertEqual(action, "use_dispenser_A_on_jar")
        self.assertEqual(teacher.last_selection_mode, "greedy")


if __name__ == "__main__":
    unittest.main()
