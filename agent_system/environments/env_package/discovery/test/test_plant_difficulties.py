from __future__ import annotations

import contextlib
import io
import unittest

from agent_system.environments.env_package.discovery.plant.envs import TaskAdapter
from agent_system.environments.env_package.discovery.plant.teacher import (
    PlantRuleBasedTeacher,
    candidate_matches,
    initial_candidates,
    required_selections,
)
from agent_system.environments.env_package.discovery.runtime.envs import (
    DiscoveryWorldEnv,
)


class PlantDifficultyTest(unittest.TestCase):
    def test_candidate_spaces_cover_all_difficulties(self):
        self.assertEqual(len(initial_candidates("Easy")), 5)
        self.assertEqual(len(initial_candidates("Normal")), 15)
        self.assertEqual(len(initial_candidates("Challenge")), 285)

    def test_every_challenge_candidate_has_a_satisfying_configuration(self):
        for rule in initial_candidates("Challenge"):
            with self.subTest(rule=rule):
                self.assertTrue(candidate_matches(rule, required_selections(rule)))

    def test_easy_answer_state_exposes_all_nutrient_choices(self):
        candidate = initial_candidates("Easy")[0]
        info = {
            "plant_difficulty": "Easy",
            "plant_tools": ["soil nutrient meter"],
            "plant_rule_candidates": [candidate],
            "plant_experiment_memory": [{}, {}],
            "plant_unmeasured_plots": 2,
            "won": False,
        }

        self.assertEqual(
            TaskAdapter.valid_skills(info, env=None),
            [
                "select_potassium",
                "select_titanium",
                "select_lithium",
                "select_thorium",
                "select_barium",
            ],
        )

    def test_challenge_controller_exposes_compound_rule_choices(self):
        rule = {
            "rule_type": "xor",
            "conditions": [
                {"nutrient": "potassium", "level": 3},
                {"nutrient": "thorium", "level": 2},
            ],
        }
        info = {
            "plant_difficulty": "Challenge",
            "plant_tools": ["soil nutrient meter", "shovel", "seed jar"],
            "plant_rule_candidates": [rule],
            "plant_unmeasured_plots": 0,
            "plant_active_field": 1,
            "won": False,
        }

        skills = TaskAdapter.valid_skills(info, env=None)

        self.assertEqual(len(skills), 8)
        self.assertIn("set_potassium_high", skills)
        self.assertIn("set_thorium_medium", skills)
        self.assertIn("commit_field_configuration", skills)
        self.assertIn("cancel_field_configuration", skills)

    def test_teacher_completes_each_difficulty(self):
        for difficulty in ("Easy", "Normal", "Challenge"):
            with self.subTest(difficulty=difficulty):
                env = DiscoveryWorldEnv(
                    seed=0,
                    scenario_name="Plant Nutrients",
                    difficulty=difficulty,
                    max_steps=60,
                )
                saw_policy_choice = False

                with contextlib.redirect_stdout(io.StringIO()):
                    _, info = env.reset()
                    env.teacher.suboptimal_probability = 0.0
                    for _ in range(60):
                        valid_skills = info["valid_skills"]
                        saw_policy_choice |= len(valid_skills) > 1
                        action = env.teacher.select_skill(info)
                        if action is None:
                            break
                        self.assertIn(action, valid_skills)
                        _, _, done, info = env.step(action)
                        self.assertEqual(info["action_status"], "success")
                        if done:
                            break

                self.assertTrue(saw_policy_choice)
                self.assertTrue(info["won"])
                self.assertAlmostEqual(info["score_normalized"], 1.0)

    def test_teacher_uses_observable_candidates_only(self):
        info = {
            "plant_difficulty": "Easy",
            "plant_experiment_memory": [
                {
                    "nutrients": {
                        "potassium": 3,
                        "titanium": 0,
                        "lithium": 0,
                        "thorium": 0,
                        "barium": 0,
                    },
                    "grew": True,
                }
            ],
        }

        self.assertEqual(
            PlantRuleBasedTeacher.candidates(info),
            [
                {
                    "rule_type": "presence",
                    "nutrient": "potassium",
                    "level": 3,
                    "conditions": [{"nutrient": "potassium", "level": 3}],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
