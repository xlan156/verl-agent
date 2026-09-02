from __future__ import annotations

import contextlib
import io
import unittest

from agent_system.environments.env_package.discovery.runtime.envs import DiscoveryWorldEnv


class ReactorDifficultyTest(unittest.TestCase):
    def test_teacher_actions_are_valid_and_special_states_offer_choice(self):
        for difficulty in ("Normal",):
            with self.subTest(difficulty=difficulty):
                env = DiscoveryWorldEnv(
                    seed=0, scenario_name="Reactor Lab", difficulty=difficulty, max_steps=50
                )
                saw_multiple_choices = False
                with contextlib.redirect_stdout(io.StringIO()):
                    _, info = env.reset()
                    for _ in range(50):
                        valid_skills = info["valid_skills"]
                        saw_multiple_choices |= len(valid_skills) > 1
                        action = env.teacher.select_skill(info)
                        if action is None:
                            break
                        self.assertTrue(
                            action in valid_skills
                            or (action.startswith("set_reactor_") and info.get("reactor_dynamic_frequency")),
                            f"action={action}, valid_skills={valid_skills}",
                        )
                        _, _, done, info = env.step(action)
                        self.assertEqual(info["action_status"], "success")
                        if done:
                            break
                self.assertTrue(saw_multiple_choices)
                self.assertTrue(info["won"])
                self.assertAlmostEqual(info["score_normalized"], 1.0)
                env.close()


if __name__ == "__main__":
    unittest.main()
