import math
from types import SimpleNamespace

from agent_system.environments.env_package.discovery.rewards import (
    DiscoveryWorldRewardMixin,
    GAME_PROGRESS_REWARD_SCALE,
    MAX_NON_TERMINAL_REWARD,
)


class RewardHarness(DiscoveryWorldRewardMixin):
    pass


def test_game_progress_scale_makes_eight_percent_step_visible():
    reward = DiscoveryWorldRewardMixin._game_progress_reward(
        SimpleNamespace(_prev_score=0.25),
        cur_score=1.0 / 3.0,
    )
    assert math.isclose(reward, GAME_PROGRESS_REWARD_SCALE / 12.0)
    assert reward > 2.0


def test_positive_clip_preserves_largest_normalized_progress_step():
    reward = DiscoveryWorldRewardMixin.clip_reward(
        SimpleNamespace(),
        GAME_PROGRESS_REWARD_SCALE + 1.0,
    )
    assert reward == MAX_NON_TERMINAL_REWARD


def test_negative_clip_remains_bounded():
    reward = DiscoveryWorldRewardMixin.clip_reward(SimpleNamespace(), -100.0)
    assert reward == -2.0


def test_target_distance_rewards_moving_toward_hidden_target():
    env = RewardHarness()
    env._hidden_chemical_target = (1, 1, 1, 0)
    env._last_info = {"chemical_dict": {"A": 1, "B": 0, "C": 0, "D": 0}}
    reward = env._target_distance_reward(
        {"chemical_dict": {"A": 1, "B": 1, "C": 0, "D": 0}},
    )
    assert reward == 1.0


def test_target_distance_penalizes_moving_away_from_hidden_target():
    env = RewardHarness()
    env._hidden_chemical_target = (1, 1, 1, 0)
    env._last_info = {"chemical_dict": {"A": 1, "B": 1, "C": 0, "D": 0}}
    reward = env._target_distance_reward(
        {"chemical_dict": {"A": 1, "B": 0, "C": 0, "D": 0}},
    )
    assert reward == -1.0
