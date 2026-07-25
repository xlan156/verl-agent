import json
from types import SimpleNamespace

import numpy as np

from agent_system.environments.env_package.discovery.dynamic_sampler import (
    DynamicSeedSampler,
)
from agent_system.multi_turn_rollout.utils import (
    filter_group_data,
    reward_variance_group_mask,
)


def test_reward_variance_group_mask_filters_constant_groups():
    mask = reward_variance_group_mask(
        episode_rewards=np.array([0.0, 0.0, 1.0, 2.0]),
        batch_size=2,
        group_n=2,
    )
    assert mask.tolist() == [False, True]


def test_teacher_only_variance_does_not_make_group_informative():
    shaped_mask = reward_variance_group_mask(
        episode_rewards=np.array([1.0, 3.0]),
        batch_size=1,
        group_n=2,
    )
    task_mask = reward_variance_group_mask(
        episode_rewards=np.array([0.0, 0.0]),
        batch_size=1,
        group_n=2,
    )
    assert shaped_mask.tolist() == [True]
    assert task_mask.tolist() == [False]


def test_group_filter_uses_teacher_free_rewards_but_keeps_shaped_returns():
    config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=2),
        env=SimpleNamespace(rollout=SimpleNamespace(n=2)),
        algorithm=SimpleNamespace(
            filter_groups={"reward_variance_epsilon": 0.0}
        ),
    )
    batch_list = [
        [{"uid": "teacher-only"}],
        [{"uid": "teacher-only"}],
        [{"uid": "task-progress"}],
        [{"uid": "task-progress"}],
    ]
    shaped_rewards = np.array([0.0, 4.0, 1.0, 3.0])
    task_rewards = np.array([0.0, 0.0, 1.0, 3.0])

    filtered = filter_group_data(
        batch_list=batch_list,
        episode_rewards=shaped_rewards,
        episode_lengths=np.ones(4),
        success={"success_rate": np.zeros(4)},
        traj_uid=np.arange(4),
        tool_callings=np.zeros(4),
        config=config,
        filter_rewards=task_rewards,
    )

    kept_batch, kept_rewards, *_ = filtered
    assert [row[0]["uid"] for row in kept_batch] == [
        "task-progress",
        "task-progress",
    ]
    assert kept_rewards.tolist() == [1.0, 3.0]


def test_sampler_warmup_covers_seed_pool():
    sampler = DynamicSeedSampler(
        seeds=[0, 1, 2, 3],
        config={"min_attempts_per_seed": 1},
        rng_seed=7,
    )
    sampled = sampler.sample(4)
    assert sorted(sampled) == [0, 1, 2, 3]


def test_sampler_prioritizes_hard_informative_seed_with_uniform_floor():
    sampler = DynamicSeedSampler(
        seeds=[0, 1, 2],
        config={
            "uniform_ratio": 0.3,
            "ema_alpha": 1.0,
            "min_attempts_per_seed": 0,
            "max_probability_per_seed": 0.8,
        },
        rng_seed=3,
    )
    sampler.observe_group(0, [0.0, 1.0], [0.0, 0.0], accepted=True)
    sampler.observe_group(1, [0.0, 1.0], [0.0, 1.0], accepted=True)
    sampler.observe_group(2, [0.0, 0.0], [1.0, 1.0], accepted=False)

    probabilities = dict(zip(sampler.seeds, sampler.probabilities()))
    assert probabilities[1] > probabilities[2]
    assert probabilities[0] > probabilities[2]
    assert all(value >= 0.3 / 3 for value in probabilities.values())


def test_sampler_priority_ignores_teacher_only_reward_variance():
    sampler = DynamicSeedSampler(
        seeds=[0, 1, 2],
        config={
            "uniform_ratio": 0.0,
            "ema_alpha": 1.0,
            "min_attempts_per_seed": 0,
            "max_probability_per_seed": 0.8,
            "min_informativeness": 0.0,
        },
        rng_seed=3,
    )
    sampler.observe_group(
        0,
        rewards=[0.0, 4.0],
        task_rewards=[0.0, 0.0],
        successes=[0.0, 0.0],
        accepted=False,
    )
    sampler.observe_group(
        1,
        rewards=[0.0, 2.0],
        task_rewards=[0.0, 2.0],
        successes=[0.0, 0.0],
        accepted=True,
    )
    sampler.observe_group(
        2,
        rewards=[2.0, 2.0],
        task_rewards=[2.0, 2.0],
        successes=[1.0, 1.0],
        accepted=False,
    )

    probabilities = dict(zip(sampler.seeds, sampler.probabilities()))
    assert probabilities[1] > probabilities[0]
    assert sampler.stats[0].reward_std_ema > 0.0
    assert sampler.stats[0].task_reward_std_ema == 0.0


def test_sampler_state_round_trip_is_json_serializable():
    sampler = DynamicSeedSampler([4, 5], {"min_attempts_per_seed": 0}, rng_seed=11)
    sampler.observe_group(4, [0.0, 2.0], [0.0, 1.0], accepted=True)
    state = json.loads(json.dumps(sampler.state_dict()))

    restored = DynamicSeedSampler([4, 5], {"min_attempts_per_seed": 0}, rng_seed=99)
    restored.load_state_dict(state)

    assert restored.state_dict()["stats"] == state["stats"]
    assert restored.total_groups == 1
    assert restored.sample(1) == sampler.sample(1)


def test_sampler_loads_version_one_checkpoint_without_reusing_shaped_variance():
    sampler = DynamicSeedSampler([4], {"min_attempts_per_seed": 0}, rng_seed=11)
    old_state = {
        "version": 1,
        "seeds": [4],
        "stats": {
            "4": {
                "groups": 3,
                "trajectories": 6,
                "accepted_groups": 2,
                "rejected_groups": 1,
                "successes": 0,
                "success_ema": 0.0,
                "reward_std_ema": 2.0,
                "learning_progress_ema": 0.0,
                "last_sampled_step": 2,
            }
        },
        "total_groups": 3,
        "total_trajectories": 6,
    }

    sampler.load_state_dict(old_state)

    assert sampler.stats[4].reward_std_ema == 2.0
    assert sampler.stats[4].task_reward_std_ema == 0.0
