import numpy as np

from agent_system.environments.env_package.discovery.dynamic_sampler import DynamicSeedSampler


def test_jeffreys_prior_is_uniform_before_observations():
    sampler = DynamicSeedSampler([1, 2, 3, 4])

    np.testing.assert_allclose(sampler.probabilities(), [0.25, 0.25, 0.25, 0.25])


def test_probabilities_follow_beta_posterior_means():
    sampler = DynamicSeedSampler([1, 2])
    sampler.observe_group(1, [0.0, 1.0], [0.0, 0.0], accepted=True)
    sampler.observe_group(2, [0.0, 0.0], [0.0, 0.0], accepted=False)

    np.testing.assert_allclose(sampler.probabilities(), [0.75, 0.25])
