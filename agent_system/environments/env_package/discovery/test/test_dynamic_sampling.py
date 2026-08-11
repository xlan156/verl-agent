from agent_system.environments.env_package.discovery.dynamic_sampler import DynamicSeedSampler
import numpy as np

cap_prob = DynamicSeedSampler._cap_probabilities
def test_cap_prob():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    upper_bound = 0.3
    capped_probabilities = cap_prob(probabilities, upper_bound)
    print("capped probabilities:", capped_probabilities)


def test_probabilities():
    sampler = DynamicSeedSampler(seed_pool=[1, 2, 3, 4], initial_probabilities=[0.25, 0.25, 0.25, 0.25])
    

test_cap_prob()
    