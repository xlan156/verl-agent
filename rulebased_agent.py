from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.helpers import all_action_abbr
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgent

def run_agent_loop():
    env = DiscoveryWorldEnv(
        scenario_name="Combinatorial Chemistry",
        difficulty="Easy",
        seed=0,
        max_steps=50
    )
    agent = RulebasedAgent(env)
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(info)
        obs, reward, done, info = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}, Done: {done}")


if __name__ == "__main__":
    run_agent_loop()
                    