#!/usr/bin/env python3
"""Run the rule-based Reactor Lab teacher at every supported difficulty."""
import argparse, csv, json
from pathlib import Path
from agent_system.environments.env_package.discovery.runtime.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.reactor.state import prompt_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/reactor_teacher_state_response.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    rows = []
    for difficulty in ("Easy", "Normal", "Challenge"):
        for seed in args.seeds:
            env = DiscoveryWorldEnv(seed=seed, scenario_name="Reactor Lab", difficulty=difficulty, max_steps=args.max_steps)
            _, info = env.reset()
            for step in range(args.max_steps):
                action = env.teacher.select_skill(info)
                if action is None:
                    break
                state = prompt_state(info)
                response = f"<think>Choose the next rule-based action from the observable reactor state.</think><action>{action}</action>"
                rows.append({"difficulty": difficulty.lower(), "seed": seed, "step": step, "state": json.dumps(state, sort_keys=True), "response": response, "action": action})
                _, _, done, info = env.step(action)
                if done:
                    break
            env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["difficulty", "seed", "step", "state", "response", "action"])
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")

if __name__ == "__main__":
    main()
