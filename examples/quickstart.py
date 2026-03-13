#!/usr/bin/env python3
"""Minimal example: run a single debugging episode with an oracle agent.

This demonstrates the core DSLDebugEnv API without any ML dependencies.
The oracle agent knows the correct code and submits it directly.
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dsl_debug import DSLDebugEnv
from dsl_debug.data_generator import DataGenerator


def main():
    # Generate a random debugging problem
    gen = DataGenerator(seed=42)
    example = gen.generate_example(difficulty="medium")
    if not example:
        print("Failed to generate example")
        return

    print("=" * 60)
    print("DSL Debug Environment — Quickstart")
    print("=" * 60)

    # Create environment and start episode
    env = DSLDebugEnv(max_turns=8)
    obs = env.reset(example)
    print(f"\n--- Initial Observation ---\n{obs}\n")

    # Turn 1: Run the buggy code to see what's wrong
    action = f'<tool_call>{{"name": "run", "arguments": {{"code": {json.dumps(example.buggy_code)}}}}}</tool_call>'
    result = env.step(action)
    print(f"--- Turn 1: Run buggy code ---\n{result.observation}\n")
    print(f"  reward={result.reward}, done={result.done}")

    # Turn 2: Submit the correct fix
    action = f'<tool_call>{{"name": "submit", "arguments": {{"code": {json.dumps(example.correct_code)}}}}}</tool_call>'
    result = env.step(action)
    print(f"\n--- Turn 2: Submit fix ---\n{result.observation}\n")
    print(f"  reward={result.reward}, done={result.done}")

    # Show final rewards
    rewards = env.get_rewards()
    print(f"\n--- Episode Summary ---")
    print(f"  Outcome: {'CORRECT' if rewards['outcome_reward'] == 1.0 else 'WRONG'}")
    print(f"  Turns used: {rewards['total_turns']}")
    print(f"  Turn rewards: {rewards['turn_rewards']}")

    # Show what the bug was
    print(f"\n--- Bug Info ---")
    for bug in example.bug_info:
        print(f"  {bug['category']}: {bug['description']}")


if __name__ == "__main__":
    main()
