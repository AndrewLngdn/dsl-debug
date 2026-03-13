"""Tests for the HTTP server (requires test sets to be available)."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pytest
from dsl_debug import DSLDebugEnv
from dsl_debug.data_generator import DataGenerator


class TestServerEndpoints:
    """Test server logic without actually starting FastAPI."""

    def test_episode_lifecycle(self):
        """Full episode: reset -> run -> submit."""
        gen = DataGenerator(seed=42)
        example = gen.generate_example(difficulty="easy")
        assert example is not None

        env = DSLDebugEnv(max_turns=8)
        obs = env.reset(example)
        assert "Buggy Code" in obs

        # Run buggy code
        action = f'<tool_call>{{"name": "run", "arguments": {{"code": {json.dumps(example.buggy_code)}}}}}</tool_call>'
        result = env.step(action)
        assert not result.done

        # Submit correct code
        action = f'<tool_call>{{"name": "submit", "arguments": {{"code": {json.dumps(example.correct_code)}}}}}</tool_call>'
        result = env.step(action)
        assert result.done
        assert result.info["correct"]

    def test_multiple_episodes(self):
        """Can run multiple episodes sequentially."""
        gen = DataGenerator(seed=42)
        env = DSLDebugEnv(max_turns=8)

        for _ in range(5):
            example = gen.generate_example(difficulty="easy")
            if example is None:
                continue
            env.reset(example)
            action = f'<tool_call>{{"name": "submit", "arguments": {{"code": {json.dumps(example.correct_code)}}}}}</tool_call>'
            result = env.step(action)
            assert result.done
            assert result.info["correct"]
