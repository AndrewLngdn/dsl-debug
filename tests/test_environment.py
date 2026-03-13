"""Tests for the RL environment."""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dsl_debug import DSLDebugEnv
from dsl_debug.environment import parse_tool_call
from dsl_debug.data_generator import DataGenerator
from dsl_debug.interpreter import Interpreter


@pytest.fixture
def example():
    gen = DataGenerator(seed=42)
    return gen.generate_example(difficulty="medium")


@pytest.fixture
def env():
    return DSLDebugEnv(max_turns=8)


class TestToolCallParsing:
    def test_parse_valid(self):
        text = '<tool_call>{"name": "run", "arguments": {"code": "x = 1"}}</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        name, args = result
        assert name == "run"
        assert args["code"] == "x = 1"

    def test_parse_with_surrounding_text(self):
        text = 'Let me run the code. <tool_call>{"name": "run", "arguments": {"code": "test"}}</tool_call> Done.'
        result = parse_tool_call(text)
        assert result is not None
        assert result[0] == "run"

    def test_parse_invalid_json(self):
        text = "<tool_call>not json</tool_call>"
        assert parse_tool_call(text) is None

    def test_parse_no_tags(self):
        text = "Just some regular text"
        assert parse_tool_call(text) is None


class TestEnvironment:
    def test_reset(self, env, example):
        obs = env.reset(example)
        assert "Buggy Code" in obs
        assert env.turn == 0
        assert not env.done

    def test_run_tool(self, env, example):
        env.reset(example)
        action = f'<tool_call>{{"name": "run", "arguments": {{"code": {json.dumps(example.buggy_code)}}}}}</tool_call>'
        result = env.step(action)
        assert not result.done or env.turn >= env.max_turns
        assert result.reward == 0.0

    def test_inspect_tool(self, env, example):
        env.reset(example)
        nodes = Interpreter().get_node_names(example.buggy_code)
        if nodes:
            action = f'<tool_call>{{"name": "inspect", "arguments": {{"node_name": "{nodes[0]}"}}}}</tool_call>'
            result = env.step(action)
            assert result.reward == 0.0

    def test_inspect_repeat_no_crash(self, env, example):
        """Re-inspecting the same node is allowed (no penalty — rewards are binary)."""
        env.reset(example)
        nodes = Interpreter().get_node_names(example.buggy_code)
        if nodes:
            node = nodes[0]
            action = f'<tool_call>{{"name": "inspect", "arguments": {{"node_name": "{node}"}}}}</tool_call>'
            env.step(action)
            result = env.step(action)  # second time
            assert result.reward == 0.0

    def test_invalid_tool_call(self, env, example):
        """Invalid tool call returns 0.0 reward (rewards are binary: 0.0 or 1.0 on submit)."""
        env.reset(example)
        result = env.step("this is not a valid tool call")
        assert result.reward == 0.0

    def test_submit_correct(self, env, example):
        env.reset(example)
        action = f'<tool_call>{{"name": "submit", "arguments": {{"code": {json.dumps(example.correct_code)}}}}}</tool_call>'
        result = env.step(action)
        assert result.done
        assert result.info["correct"]
        assert result.reward == 1.0

    def test_submit_wrong(self, env, example):
        env.reset(example)
        action = f'<tool_call>{{"name": "submit", "arguments": {{"code": {json.dumps(example.buggy_code)}}}}}</tool_call>'
        result = env.step(action)
        assert result.done
        assert not result.info["correct"]
        assert result.reward == 0.0

    def test_max_turns(self, example):
        env = DSLDebugEnv(max_turns=2)
        env.reset(example)
        action = f'<tool_call>{{"name": "run", "arguments": {{"code": {json.dumps(example.buggy_code)}}}}}</tool_call>'
        env.step(action)
        result = env.step(action)
        assert result.done

    def test_read_docs(self, env, example):
        env.reset(example)
        action = '<tool_call>{"name": "read_docs", "arguments": {"operation": "filter"}}</tool_call>'
        result = env.step(action)
        assert "filter" in result.observation.lower()
        assert result.reward == 0.0


class TestRewardDeterminism:
    """Correct code must always match expected output."""

    def test_correct_code_always_matches(self):
        gen = DataGenerator(seed=42)
        interp = Interpreter()
        for i in range(50):
            example = gen.generate_example(difficulty="medium")
            if example is None:
                continue
            result = interp.run(example.correct_code, example.datasets)
            assert result.success, f"Example {i}: correct_code failed"
            assert result.data == example.expected_output

    def test_buggy_code_never_matches(self):
        gen = DataGenerator(seed=42)
        interp = Interpreter()
        for i in range(50):
            example = gen.generate_example(difficulty="medium")
            if example is None:
                continue
            result = interp.run(example.buggy_code, example.datasets)
            if result.success:
                assert result.data != example.expected_output
