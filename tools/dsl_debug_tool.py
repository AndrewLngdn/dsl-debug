"""
DSL Debug Multi-Turn Tools for verl 0.7.0+ (server mode).

Compatible with verl's ToolAgentLoop (experimental.agent_loop.tool_agent_loop).
  - Returns ToolResponse objects instead of plain strings
  - create() returns tuple[str, ToolResponse]
  - Environment state keyed by agent_data.request_id (stable per episode)
  - Env data passed via tools_kwargs.{tool_name}.create_kwargs in parquet

Monkey-patches applied at import time (verl 0.7 workarounds):
  1. BaseTool.__init__: Suppresses per-tool JSON schema printing to stdout.
     Without this, 8 workers × 4 tools produces thousands of lines per step.
  2. ToolAgentLoop._handle_processing_tools_state: Adds episode termination
     on tool "done" signal. verl's agent loop discards the 3rd return value
     from tool.execute(), so tools have no way to end an episode. This patch
     checks for {"done": True} in tool_info and returns TERMINATED, preventing
     the submit-retry exploit (model calling submit repeatedly).
A third monkey-patch (DAPO group filtering) is in reward_fn.py.
A fourth patch (sed in Dockerfile) increases sglang HTTP engine retry
  attempts from 3→10 and delay from 2s→3s for TP>=2 startup timing.

Four tools wrapping DSLDebugEnv:
- RunTool: Execute DSL code and see output/errors
- InspectTool: View intermediate node output
- ReadDocsTool: Read DSL operation documentation
- SubmitTool: Submit corrected code (ends episode, provides reward)

NOTE: DAPO group filtering is applied in reward_fn.py (not here) because
compute_advantage runs in the TaskRunner process, not the WorkerDict process.
"""

import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Suppress BaseTool.__init__ print spam (prints full tool schema JSON per
# instantiation — 4 tools × ~2000 rollouts/step = 5M+ log lines).
# We monkey-patch rather than editing verl source per project conventions.
# ---------------------------------------------------------------------------
_original_base_tool_init = BaseTool.__init__


def _quiet_base_tool_init(self, config, tool_schema):
    self.config = config
    self.tool_schema = tool_schema or self.get_openai_tool_schema()
    assert self.tool_schema is not None, "Tool schema is not set!"
    self.name = self.tool_schema.function.name
    # Intentionally omit the print(json.dumps(...)) from original


BaseTool.__init__ = _quiet_base_tool_init


# ---------------------------------------------------------------------------
# Monkey-patch ToolAgentLoop to terminate on tool "done" signal.
#
# verl's ToolAgentLoop discards the 3rd return value from tool.execute()
# and has no mechanism for tools to end an episode. This means submit can
# be called repeatedly (model learns brute-force retry instead of debugging).
#
# Fix: check the tool result dict for {"done": True} and terminate.
# The tool response is still included in the trajectory so the model sees
# the submit feedback, but no further generation happens.
# ---------------------------------------------------------------------------
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, AgentState

_original_handle_processing_tools = ToolAgentLoop._handle_processing_tools_state


async def _handle_processing_tools_with_done(self, agent_data):
    """Patched version that checks for 'done' in tool results."""
    import asyncio
    from verl.utils.profiler import simple_timer

    add_messages = []
    new_images_this_turn = []
    episode_done = False

    tasks = []
    tool_call_names = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
        tool_call_names.append(tool_call.name)

    with simple_timer("tool_calls", agent_data.metrics):
        responses = await asyncio.gather(*tasks)

    for tool_response, tool_reward, tool_info in responses:
        # Check for done signal from tool
        if isinstance(tool_info, dict) and tool_info.get("done", False):
            episode_done = True

        # Text-only content (we only use text tools)
        message = {"role": "tool", "content": tool_response.text or ""}
        add_messages.append(message)

        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

    agent_data.messages.extend(add_messages)

    # Tokenize tool response
    if self.tool_parser_name == "gpt-oss":
        from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
        tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
        response_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
        )
    else:
        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn,
            videos=None,
            remove_system_prompt=True,
        )

    if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
        return AgentState.TERMINATED

    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)
    agent_data.user_turns += 1

    # Terminate if any tool signaled done (submit was called)
    if episode_done:
        return AgentState.TERMINATED

    return AgentState.GENERATING


ToolAgentLoop._handle_processing_tools_state = _handle_processing_tools_with_done
print("[monkey-patch] ToolAgentLoop terminates on tool done signal (submit is final)")


# =============================================================================
# Episode progress tracker (logs every LOG_INTERVAL_S seconds)
# =============================================================================
_PROGRESS = {
    "episodes_started": 0,
    "episodes_submitted": 0,
    "tool_calls": 0,
    "last_log_time": 0.0,
    "batch_start_time": 0.0,
}
_LOG_INTERVAL_S = int(os.environ.get("DSL_DEBUG_LOG_INTERVAL_S", "30"))


def _log_progress(event: str) -> None:
    """Update counters and periodically print progress."""
    _PROGRESS[event] = _PROGRESS.get(event, 0) + 1
    now = time.time()
    if now - _PROGRESS["last_log_time"] >= _LOG_INTERVAL_S:
        elapsed = now - _PROGRESS["batch_start_time"] if _PROGRESS["batch_start_time"] else 0
        print(
            f"[progress] episodes={_PROGRESS['episodes_started']} "
            f"submitted={_PROGRESS['episodes_submitted']} "
            f"tool_calls={_PROGRESS['tool_calls']} "
            f"active_envs={len(_ENV_STORE)} "
            f"elapsed={elapsed:.0f}s"
        )
        _PROGRESS["last_log_time"] = now


def _reset_progress() -> None:
    """Reset counters at start of each batch."""
    _PROGRESS["episodes_started"] = 0
    _PROGRESS["episodes_submitted"] = 0
    _PROGRESS["tool_calls"] = 0
    _PROGRESS["batch_start_time"] = time.time()
    _PROGRESS["last_log_time"] = 0.0


# =============================================================================
# Shared environment store (module-level, keyed by agent_data.request_id)
# =============================================================================
# In verl 0.7, each tool call gets create→execute→release with an ephemeral
# instance_id. But tools within the same episode share state via request_id.

_ENV_STORE: dict[str, dict] = {}


def _get_or_create_env(
    key: str,
    datasets: dict | None = None,
    buggy_code: str = "",
    correct_code: str = "",
    expected_output: list | None = None,
    bug_info: dict | None = None,
    difficulty: str = "medium",
    domain: str = "unknown",
    num_nodes: int = 0,
    has_join: bool = False,
    max_turns: int = 8,
    **kwargs,
) -> dict:
    """Create or retrieve a shared environment instance."""
    if key in _ENV_STORE:
        return _ENV_STORE[key]

    from dsl_debug.data_generator import Example
    from dsl_debug.environment import DSLDebugEnv

    # Deserialize JSON-encoded fields from parquet (see prepare_data.py)
    if isinstance(datasets, str):
        datasets = json.loads(datasets)
    if isinstance(expected_output, str):
        expected_output = json.loads(expected_output)
    if isinstance(bug_info, str):
        bug_info = json.loads(bug_info)

    example = Example(
        datasets=datasets or {},
        correct_code=correct_code,
        buggy_code=buggy_code,
        expected_output=expected_output or [],
        bug_info=bug_info or {},
        difficulty=difficulty,
        domain=domain,
        num_nodes=num_nodes,
        has_join=has_join,
    )

    env = DSLDebugEnv(max_turns=max_turns)
    initial_obs = env.reset(example)

    state = {
        "env": env,
        "example": example,
        "initial_obs": initial_obs,
        "done": False,
        "submitted": False,
    }
    _ENV_STORE[key] = state

    # Track episode creation; reset counters if store was empty (new batch)
    if len(_ENV_STORE) == 1:
        _reset_progress()
    _log_progress("episodes_started")

    return state


def _get_env_from_agent_data(agent_data) -> dict:
    """Get or lazily create env from agent_data, keyed by request_id.

    In verl 0.7's ToolAgentLoop, each tool call gets a fresh instance_id,
    but agent_data.request_id is stable for the entire episode. We use it
    as the env store key so all tools share state within an episode.

    Env initialization data comes from tools_kwargs in the parquet, structured as:
        tools_kwargs.{any_tool_name}.create_kwargs = {datasets, buggy_code, ...}
    """
    request_id = agent_data.request_id
    if request_id in _ENV_STORE:
        return _ENV_STORE[request_id]

    # Extract create_kwargs from tools_kwargs (any tool's entry will do)
    env_kwargs = {}
    for tool_name, tool_config in agent_data.tools_kwargs.items():
        create_kwargs = tool_config.get("create_kwargs", {})
        if create_kwargs:
            env_kwargs = create_kwargs
            break

    if not env_kwargs:
        raise RuntimeError(
            f"No create_kwargs found in tools_kwargs for request {request_id}. "
            f"Ensure parquet has extra_info.tools_kwargs with env data."
        )

    return _get_or_create_env(request_id, **env_kwargs)


def _release_env(key: str) -> None:
    """Remove an environment instance."""
    _ENV_STORE.pop(key, None)


# =============================================================================
# RunTool
# =============================================================================

class RunTool(BaseTool):
    """Execute DSL code and see the output or errors."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        state = _get_env_from_agent_data(agent_data)
        if state["done"]:
            return ToolResponse(text="Episode already finished."), 0.0, {"done": True}

        env = state["env"]
        result = env._handle_run(parameters)
        _log_progress("tool_calls")
        return ToolResponse(text=result.observation), result.reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass  # Cleanup handled by SubmitTool


# =============================================================================
# InspectTool
# =============================================================================

class InspectTool(BaseTool):
    """View the intermediate output at a specific node in the program."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        state = _get_env_from_agent_data(agent_data)
        if state["done"]:
            return ToolResponse(text="Episode already finished."), 0.0, {"done": True}

        env = state["env"]
        result = env._handle_inspect(parameters)
        _log_progress("tool_calls")
        return ToolResponse(text=result.observation), result.reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


# =============================================================================
# ReadDocsTool
# =============================================================================

class ReadDocsTool(BaseTool):
    """Read documentation for a DSL operation."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        state = _get_env_from_agent_data(agent_data)
        if state["done"]:
            return ToolResponse(text="Episode already finished."), 0.0, {"done": True}

        env = state["env"]
        result = env._handle_read_docs(parameters)
        _log_progress("tool_calls")
        return ToolResponse(text=result.observation), result.reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass


# =============================================================================
# SubmitTool
# =============================================================================

class SubmitTool(BaseTool):
    """Submit corrected code. This ends the episode."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        state = _get_env_from_agent_data(agent_data)
        if state["done"]:
            return ToolResponse(text="Episode already finished."), 0.0, {"done": True}

        env = state["env"]
        result = env._handle_submit(parameters)
        state["done"] = True
        state["submitted"] = True
        _log_progress("episodes_submitted")

        correct = result.info.get("correct", False) if result.info else False
        return ToolResponse(text=result.observation), 0.0, {"correct": correct, "done": True}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Episode reward: 1.0 if correct submission, 0.0 otherwise."""
        # In v0.7, calc_reward may not be called by the ToolAgentLoop.
        # The reward is handled by custom_reward_function instead.
        # But we implement it for compatibility.
        state = _ENV_STORE.get(instance_id)
        if state is None:
            return 0.0
        return state["env"].outcome_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up the shared environment instance.

        Note: In v0.7, instance_id is ephemeral (per-tool-call), but the env
        is keyed by request_id. We clean up envs for submitted episodes to
        prevent memory leaks.
        """
        # Clean up envs that are marked as submitted
        # (done lazily — env persists until next GC)
        submitted_keys = [k for k, v in _ENV_STORE.items() if v.get("submitted")]
        for key in submitted_keys:
            _release_env(key)
