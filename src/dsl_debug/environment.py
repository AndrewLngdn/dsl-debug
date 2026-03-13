"""Multi-turn debugging environment with a Gymnasium-style reset/step API.

Provides a tool-use interface where an LLM agent can:
- run(code): Execute DSL code and see output/errors
- inspect(node_name): See intermediate output at a specific node
- read_docs(operation): Read documentation for a DSL operation
- submit(code): Submit a fix (ends episode, provides reward)

The environment parses <tool_call> XML tags from the model's raw text output,
matching the Qwen2.5 native tool-calling format used during training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .interpreter import Interpreter, format_table, Result


# ---------------------------------------------------------------------------
# DSL documentation (for read_docs tool)
# ---------------------------------------------------------------------------

DSL_DOCS = {
    "load": """load(name)
Load a named dataset into a variable.
Example: sales = load("sales")
The dataset must exist in the environment.""",

    "filter": """filter(column operator value)
Keep rows where the condition is true.
Operators: ==, !=, >, <, >=, <=
String values must be quoted: filter(status == "completed")
Numeric values are unquoted: filter(amount > 100)
Example: x |> filter(region == "west")""",

    "select": """select(col1, col2, ...)
Keep only the named columns, drop all others.
Example: x |> select(name, amount, status)""",

    "group_by": """group_by(col1, col2, ...)
Group rows by the specified columns. Must be followed by aggregate().
Example: x |> group_by(region) |> aggregate(total: sum(amount))""",

    "aggregate": """aggregate(name: func(col), ...)
Compute aggregate values for each group. Must follow group_by().
Functions: sum(col), count(), avg(col), min(col), max(col)
count() takes no argument.
Example: x |> group_by(dept) |> aggregate(total_sal: sum(salary), headcount: count())""",

    "join": """join(left, right, on: left_key == right_key)
Inner join two datasets on matching keys.
Only rows with matching keys in both datasets are kept.
Example: result = join(orders, customers, on: customer_id == id)""",

    "compute": """compute(name: expression)
Add a new computed column. Expression supports +, -, *, / with column references.
Example: x |> compute(avg_deal: revenue / count)""",

    "sort_by": """sort_by(column, asc/desc)
Sort rows by a column. Default is asc (ascending).
Example: x |> sort_by(revenue, desc)""",

    "take": """take(n)
Keep only the first n rows.
Example: x |> take(5)""",

    "emit": """emit variable
Output the final result. Every program must have exactly one emit statement.
Example: emit result""",

    "pipe": """|> (pipe operator)
Chain operations on a dataset. Each |> passes the result to the next operation.
Example:
  result = data
    |> filter(status == "active")
    |> sort_by(score, desc)
    |> take(10)""",
}

# All docs as a single reference
DSL_REFERENCE = "\n\n".join(f"### {name}\n{doc}" for name, doc in DSL_DOCS.items())


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

RE_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Fallback: bare JSON object with "name" key (models sometimes omit XML tags after SFT)
RE_BARE_JSON = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse a single tool call from model output. Returns (name, arguments) or None.

    Supports two formats:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>  (preferred)
    2. {"name": "...", "arguments": {...}}  (bare JSON fallback for SFT models)
    """
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse all tool calls from model output. Returns list of (name, arguments).

    Models may produce multiple <tool_call> blocks in a single response (parallel
    tool use). We process all of them sequentially within a single turn.
    """
    results = []

    # Find all XML-wrapped tool calls
    for m in RE_TOOL_CALL.finditer(text):
        json_str = m.group(1).strip()
        if not json_str:
            continue
        try:
            payload = json.loads(json_str)
            name = payload.get("name", "")
            args = payload.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                results.append((name, args))
        except json.JSONDecodeError:
            continue

    if results:
        return results

    # Fallback: try bare JSON (single call only)
    m2 = RE_BARE_JSON.search(text)
    if m2:
        try:
            payload = json.loads(m2.group(0))
            name = payload.get("name", "")
            args = payload.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                return [(name, args)]
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Example dataclass
# ---------------------------------------------------------------------------

@dataclass
class Example:
    """A single debugging problem with buggy code and expected output."""
    datasets: dict[str, list[dict]]
    correct_code: str
    buggy_code: str
    expected_output: list[dict]
    bug_info: list[dict]
    difficulty: str
    domain: str
    num_nodes: int = 0
    has_join: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single environment step."""
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


def _str_arg(args, key: str) -> str:
    """Extract a string argument, coercing non-strings to str.

    Models sometimes emit non-string values (e.g. int node names) in tool
    call arguments. This normalizes them so handlers don't need to worry
    about type mismatches from malformed JSON. If args is not a dict
    (e.g. verl passes a raw string from a malformed tool call), return empty.
    """
    if not isinstance(args, dict):
        return ""
    val = args.get(key, "")
    if not isinstance(val, str):
        return str(val) if val else ""
    return val


class DSLDebugEnv:
    """Multi-turn debugging environment with Gymnasium-style reset/step API.

    Usage:
        env = DSLDebugEnv(max_turns=8)
        obs = env.reset(example)       # -> str (initial observation)
        result = env.step(action_str)   # -> StepResult
        # result.observation: tool response text
        # result.reward: 0.0 until submit, then 0.0 or 1.0
        # result.done: True after submit or max turns
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.interpreter = Interpreter()

        # Episode state
        self.example: Example | None = None
        self.turn: int = 0
        self.done: bool = False
        self.inspected_nodes: set[str] = set()
        self.turn_rewards: list[float] = []
        self.outcome_reward: float = 0.0

    def reset(self, example: Example) -> str:
        """Start a new debugging episode. Returns the initial observation."""
        self.example = example
        self.turn = 0
        self.done = False
        self.inspected_nodes = set()
        self.turn_rewards = []
        self.outcome_reward = 0.0

        # Build initial observation with the buggy code and expected output
        expected = example.expected_output
        n_rows = len(expected)
        cols = list(expected[0].keys()) if expected else []

        obs_parts = [
            "You are debugging a dataflow program. The program has a bug that causes incorrect output.",
            "",
            "## Buggy Code",
            "```",
            example.buggy_code.strip(),
            "```",
            "",
            f"## Expected Output",
            f"The correct program should produce a table with {n_rows} rows and columns: {', '.join(cols)}.",
            f"Expected result:",
            "```",
            format_table(expected),
            "```",
            "",
            f"Available nodes for inspect: {self.interpreter.get_node_names(example.buggy_code)}",
            f"You have {self.max_turns} turns to find and fix the bug.",
        ]

        return "\n".join(obs_parts)

    def step(self, action: str) -> StepResult:
        """Execute model's tool call(s). Returns StepResult.

        Args:
            action: Raw model output text containing <tool_call> XML tags.

        Returns:
            StepResult with observation, reward (0.0 or 1.0 on submit), and done flag.

        Supports multiple tool calls in a single response — they are processed
        sequentially but count as a single turn. If a 'submit' call appears,
        it is always processed last and ends the episode.
        """
        if self.done:
            return StepResult(
                observation="Episode is already done.",
                reward=0.0,
                done=True,
            )

        self.turn += 1
        calls = parse_tool_calls(action)

        if not calls:
            self.turn_rewards.append(0.0)
            if self.turn >= self.max_turns:
                self.done = True
                return StepResult(
                    observation="Invalid tool call format. Max turns reached. Episode ended.",
                    reward=0.0,
                    done=True,
                    info={"reason": "invalid_tool_call_and_max_turns"},
                )
            return StepResult(
                observation=(
                    "Invalid tool call. Use the format: "
                    '<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>'
                ),
                reward=0.0,
                done=False,
            )

        # Process submit last if present (other calls first for info gathering)
        submit_call = None
        other_calls = []
        for name, args in calls:
            if name == "submit":
                submit_call = (name, args)
            else:
                other_calls.append((name, args))

        observations = []
        total_reward = 0.0

        for name, args in other_calls:
            result = self._dispatch_tool(name, args)
            observations.append(result.observation)
            total_reward += result.reward
            if result.done:
                return StepResult(
                    observation="\n\n".join(observations),
                    reward=total_reward,
                    done=True,
                    info=result.info,
                )

        if submit_call:
            result = self._dispatch_tool(*submit_call)
            observations.append(result.observation)
            total_reward += result.reward
            return StepResult(
                observation="\n\n".join(observations),
                reward=total_reward,
                done=True,
                info=result.info,
            )

        combined_obs = "\n\n".join(observations)
        remaining = self.max_turns - self.turn
        if remaining <= 0:
            self.done = True
            combined_obs += "\nMax turns reached. Episode ended."
        return StepResult(
            observation=combined_obs,
            reward=total_reward,
            done=self.done,
        )

    _TOOL_HANDLERS = {"run": "_handle_run", "inspect": "_handle_inspect",
                      "read_docs": "_handle_read_docs", "submit": "_handle_submit"}

    def _dispatch_tool(self, name: str, args: dict) -> StepResult:
        """Dispatch a single tool call. Does NOT increment turn counter."""
        handler_name = self._TOOL_HANDLERS.get(name)
        if handler_name:
            return getattr(self, handler_name)(args)

        self.turn_rewards.append(0.0)
        if self.turn >= self.max_turns:
            self.done = True
        return StepResult(
            observation=f"Unknown tool: '{name}'. Available tools: run, inspect, read_docs, submit.",
            reward=0.0,
            done=self.done,
        )

    def _handle_run(self, args: dict) -> StepResult:
        code = _str_arg(args, "code")
        self.turn_rewards.append(0.0)
        if not code:
            return StepResult(
                observation="Error: 'code' argument is required for run.",
                reward=0.0,
                done=False,
            )

        result = self.interpreter.run(code, self.example.datasets)
        if result.success:
            obs = f"Output:\n```\n{format_table(result.data)}\n```"
        else:
            obs = f"Error: {result.error}"

        return StepResult(observation=obs, reward=0.0, done=False)

    def _handle_inspect(self, args: dict) -> StepResult:
        node_name = _str_arg(args, "node_name")
        self.turn_rewards.append(0.0)
        if not node_name:
            return StepResult(
                observation="Error: 'node_name' argument is required for inspect.",
                reward=0.0,
                done=False,
            )

        self.inspected_nodes.add(node_name)

        result = self.interpreter.inspect(
            self.example.buggy_code, self.example.datasets, node_name
        )

        if result.success:
            obs = f"Node '{node_name}' output:\n```\n{format_table(result.data)}\n```"
        else:
            obs = f"Error inspecting '{node_name}': {result.error}"

        return StepResult(observation=obs, reward=0.0, done=False)

    def _handle_read_docs(self, args: dict) -> StepResult:
        operation = args.get("operation", "")
        self.turn_rewards.append(0.0)

        if operation in DSL_DOCS:
            obs = f"Documentation for '{operation}':\n{DSL_DOCS[operation]}"
        elif operation == "all" or not operation:
            obs = f"DSL Reference:\n{DSL_REFERENCE}"
        else:
            obs = f"No documentation for '{operation}'. Available: {list(DSL_DOCS.keys())}"

        return StepResult(observation=obs, reward=0.0, done=False)

    def _handle_submit(self, args: dict) -> StepResult:
        code = _str_arg(args, "code")
        self.done = True

        if not code:
            self.outcome_reward = 0.0
            self.turn_rewards.append(0.0)
            return StepResult(
                observation="Error: 'code' argument is required for submit. Episode ended with no fix.",
                reward=0.0,
                done=True,
                info={"correct": False, "reason": "empty_submission"},
            )

        # Run the submitted code
        result = self.interpreter.run(code, self.example.datasets)

        if not result.success:
            self.outcome_reward = 0.0
            self.turn_rewards.append(0.0)
            return StepResult(
                observation=f"Submission failed with error: {result.error}",
                reward=0.0,
                done=True,
                info={"correct": False, "reason": "runtime_error", "error": result.error},
            )

        # Compare output to expected
        correct = result.data == self.example.expected_output
        self.outcome_reward = 1.0 if correct else 0.0
        self.turn_rewards.append(0.0)

        if correct:
            obs = "Correct! Your fix produces the expected output."
        else:
            obs = (
                f"Incorrect. Your output:\n```\n{format_table(result.data)}\n```\n"
                f"Expected:\n```\n{format_table(self.example.expected_output)}\n```"
            )

        return StepResult(
            observation=obs,
            reward=self.outcome_reward,
            done=True,
            info={"correct": correct, "submitted_output": result.data},
        )

    def get_rewards(self) -> dict:
        """Get complete reward information for the episode."""
        return {
            "outcome_reward": self.outcome_reward,
            "turn_rewards": self.turn_rewards,
            "total_turns": self.turn,
        }

    def get_system_prompt(self) -> str:
        """Get the system prompt for the model.

        Uses Qwen2.5's native tool-calling format (<tools> + <tool_call>) so
        the model generates proper tool-call special tokens.

        The <tools> block is generated from TOOL_SCHEMAS (the single source of
        truth for tool definitions) so the prompt and the eval harness / verl
        adapter can never drift out of sync.
        """
        tools_block = "\n".join(json.dumps(s, separators=(",", ":")) for s in TOOL_SCHEMAS)
        return (
            "You are an expert debugger for a dataflow programming language. "
            "Your task is to find and fix bugs in dataflow programs.\n\n"
            "The dataflow language uses pipe operators (|>) to chain data transformations. "
            "Programs define variables, transform data through operations like filter, select, "
            "group_by+aggregate, join, compute, sort_by, and take, then emit a final result.\n\n"
            "# Tools\n\n"
            "You may call one or more functions to assist with debugging. "
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_block}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "Strategy:\n"
            "1. Run the buggy code to see the current (incorrect) output\n"
            "2. Inspect intermediate nodes to narrow down where the bug is\n"
            "3. Once you identify the bug, submit the corrected code\n"
        )


# Tool schemas in OpenAI function-calling format, used by eval harnesses
# and framework adapters to configure the model's available tools
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Execute DSL code and see the output or errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The DSL program to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "View the intermediate output at a specific node in the program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_name": {
                        "type": "string",
                        "description": "Name of the variable/node to inspect",
                    }
                },
                "required": ["node_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_docs",
            "description": "Read documentation for a DSL operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation name (e.g. filter, join, sort_by)",
                    }
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your corrected code. This ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The corrected DSL program",
                    }
                },
                "required": ["code"],
            },
        },
    },
]
