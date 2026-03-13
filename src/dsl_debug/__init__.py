"""DSL Debug Environment: A multi-turn debugging environment for LLM agents.

Core API (zero dependencies):
    from dsl_debug import DSLDebugEnv, Example

    env = DSLDebugEnv(max_turns=8)
    obs = env.reset(example)       # initial observation string
    result = env.step(action_str)  # StepResult(observation, reward, done, info)

Test set loading (requires pandas + pyarrow):
    from dsl_debug import load_test_set

    problems = load_test_set("standard")  # -> list[Example]
"""

from .environment import DSLDebugEnv, StepResult, Example, TOOL_SCHEMAS
from .interpreter import Interpreter, format_table

__all__ = [
    "DSLDebugEnv",
    "StepResult",
    "Example",
    "Interpreter",
    "format_table",
    "TOOL_SCHEMAS",
    "load_test_set",
    "load_test_set_raw",
]


def load_test_set(split: str = "standard") -> list[Example]:
    """Load a test set split as a list of Example objects.

    Args:
        split: One of "standard" (481 problems), "nonlocal" (200), "intent_mismatch" (177)

    Returns:
        List of Example objects ready for env.reset()

    Requires: pandas, pyarrow (install with `pip install dsl-debug[test_sets]`)
    """
    from .test_set import load_test_set as _load
    return _load(split)


def load_test_set_raw(split: str = "standard") -> list[dict]:
    """Load a test set split with pre-rendered prompts from parquet.

    Returns list of dicts with keys: prompt (list of message dicts), example (Example).
    The prompts match the training data exactly, avoiding prompt drift.

    Requires: pandas, pyarrow
    """
    from .test_set import load_test_set_raw as _load
    return _load(split)
