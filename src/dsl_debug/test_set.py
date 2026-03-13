"""Test set data loader for the DSL debug environment.

Loads pre-generated test problems from parquet files bundled as package data.
Each problem contains a buggy DSL program, the correct version, datasets,
and expected output.

Requires: pandas, pyarrow
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .environment import Example


def _json_field(value: Any) -> Any:
    """Parse a JSON string if needed, otherwise return as-is."""
    return json.loads(value) if isinstance(value, str) else value


# Parquets live in data/test_sets/ at the project root
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test_sets"

_SPLIT_FILES = {
    "standard": "standard.parquet",
    "nonlocal": "nonlocal.parquet",
    "intent_mismatch": "intent_mismatch.parquet",
}


def _load_parquet(path):
    """Load a parquet file, raising a helpful error if pandas is missing."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas and pyarrow are required to load test sets. "
            "Install with: pip install pandas pyarrow"
        )
    return pd.read_parquet(path)


def _example_from_extra_info(row) -> Example:
    """Build an Example from a verl-format parquet row."""
    extra = row["extra_info"]
    tk = extra["tools_kwargs"]
    create_kwargs = next(iter(tk.values()))["create_kwargs"]

    return Example(
        datasets=_json_field(create_kwargs["datasets"]),
        correct_code=create_kwargs["correct_code"],
        buggy_code=create_kwargs["buggy_code"],
        expected_output=_json_field(create_kwargs["expected_output"]),
        bug_info=_json_field(create_kwargs.get("bug_info", {})),
        difficulty=create_kwargs.get("difficulty", "medium"),
        domain=create_kwargs.get("domain", "unknown"),
        num_nodes=create_kwargs.get("num_nodes", 0),
        has_join=create_kwargs.get("has_join", False),
    )


def _resolve_split(split: str) -> Path:
    """Resolve a split name to a parquet path, raising on invalid split."""
    if split not in _SPLIT_FILES:
        available = ", ".join(_SPLIT_FILES.keys())
        raise ValueError(f"Unknown split: '{split}'. Available: {available}")
    path = _DATA_DIR / _SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Test set file not found: {path}")
    return path


def load_test_set(split: str = "standard") -> list[Example]:
    """Load a test set split as a list of Example objects.

    Args:
        split: One of "standard", "nonlocal", "intent_mismatch"

    Returns:
        List of Example objects ready for env.reset()
    """
    path = _resolve_split(split)
    df = _load_parquet(path)
    return [_example_from_extra_info(row) for _, row in df.iterrows()]


def load_test_set_raw(split: str = "standard"):
    """Load a test set split as raw parquet rows with pre-rendered prompts.

    Returns:
        List of dicts with keys: prompt (list of message dicts), example (Example)
    """
    path = _resolve_split(split)
    df = _load_parquet(path)
    return [
        {
            "prompt": [dict(m) for m in row["prompt"]],
            "example": _example_from_extra_info(row),
        }
        for _, row in df.iterrows()
    ]


def load_test_set_raw_from_file(path: str):
    """Load an arbitrary parquet file as raw test set rows."""
    df = _load_parquet(path)
    return [
        {
            "prompt": [dict(m) for m in row["prompt"]],
            "example": _example_from_extra_info(row),
        }
        for _, row in df.iterrows()
    ]


def test_set_info() -> dict:
    """Get information about available test set splits."""
    info = {}
    for split, filename in _SPLIT_FILES.items():
        path = _DATA_DIR / filename
        if path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                info[split] = {"count": len(df), "path": str(path)}
            except (ImportError, FileNotFoundError):
                info[split] = {"count": "?", "path": str(path)}
        else:
            info[split] = {"count": 0, "path": str(path), "missing": True}
    return info
