"""HTTP server for the DSL Debug Environment.

Provides a stateful REST API for running multi-turn debugging episodes
over HTTP, making the environment accessible to any language or framework.

Endpoints:
    POST /reset     - Start a new episode
    POST /step      - Execute an action in an episode
    GET  /test_set/{split}  - List available problems
    GET  /health    - Health check

Usage:
    uvicorn server.app:app --port 8080
    # or: python -m server.app
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent to path so dsl_debug is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsl_debug import DSLDebugEnv, Example, load_test_set

app = FastAPI(
    title="DSL Debug Environment",
    description="Multi-turn debugging environment for LLM agents",
    version="0.1.0",
)

# In-memory episode store (stateful server)
_episodes: dict[str, dict] = {}

# Cache loaded test sets
_test_set_cache: dict[str, list[Example]] = {}


def _get_test_set(split: str) -> list[Example]:
    """Load and cache a test set split."""
    if split not in _test_set_cache:
        _test_set_cache[split] = load_test_set(split)
    return _test_set_cache[split]


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    problem_id: str  # format: "standard/42" or "nonlocal/0"
    max_turns: int = 8


class ResetResponse(BaseModel):
    episode_id: str
    observation: str


class StepRequest(BaseModel):
    episode_id: str
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict


class ProblemPreview(BaseModel):
    id: str
    difficulty: str
    domain: str
    has_join: bool
    num_nodes: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "active_episodes": len(_episodes)}


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest):
    """Start a new debugging episode.

    problem_id format: "{split}/{index}" e.g. "standard/42"
    """
    parts = req.problem_id.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(400, "problem_id must be '{split}/{index}', e.g. 'standard/42'")

    split, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        raise HTTPException(400, f"Invalid index: {idx_str}")

    try:
        problems = _get_test_set(split)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))

    if idx < 0 or idx >= len(problems):
        raise HTTPException(404, f"Index {idx} out of range [0, {len(problems)})")

    example = problems[idx]
    env = DSLDebugEnv(max_turns=req.max_turns)
    obs = env.reset(example)

    episode_id = str(uuid.uuid4())
    _episodes[episode_id] = {"env": env, "example": example}

    return ResetResponse(episode_id=episode_id, observation=obs)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    """Execute an action in an active episode.

    The action should be the model's raw text output containing
    <tool_call> XML tags.
    """
    ep = _episodes.get(req.episode_id)
    if ep is None:
        raise HTTPException(404, f"Episode not found: {req.episode_id}")

    env = ep["env"]
    result = env.step(req.action)

    # Clean up finished episodes to prevent memory leaks
    if result.done:
        _episodes.pop(req.episode_id, None)

    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/test_set/{split}", response_model=list[ProblemPreview])
async def list_test_set(split: str):
    """List all problems in a test set split."""
    try:
        problems = _get_test_set(split)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))

    return [
        ProblemPreview(
            id=f"{split}/{i}",
            difficulty=p.difficulty,
            domain=p.domain,
            has_join=p.has_join,
            num_nodes=p.num_nodes,
        )
        for i, p in enumerate(problems)
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
