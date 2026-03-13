"""
Integration tests for the DSL Debug HTTP server running in Docker.

Runs against the live container at BASE_URL. Tests:
  - All endpoints and HTTP status codes
  - All three test_set splits
  - Full episode lifecycle (inspect → run → submit correct/wrong)
  - Turn exhaustion
  - Stale episode cleanup
  - Concurrent episodes
  - All error paths (bad input, missing params, unknown IDs)
  - Sampling across multiple problems per split
"""

import json
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE_URL = "http://localhost:8080"
SPLITS = ["standard", "nonlocal", "intent_mismatch"]

# ─── HTTP helpers ────────────────────────────────────────────────────────────

def _request(method: str, path: str, body: Any = None) -> tuple[int, Any]:
    url = BASE_URL + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):    return _request("GET", path)
def post(path, body): return _request("POST", path, body)


def tool_call(name: str, **kwargs) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": kwargs})}</tool_call>'


def reset(problem_id: str, max_turns: int = 8):
    status, body = post("/reset", {"problem_id": problem_id, "max_turns": max_turns})
    return status, body


def step(episode_id: str, action: str):
    status, body = post("/step", {"episode_id": episode_id, "action": action})
    return status, body


# ─── Test runner ─────────────────────────────────────────────────────────────

@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def ok(self, name: str):
        self.passed.append(name)
        print(f"  ✓ {name}")

    def fail(self, name: str, reason: str):
        self.failed.append((name, reason))
        print(f"  ✗ {name}: {reason}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"Results: {len(self.passed)}/{total} passed")
        if self.failed:
            print("\nFailed:")
            for name, reason in self.failed:
                print(f"  ✗ {name}")
                print(f"    {reason}")
        return len(self.failed) == 0


R = Results()


def test(name: str, condition: bool, detail: str = ""):
    if condition:
        R.ok(name)
    else:
        R.fail(name, detail or "assertion failed")


def run_test(name: str):
    """Decorator to catch unexpected exceptions in a test block."""
    def decorator(fn):
        try:
            fn()
            R.ok(name)
        except AssertionError as e:
            R.fail(name, str(e))
        except Exception as e:
            R.fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    return decorator


# ─── Tests ───────────────────────────────────────────────────────────────────

print("\n=== Health & Benchmark ===")

@run_test("health check returns ok")
def _():
    status, body = get("/health")
    assert status == 200, f"status={status}"
    assert body["status"] == "ok", body
    assert "active_episodes" in body

@run_test("health: active_episodes is int")
def _():
    _, body = get("/health")
    assert isinstance(body["active_episodes"], int)

for split in SPLITS:
    @run_test(f"test_set/{split} returns list")
    def _(s=split):
        status, body = get(f"/test_set/{s}")
        assert status == 200, f"status={status}"
        assert isinstance(body, list), type(body)
        assert len(body) > 0, "empty test_set"

    @run_test(f"test_set/{split} problem schema")
    def _(s=split):
        _, body = get(f"/test_set/{s}")
        p = body[0]
        for key in ("id", "difficulty", "domain", "has_join", "num_nodes"):
            assert key in p, f"missing key: {key}"
        assert p["id"].startswith(s + "/"), p["id"]
        assert p["difficulty"] in ("easy", "medium", "hard"), p["difficulty"]
        assert isinstance(p["has_join"], bool)
        assert isinstance(p["num_nodes"], int)

@run_test("test_set/invalid_split returns 404")
def _():
    status, _ = get("/test_set/doesnotexist")
    assert status == 404, f"expected 404, got {status}"


print("\n=== Reset Errors ===")

@run_test("reset: bad problem_id format → 400")
def _():
    status, body = post("/reset", {"problem_id": "no-slash-here"})
    assert status == 400, f"got {status}"

@run_test("reset: non-integer index → 400")
def _():
    status, _ = post("/reset", {"problem_id": "standard/abc"})
    assert status == 400, f"got {status}"

@run_test("reset: negative index → 404")
def _():
    status, _ = post("/reset", {"problem_id": "standard/-1"})
    assert status == 404, f"got {status}"

@run_test("reset: out-of-range index → 404")
def _():
    status, _ = post("/reset", {"problem_id": "standard/999999"})
    assert status == 404, f"got {status}"

@run_test("reset: invalid split → 404")
def _():
    status, _ = post("/reset", {"problem_id": "fakesplit/0"})
    assert status == 404, f"got {status}"

@run_test("reset: returns episode_id and observation")
def _():
    status, body = reset("standard/0")
    assert status == 200, f"got {status}: {body}"
    assert "episode_id" in body
    assert "observation" in body
    assert "Buggy Code" in body["observation"]
    assert len(body["episode_id"]) > 0


print("\n=== Step Errors ===")

@run_test("step: unknown episode_id → 404")
def _():
    status, _ = post("/step", {"episode_id": "00000000-dead-beef-0000-000000000000",
                               "action": tool_call("run", code="emit x")})
    assert status == 404, f"got {status}"

@run_test("step: malformed tool call → no crash, reward=0")
def _():
    _, b = reset("standard/1")
    eid = b["episode_id"]
    status, body = step(eid, "this is not a tool call at all")
    assert status == 200, f"got {status}"
    assert body["reward"] == 0.0
    assert not body["done"]

@run_test("step: invalid JSON in tool call → no crash, reward=0")
def _():
    _, b = reset("standard/1")
    eid = b["episode_id"]
    status, body = step(eid, "<tool_call>not json</tool_call>")
    assert status == 200
    assert body["reward"] == 0.0

@run_test("step: unknown tool name → no crash, reward=0")
def _():
    _, b = reset("standard/1")
    eid = b["episode_id"]
    status, body = step(eid, tool_call("nonexistent_tool", foo="bar"))
    assert status == 200
    assert body["reward"] == 0.0

@run_test("step: inspect with missing node_name arg → no crash")
def _():
    _, b = reset("standard/1")
    eid = b["episode_id"]
    status, body = step(eid, '<tool_call>{"name": "inspect", "arguments": {}}</tool_call>')
    assert status == 200
    assert body["reward"] == 0.0


print("\n=== Full Episode Lifecycle ===")

@run_test("full: inspect → run → submit correct → reward=1.0, done=True")
def _():
    _, b = reset("standard/0")
    eid = b["episode_id"]

    # Step 1: inspect a node (should see data)
    _, r1 = step(eid, tool_call("inspect", node_name="filtered_behavior_incident"))
    assert not r1["done"]
    assert "incident" in r1["observation"].lower() or "Node" in r1["observation"]

    # Step 2: run the buggy code explicitly
    _, b2 = get("/test_set/standard")
    # find the correct_code via the environment — just use the known fix
    fix = '''-- Load behavior_monitoring__behavior_incident data
behavior_incident = load("behavior_incident")

filtered_behavior_incident = behavior_incident
  |> filter(incident_type_code == "NOISE")

summary = filtered_behavior_incident
  |> group_by(incident_type_code)
  |> aggregate(max_student_id: max(student_id), count: count())
  |> compute(computed_val: max_student_id * count)

result = summary
  |> sort_by(max_student_id, asc)
  |> take(5)

emit result'''

    status, r3 = step(eid, tool_call("submit", code=fix))
    assert status == 200, r3
    assert r3["reward"] == 1.0
    assert r3["done"]
    assert r3["info"]["correct"]

@run_test("full: submit wrong code → reward=0.0, done=True")
def _():
    _, b = reset("standard/0")
    eid = b["episode_id"]
    # Submit the buggy code as-is (should not match expected)
    wrong = '''behavior_incident = load("behavior_incident")
filtered_behavior_incident = behavior_incident
  |> filter(incident_type_code < "NOISE")
emit filtered_behavior_incident'''
    _, r = step(eid, tool_call("submit", code=wrong))
    assert r["done"]
    assert r["reward"] == 0.0
    assert not r["info"]["correct"]

@run_test("full: read_docs tool returns content")
def _():
    _, b = reset("standard/0")
    eid = b["episode_id"]
    _, r = step(eid, tool_call("read_docs", operation="filter"))
    assert not r["done"]
    assert r["reward"] == 0.0
    assert "filter" in r["observation"].lower()

@run_test("full: inspect invalid node name → error message, no crash")
def _():
    _, b = reset("standard/0")
    eid = b["episode_id"]
    _, r = step(eid, tool_call("inspect", node_name="this_node_does_not_exist"))
    assert not r["done"]
    assert r["reward"] == 0.0


print("\n=== Turn Exhaustion ===")

@run_test("turn exhaustion: done=True after max_turns")
def _():
    _, b = reset("standard/2", max_turns=3)
    eid = b["episode_id"]
    dummy = tool_call("inspect", node_name="zzz_fake")
    for i in range(3):
        _, r = step(eid, dummy)
        if i < 2:
            assert not r["done"], f"done too early at turn {i+1}"
    assert r["done"], "should be done after max_turns"

@run_test("turn exhaustion: step after done → 404 (episode cleaned up)")
def _():
    _, b = reset("standard/2", max_turns=1)
    eid = b["episode_id"]
    # Exhaust the one turn
    step(eid, tool_call("inspect", node_name="zzz_fake"))
    # Now the episode should be gone
    status, _ = step(eid, tool_call("inspect", node_name="zzz_fake"))
    assert status == 404, f"expected 404, got {status}"

@run_test("stale episode: done episode ID returns 404")
def _():
    _, b = reset("standard/3")
    eid = b["episode_id"]
    # Submit (terminates episode, server deletes it)
    step(eid, tool_call("submit", code="emit x"))
    status, _ = step(eid, tool_call("inspect", node_name="x"))
    assert status == 404, f"expected 404 for stale episode, got {status}"


print("\n=== Concurrent Episodes ===")

@run_test("concurrent: 5 independent episodes don't interfere")
def _():
    episodes = []
    for i in range(5):
        _, b = reset(f"standard/{i}")
        episodes.append((b["episode_id"], i))

    # Step each one differently and verify they're independent
    for eid, idx in episodes:
        _, r = step(eid, tool_call("inspect", node_name="zzz_fake"))
        assert not r["done"], f"episode {idx} ended unexpectedly"
        assert r["reward"] == 0.0

    # Health should show 5 active episodes
    _, h = get("/health")
    assert h["active_episodes"] >= 5, f"expected >=5 active, got {h['active_episodes']}"


print("\n=== Multi-Split Sampling ===")

# Sample a few problems from each split to ensure they load and run correctly
SAMPLE_INDICES = [0, 1, 5]

for split in SPLITS:
    _, problems = get(f"/test_set/{split}")
    n = len(problems)

    @run_test(f"{split}: test_set has problems")
    def _(s=split, count=n):
        assert count > 0

    for idx in SAMPLE_INDICES:
        if idx >= n:
            continue

        @run_test(f"{split}/{idx}: reset succeeds, observation looks right")
        def _(s=split, i=idx):
            status, b = reset(f"{s}/{i}")
            assert status == 200, f"got {status}"
            assert "Buggy Code" in b["observation"]
            assert "Expected Output" in b["observation"]
            assert "turns" in b["observation"].lower() or "turn" in b["observation"].lower()

        @run_test(f"{split}/{idx}: inspect first available node")
        def _(s=split, i=idx):
            _, b = reset(f"{s}/{i}")
            eid = b["episode_id"]
            # Extract node names from the observation (lines: "Available nodes: [...]")
            obs = b["observation"]
            import re
            m = re.search(r"Available nodes.*?\[([^\]]+)\]", obs)
            if m:
                nodes = [n.strip().strip("'\"") for n in m.group(1).split(",")]
                _, r = step(eid, tool_call("inspect", node_name=nodes[0]))
                assert r["reward"] == 0.0
                assert "output" in r["observation"].lower() or "Error" in r["observation"]


# ─── Final summary ────────────────────────────────────────────────────────────

success = R.summary()
sys.exit(0 if success else 1)
