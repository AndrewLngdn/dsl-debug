#!/bin/bash
# =============================================================================
# Smoke Test — End-to-end validation of dsl-debug CLI on a fresh instance
# =============================================================================
#
# Run this on a fresh Vast.ai instance to verify the Docker image works.
# Every test should pass without manual intervention.
#
# Usage:
#   bash /workspace/dsl-debug/scripts/smoke_test.sh
#
# Expected: 2x GPU instance (A100/A6000/etc), andrewlngdn/dsl-debug-train image

set -euo pipefail

PASS=0
FAIL=0
SKIP=0
RESULTS=()

pass() { PASS=$((PASS+1)); RESULTS+=("✓ $1"); echo "✓ PASS: $1"; }
fail() { FAIL=$((FAIL+1)); RESULTS+=("✗ $1: $2"); echo "✗ FAIL: $1 — $2"; }
skip() { SKIP=$((SKIP+1)); RESULTS+=("- $1 (skipped)"); echo "- SKIP: $1"; }

echo "============================================================"
echo "  dsl-debug smoke test — $(date)"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# 1. CLI exists and is executable
# ------------------------------------------------------------------
echo "--- Test 1: CLI availability ---"
if command -v dsl-debug &>/dev/null; then
    pass "dsl-debug CLI found at $(which dsl-debug)"
else
    fail "dsl-debug CLI" "not found in PATH"
fi

# ------------------------------------------------------------------
# 2. Help text
# ------------------------------------------------------------------
echo ""
echo "--- Test 2: Help text ---"
if dsl-debug help 2>&1 | grep -q "Commands:"; then
    pass "dsl-debug help"
else
    fail "dsl-debug help" "no 'Commands:' in output"
fi

# ------------------------------------------------------------------
# 3. Status (should work even with no GPU processes)
# ------------------------------------------------------------------
echo ""
echo "--- Test 3: Status ---"
STATUS_OUT=$(dsl-debug status 2>&1 || true)
if echo "$STATUS_OUT" | grep -q "GPU Status\|Processes"; then
    pass "dsl-debug status"
else
    fail "dsl-debug status" "unexpected output"
fi

# ------------------------------------------------------------------
# 4. Python package importable
# ------------------------------------------------------------------
echo ""
echo "--- Test 4: Package import ---"
if python3 -c "from dsl_debug import DSLDebugEnv, load_test_set_raw, TOOL_SCHEMAS; print(f'OK: {len(TOOL_SCHEMAS)} tools, {len(load_test_set_raw(\"standard\"))} standard problems')" 2>&1; then
    pass "dsl_debug package import"
else
    fail "dsl_debug package import" "import failed"
fi

# ------------------------------------------------------------------
# 5. Unit tests (no GPU needed)
# ------------------------------------------------------------------
echo ""
echo "--- Test 5: Unit tests ---"
TEST_OUT=$(dsl-debug test -- --ignore=tests/test_server.py --ignore=tests/test_server_integration.py 2>&1 || true)
if echo "$TEST_OUT" | tail -5 | grep -q "passed"; then
    pass "dsl-debug test (unit)"
else
    fail "dsl-debug test" "tests failed"
fi

# ------------------------------------------------------------------
# 6. Quickstart (no GPU needed — oracle agent)
# ------------------------------------------------------------------
echo ""
echo "--- Test 6: Quickstart ---"
if dsl-debug quickstart 2>&1 | grep -q -i "correct\|success\|solved"; then
    pass "dsl-debug quickstart"
else
    fail "dsl-debug quickstart" "quickstart did not report success"
fi

# ------------------------------------------------------------------
# 7. Training data exists
# ------------------------------------------------------------------
echo ""
echo "--- Test 7: Training data ---"
SFT_COUNT=$(python3 -c "import pandas; print(len(pandas.read_parquet('/workspace/dsl_debug_data/sft/train.parquet')))" 2>/dev/null || echo "0")
RL_COUNT=$(python3 -c "import pandas; print(len(pandas.read_parquet('/workspace/dsl_debug_data/rl/train.parquet')))" 2>/dev/null || echo "0")
if [ "$SFT_COUNT" -gt 0 ] && [ "$RL_COUNT" -gt 0 ]; then
    pass "Training data: SFT=${SFT_COUNT} rows, RL=${RL_COUNT} rows"
elif [ ! -f /workspace/dsl_debug_data/sft/train.parquet ]; then
    skip "Training data (not present — Docker image only)"
else
    fail "Training data" "SFT=${SFT_COUNT}, RL=${RL_COUNT} — expected >0"
fi

# ------------------------------------------------------------------
# 8. Test set splits loadable
# ------------------------------------------------------------------
echo ""
echo "--- Test 8: Test set splits ---"
BENCH_OK=true
for split in standard nonlocal intent_mismatch; do
    COUNT=$(python3 -c "from dsl_debug import load_test_set_raw; print(len(load_test_set_raw('${split}')))" 2>/dev/null || echo "0")
    if [ "$COUNT" -gt 0 ]; then
        echo "  ${split}: ${COUNT} problems"
    else
        BENCH_OK=false
        echo "  ${split}: FAILED (0 problems)"
    fi
done
if $BENCH_OK; then
    pass "All test set splits loadable"
else
    fail "Test set splits" "one or more splits returned 0"
fi

# ------------------------------------------------------------------
# 9. GPU check
# ------------------------------------------------------------------
echo ""
echo "--- Test 9: GPU availability ---"
if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
else
    GPU_COUNT=0
fi
if [ "$GPU_COUNT" -ge 2 ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    pass "GPUs: ${GPU_COUNT}x ${GPU_NAME}"
else
    skip "GPU check (${GPU_COUNT} GPUs, need 2 for sglang TP=2)"
fi

# ------------------------------------------------------------------
# 10. Download model from HF and launch sglang
# ------------------------------------------------------------------
echo ""
echo "--- Test 10: Model download + sglang server ---"
if [ "$GPU_COUNT" -ge 2 ]; then
    dsl-debug kill 2>/dev/null || true
    sleep 2

    echo "  Downloading model from HuggingFace and starting sglang..."
    if dsl-debug sglang 2>&1 | tee /tmp/sglang_test.log | tail -3; then
        if curl -s http://localhost:30000/health | grep -q -i "ok\|true"; then
            pass "sglang server (auto-download from HF)"
        else
            fail "sglang server" "started but health check failed"
        fi
    else
        fail "sglang server" "failed to start — see /tmp/sglang_test.log"
    fi
else
    skip "sglang server (no GPUs)"
fi

# ------------------------------------------------------------------
# 11. Single problem eval (quick sanity check)
# ------------------------------------------------------------------
echo ""
echo "--- Test 11: Single problem eval ---"
if curl -s http://localhost:30000/health 2>/dev/null | grep -q -i "ok\|true"; then
    RESULT=$(python3 -c "
import asyncio, openai, json
from dsl_debug import load_test_set_raw, DSLDebugEnv, TOOL_SCHEMAS, Example

# Load one problem
rows = load_test_set_raw('standard')
row = rows[0]
env = DSLDebugEnv(max_turns=8)
ex = Example(**{k: row[k] for k in ['program', 'expected_output', 'buggy_program', 'bug_description', 'error_type', 'difficulty']})
env.reset(ex)

print(f'Loaded problem: {ex.difficulty} difficulty')
print('Single-problem eval requires sglang — deferring to full eval test')
" 2>&1)
    echo "  $RESULT"
    pass "Problem loading and env reset"
else
    skip "Single problem eval (no sglang)"
fi

# ------------------------------------------------------------------
# 12. Full eval — standard split (THE key test)
# ------------------------------------------------------------------
echo ""
echo "--- Test 12: Full eval — standard split ---"
if curl -s http://localhost:30000/health 2>/dev/null | grep -q -i "ok\|true"; then
    echo "  Running full standard eval (481 problems, ~10 min)..."
    mkdir -p /workspace/smoke_test_results
    if dsl-debug eval --split standard --output /workspace/smoke_test_results/standard.json 2>&1 | tee /tmp/eval_standard.log | grep -E "TOTAL|Results"; then
        ACC=$(python3 -c "import json; print(f'{json.load(open(\"/workspace/smoke_test_results/standard.json\"))[\"accuracy\"]*100:.1f}')" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if float('${ACC}') > 80 else 1)" 2>/dev/null; then
            pass "Standard eval: ${ACC}% (expected ~86%)"
        else
            fail "Standard eval" "${ACC}% — expected >80%"
        fi
    else
        fail "Standard eval" "eval command failed"
    fi
else
    skip "Standard eval (no sglang)"
fi

# ------------------------------------------------------------------
# 13. Full eval — nonlocal split
# ------------------------------------------------------------------
echo ""
echo "--- Test 13: Full eval — nonlocal split ---"
if curl -s http://localhost:30000/health 2>/dev/null | grep -q -i "ok\|true"; then
    echo "  Running nonlocal eval (200 problems, ~8 min)..."
    if dsl-debug eval --split nonlocal --output /workspace/smoke_test_results/nonlocal.json 2>&1 | tee /tmp/eval_nonlocal.log | grep -E "TOTAL|Results"; then
        ACC=$(python3 -c "import json; print(f'{json.load(open(\"/workspace/smoke_test_results/nonlocal.json\"))[\"accuracy\"]*100:.1f}')" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if float('${ACC}') > 60 else 1)" 2>/dev/null; then
            pass "Nonlocal eval: ${ACC}% (expected ~70%)"
        else
            fail "Nonlocal eval" "${ACC}% — expected >60%"
        fi
    else
        fail "Nonlocal eval" "eval command failed"
    fi
else
    skip "Nonlocal eval (no sglang)"
fi

# ------------------------------------------------------------------
# 14. Full eval — intent_mismatch split
# ------------------------------------------------------------------
echo ""
echo "--- Test 14: Full eval — intent_mismatch split ---"
if curl -s http://localhost:30000/health 2>/dev/null | grep -q -i "ok\|true"; then
    echo "  Running intent_mismatch eval (177 problems, ~12 min)..."
    if dsl-debug eval --split intent_mismatch --output /workspace/smoke_test_results/intent_mismatch.json 2>&1 | tee /tmp/eval_intent.log | grep -E "TOTAL|Results"; then
        ACC=$(python3 -c "import json; print(f'{json.load(open(\"/workspace/smoke_test_results/intent_mismatch.json\"))[\"accuracy\"]*100:.1f}')" 2>/dev/null || echo "0")
        if python3 -c "exit(0 if float('${ACC}') > 20 else 1)" 2>/dev/null; then
            pass "Intent-mismatch eval: ${ACC}% (expected ~28%)"
        else
            fail "Intent-mismatch eval" "${ACC}% — expected >20%"
        fi
    else
        fail "Intent-mismatch eval" "eval command failed"
    fi
else
    skip "Intent-mismatch eval (no sglang)"
fi

# ------------------------------------------------------------------
# 15. Kill
# ------------------------------------------------------------------
echo ""
echo "--- Test 15: Kill ---"
KILL_OUT=$(dsl-debug kill 2>&1 || true)
if echo "$KILL_OUT" | grep -q "freed\|Done\|Stopping"; then
    sleep 3
    if ! curl -s http://localhost:30000/health 2>/dev/null | grep -q -i "ok\|true"; then
        pass "dsl-debug kill"
    else
        fail "dsl-debug kill" "server still responding after kill"
    fi
else
    fail "dsl-debug kill" "unexpected output"
fi

# ------------------------------------------------------------------
# 16. Val split eval (from file)
# ------------------------------------------------------------------
echo ""
echo "--- Test 16: Val split eval (from parquet file) ---"
if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  Restarting sglang for val eval..."
    dsl-debug sglang 2>&1 | tail -1
    if dsl-debug eval --split-file /workspace/dsl-debug/data/rl/val.parquet --output /workspace/smoke_test_results/val.json 2>&1 | grep -E "TOTAL|Results"; then
        ACC=$(python3 -c "import json; print(f'{json.load(open(\"/workspace/smoke_test_results/val.json\"))[\"accuracy\"]*100:.1f}')" 2>/dev/null || echo "0")
        pass "Val eval from file: ${ACC}%"
    else
        fail "Val eval from file" "eval command failed"
    fi
    dsl-debug kill 2>/dev/null || true
else
    skip "Val eval (no GPUs)"
fi

# ==================================================================
# Summary
# ==================================================================
echo ""
echo "============================================================"
echo "  SMOKE TEST RESULTS"
echo "============================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: ${PASS}  Failed: ${FAIL}  Skipped: ${SKIP}"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "SOME TESTS FAILED — check output above for details"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED"
    exit 0
fi
