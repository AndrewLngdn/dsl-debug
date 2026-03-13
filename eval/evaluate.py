"""Evaluation harness for the DSL Debug Environment.

Runs multi-turn debugging episodes against any OpenAI-compatible API endpoint.
Reports accuracy overall and by difficulty level.

Uses pre-rendered prompts from parquet test set files and handles tool calls
individually (matching val_standalone.py / verl training behavior exactly).

Usage:
    from dsl_debug.eval import run_evaluation

    results = run_evaluation(
        client=openai.AsyncOpenAI(base_url="http://localhost:8000/v1"),
        model="my-model",
        split="standard",
        max_concurrent=32,
    )

Or from the command line:
    python -m eval.evaluate --model my-model --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure dsl_debug is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsl_debug import DSLDebugEnv, Example, TOOL_SCHEMAS, load_test_set_raw
from dsl_debug.test_set import load_test_set_raw_from_file
from dsl_debug.environment import parse_tool_calls


def _execute_tool(env: DSLDebugEnv, name: str, args: dict):
    """Execute a single tool call. Returns (observation, done, correct)."""
    if name == "run":
        result = env._handle_run(args)
    elif name == "inspect":
        result = env._handle_inspect(args)
    elif name == "read_docs":
        result = env._handle_read_docs(args)
    elif name == "submit":
        result = env._handle_submit(args)
    else:
        from dsl_debug.environment import StepResult
        return f"Unknown tool: {name}", False, False

    correct = result.info.get("correct", False) if result.info else False
    return result.observation, result.done or env.done, correct


async def run_episode(
    client: Any,
    model: str,
    example: Example,
    messages_init: list[dict],
    idx: int,
    semaphore: asyncio.Semaphore,
    max_turns: int = 8,
    system_prompt: str | None = None,
    tokenizer: Any = None,
    stop_tokens: list[str] | None = None,
) -> dict:
    """Run one multi-turn episode.

    Uses pre-rendered messages from parquet (matching training prompts exactly).
    Handles each tool call individually with separate tool response messages,
    matching the verl training loop and val_standalone.py behavior.
    """
    async with semaphore:
        env = DSLDebugEnv(max_turns=max_turns)
        env.reset(example)

        # Use pre-rendered messages from parquet, optionally override system prompt
        messages = [dict(m) for m in messages_init]
        if system_prompt is not None:
            messages[0] = {"role": "system", "content": system_prompt}

        correct = False
        num_turns = 0
        submitted = False

        for turn in range(max_turns):
            num_turns += 1

            try:
                if tokenizer is not None:
                    # Text completions mode: format prompt locally (matches verl exactly)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tools=TOOL_SCHEMAS,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    response = await client.completions.create(
                        model=model,
                        prompt=prompt_text,
                        temperature=0,
                        max_completion_tokens=4096,
                        stop=stop_tokens or ["<|im_end|>"],
                    )
                    assistant_text = response.choices[0].text
                else:
                    # Chat completions mode
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        max_completion_tokens=4096,
                    )
                    msg = response.choices[0].message
                    assistant_text = msg.content or ""
                    # OpenAI returns structured tool_calls, not XML in content
                    if msg.tool_calls:
                        parsed = [
                            (tc.function.name, json.loads(tc.function.arguments))
                            for tc in msg.tool_calls
                        ]
                    else:
                        parsed = None
            except Exception as e:
                print(f"  [!] Problem {idx}: API error at turn {turn}: {e}")
                break

            # For text completions, parse XML tool calls from the response
            if tokenizer is not None:
                parsed = parse_tool_calls(assistant_text)

            if not parsed:
                # No tool calls — model stopped generating
                break

            if tokenizer is not None:
                # Text completions: raw text with XML tool calls
                messages.append({"role": "assistant", "content": assistant_text})
            else:
                # Chat completions: structured tool_calls from API
                messages.append(msg.model_dump(exclude_none=True))

            # Execute each tool call individually with separate tool messages
            for i, (name, args) in enumerate(parsed):
                obs, done, tc_correct = _execute_tool(env, name, args)

                if tokenizer is not None:
                    messages.append({"role": "tool", "content": obs})
                else:
                    # Chat API requires tool_call_id on tool responses
                    tc_id = msg.tool_calls[i].id if msg.tool_calls else f"call_{i}"
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs})

                if name == "submit":
                    submitted = True
                    correct = tc_correct
                    break

            if submitted or env.done:
                break

        return {
            "idx": idx,
            "correct": correct,
            "submitted": submitted,
            "num_turns": num_turns,
            "difficulty": example.difficulty,
            "domain": example.domain,
        }


async def run_evaluation(
    client: Any,
    model: str,
    split: str = "standard",
    system_prompt: str | None = None,
    max_turns: int = 8,
    max_concurrent: int = 16,
    tokenizer: Any = None,
    stop_tokens: list[str] | None = None,
    output_path: str | None = None,
    test_set_rows: list | None = None,
    benchmark_rows: list | None = None,
) -> dict:
    """Run evaluation on a test set split.

    Loads pre-rendered prompts from parquet files to ensure
    exact prompt matching with the training pipeline.
    """
    # Support old kwarg name
    if test_set_rows is None:
        test_set_rows = benchmark_rows
    if test_set_rows is None:
        test_set_rows = load_test_set_raw(split)

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        run_episode(
            client, model, row["example"], row["prompt"],
            idx, semaphore, max_turns,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            stop_tokens=stop_tokens,
        )
        for idx, row in enumerate(test_set_rows)
    ]

    print(f"Running {len(tasks)} problems (concurrency={max_concurrent}, max_turns={max_turns})...")
    start = time.time()

    results = []
    done_count = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done_count += 1
        if done_count % 20 == 0 or done_count == len(tasks):
            elapsed = time.time() - start
            correct_so_far = sum(1 for r in results if r["correct"])
            print(
                f"  [{done_count}/{len(tasks)}] "
                f"{correct_so_far}/{done_count} correct ({100*correct_so_far/done_count:.1f}%) "
                f"elapsed={elapsed:.0f}s"
            )

    elapsed = time.time() - start
    results.sort(key=lambda r: r["idx"])

    # Compute metrics by difficulty
    metrics = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            n_correct = sum(1 for r in subset if r["correct"])
            n_submitted = sum(1 for r in subset if r["submitted"])
            avg_turns = sum(r["num_turns"] for r in subset) / len(subset)
            metrics[diff] = {
                "accuracy": n_correct / len(subset),
                "n_correct": n_correct,
                "n_total": len(subset),
                "n_submitted": n_submitted,
                "avg_turns": avg_turns,
            }

    total_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    metrics["total"] = {
        "accuracy": total_correct / total if total else 0,
        "n_correct": total_correct,
        "n_total": total,
        "elapsed_s": elapsed,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({elapsed:.0f}s):")
    print(f"{'='*60}")
    for diff in ["easy", "medium", "hard"]:
        if diff in metrics:
            m = metrics[diff]
            print(
                f"  {diff:8s}: {m['accuracy']:6.1%} "
                f"({m['n_correct']}/{m['n_total']}) "
                f"submitted={m['n_submitted']} "
                f"avg_turns={m['avg_turns']:.1f}"
            )
    m = metrics["total"]
    print(f"  {'TOTAL':8s}: {m['accuracy']:6.1%} ({m['n_correct']}/{m['n_total']})")
    print(f"{'='*60}")

    output = {
        "accuracy": metrics["total"]["accuracy"],
        "metrics": metrics,
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def base_parser(description: str = "Evaluate a model on DSL Debug benchmarks"):
    """Create an argument parser with the standard eval arguments.

    Shared by eval/evaluate.py CLI and examples/evaluate_*.py scripts so
    argument definitions aren't duplicated.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name/path (auto-detected from sglang server if omitted)")
    parser.add_argument("--split", type=str, default="standard",
                        choices=["standard", "nonlocal", "intent_mismatch"],
                        help="Test set split")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to a parquet file to evaluate (overrides --split)")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    return parser


def main():
    parser = base_parser()
    parser.add_argument("--base-url", type=str, default="http://localhost:30000/v1",
                        help="sglang server URL")
    parser.add_argument("--api-key", type=str, default="none", help="API key")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Override system prompt (file path or inline text)")
    parser.add_argument("--chat-api", action="store_true",
                        help="Use chat completions API instead of text completions")
    args = parser.parse_args()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # Auto-detect model from sglang server if not specified
    if args.model is None:
        import json as _json
        import urllib.request
        try:
            with urllib.request.urlopen(f"{args.base_url}/models", timeout=5) as resp:
                models = _json.loads(resp.read()).get("data", [])
            if models:
                args.model = models[0]["id"]
                print(f"Auto-detected model: {args.model}")
            else:
                print("ERROR: No models found on server. Pass --model explicitly.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Could not auto-detect model from server: {e}")
            print("Pass --model explicitly or ensure sglang is running.")
            sys.exit(1)

    system_prompt = None
    if args.system_prompt:
        if Path(args.system_prompt).is_file():
            system_prompt = Path(args.system_prompt).read_text()
        else:
            system_prompt = args.system_prompt

    tokenizer = None
    stop_tokens = None

    if not args.chat_api:
        # Text completions with local tokenizer (default) — matches verl training exactly
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        stop_tokens = ["<|im_end|>"]
    else:
        # Chat completions API — may have tool format mismatches with sglang
        print(f"Using chat completions API (model={args.model})")

    # Load test set rows from file or named split
    test_set_rows = None
    if args.split_file:
        print(f"Loading split from file: {args.split_file}")
        test_set_rows = load_test_set_raw_from_file(args.split_file)
        split_name = Path(args.split_file).stem
    else:
        split_name = args.split

    asyncio.run(run_evaluation(
        client=client,
        model=args.model,
        split=split_name,
        system_prompt=system_prompt,
        max_turns=args.max_turns,
        max_concurrent=args.concurrency,
        output_path=args.output,
        tokenizer=tokenizer,
        stop_tokens=stop_tokens,
        test_set_rows=test_set_rows,
    ))


if __name__ == "__main__":
    main()
