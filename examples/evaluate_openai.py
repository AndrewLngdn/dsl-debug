#!/usr/bin/env python3
"""Evaluate any OpenAI-compatible model on the DSL Debug test sets.

Requires: openai, pandas, pyarrow
Install: pip install "dsl-debug[eval]"

Usage:
    # Against a local vLLM/sglang server:
    python examples/evaluate_openai.py --model my-model --base-url http://localhost:8000/v1

    # Against OpenAI API:
    python examples/evaluate_openai.py --model gpt-4o --api-key sk-...

    # Specific test set split:
    python examples/evaluate_openai.py --model my-model --split nonlocal
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from eval.evaluate import run_evaluation, base_parser


def main():
    parser = base_parser("Evaluate an OpenAI-compatible model on DSL Debug test sets")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "none")

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)

    results = asyncio.run(run_evaluation(
        client=client,
        model=args.model,
        split=args.split,
        max_turns=args.max_turns,
        max_concurrent=args.concurrency,
        output_path=args.output,
    ))

    print(f"\nFinal accuracy: {results['accuracy']:.1%}")


if __name__ == "__main__":
    main()
