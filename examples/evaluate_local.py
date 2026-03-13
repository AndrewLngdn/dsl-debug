#!/usr/bin/env python3
"""Evaluate a local HuggingFace model using sglang + text completions.

This uses apply_chat_template locally for exact prompt formatting,
matching the training setup used with verl.

Requires: openai, pandas, pyarrow, transformers

Usage:
    # Start sglang server first:
    python -m sglang.launch_server --model my-model --port 30000

    # Then run evaluation:
    python examples/evaluate_local.py --model /path/to/model --port 30000
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from eval.evaluate import run_evaluation, base_parser


def main():
    parser = base_parser("Evaluate a local model on DSL Debug test sets")
    parser.add_argument("--port", type=int, default=30000, help="sglang server port")
    args = parser.parse_args()

    # Load tokenizer for local prompt formatting
    print(f"Loading tokenizer from {args.model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="none",
    )

    results = asyncio.run(run_evaluation(
        client=client,
        model=args.model,
        split=args.split,
        max_turns=args.max_turns,
        max_concurrent=args.concurrency,
        tokenizer=tokenizer,
        stop_tokens=["<|im_end|>"],
        output_path=args.output,
    ))

    print(f"\nFinal accuracy: {results['accuracy']:.1%}")


if __name__ == "__main__":
    main()
