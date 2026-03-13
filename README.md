# DSL Debug

Training a 7B model to debug code through reinforcement learning. A custom dataflow DSL provides a controlled environment where the model learns to use tools (run code, inspect values, read docs, submit fixes) via SFT followed by GRPO.

Includes environment, data generation, training configs, evaluation harness, and pre-trained checkpoints. Base Qwen2.5-7B goes from 50% to 86% accuracy on held-out bugs.

## Results

Trained on Qwen2.5-7B-Instruct using GRPO ([verl](https://github.com/volcengine/verl) 0.7), evaluated one-shot on held-out test sets:

| Method | Standard (481) | Nonlocal (200) | Intent-Mismatch (177) |
|--------|:-:|:-:|:-:|
| Prompt Engineering (base) | 50.5% | 12.0% | 0.6% |
| SFT (step 100) | 56.3% | 40.0% | 7.9% |
| RL-only (step 30) | 78.8% | 54.0% | 14.7% |
| **SFT→RL (step 35)** | **86.1%** | **70.5%** | **28.2%** |

General capabilities preserved (small alignment tax): MMLU 74.5%, GSM8K 84.1%, HumanEval 62.2%.

## Quick Start

Everything runs from a single Docker image on a 2xA100 pod. No API keys required — all models and data are public.

```bash
# Rent a pod (or use andrewlngdn/dsl-debug-train on any provider)
https://cloud.vast.ai/cli/ # install Vast CLI
vastai set api-key YOUR_KEY 
vastai set ssh-key "$(cat ~/.ssh/id_rsa.pub)" # Or set in the Vast UI
bash scripts/vast.sh create          # cheapest 2xA100 (~$2.50/hr)
bash scripts/vast.sh ssh
```

```bash
# On the pod — verify, setup, train
dsl-debug test                       # run unit tests
dsl-debug setup base                 # download Qwen2.5-7B-Instruct (~14GB)
dsl-debug train sft                  # SFT warmup (~1.5 hrs)
```

```bash
# Continue with RL (from SFT checkpoint)
# Option A: use the SFT checkpoint you just trained (auto-merges FSDP → HF)
dsl-debug train sft-rl --model /workspace/checkpoints/global_step_100

# Option B: download pre-trained SFT checkpoint from HuggingFace
dsl-debug setup sft
dsl-debug train sft-rl
```

```bash
# Evaluate (after training completes)
dsl-debug merge <checkpoint_dir> /workspace/models/my_model
dsl-debug sglang /workspace/models/my_model
dsl-debug eval --split standard
dsl-debug eval --split nonlocal
dsl-debug eval --split intent_mismatch

# Or evaluate the pre-trained final model
dsl-debug setup final
dsl-debug sglang
dsl-debug eval --split standard
```

> **Note:** The tokenizer emits a warning about an "incorrect regex pattern" referencing Mistral. This is a false positive for Qwen2.5 and does not affect tokenization.

Validation is run **post-hoc** rather than during training. Training saves checkpoints every 5 steps (`SAVE_FREQ=5`) with in-training validation disabled (`TEST_FREQ=999`). After training, each checkpoint is merged and evaluated with `dsl-debug eval`. This is faster overall: training runs at full GPU utilization without interruption, and evaluation uses optimized inference (sglang with full GPU memory) rather than verl's built-in eval which shares GPU memory with the training process.

```bash
# Monitor and cleanup
bash scripts/vast.sh logs            # tail training logs
bash scripts/vast.sh status          # GPU utilization
bash scripts/vast.sh destroy         # clean up when done
```

### Pre-trained checkpoints

All checkpoints are public on HuggingFace:

| Model | Repo |
|-------|------|
| SFT→RL step 35 (best) | [`andrewlngdn/dsl-debug-7b-sft-rl`](https://huggingface.co/andrewlngdn/dsl-debug-7b-sft-rl) |
| SFT step 100 | [`andrewlngdn/dsl-debug-7b-sft-step100`](https://huggingface.co/andrewlngdn/dsl-debug-7b-sft-step100) |
| RL-only step 30 | [`andrewlngdn/dsl-debug-7b-rl-only-step30`](https://huggingface.co/andrewlngdn/dsl-debug-7b-rl-only-step30) |

## The Environment

### The DSL

A pipe-based dataflow language operating on tabular data:

```
sales = load("sales")
  |> filter(status == "completed")
  |> group_by(region)
  |> aggregate(revenue: sum(amount), count: count())
  |> sort_by(revenue, desc)
  |> take(5)
emit sales
```

Operations: `load`, `filter`, `select`, `group_by`+`aggregate`, `join`, `compute`, `sort_by`, `take`, `emit`.

### Bug categories

- **Standard**: Single-operator mutations (wrong column, wrong aggregation, syntax errors)
- **Nonlocal**: Bug in early pipe step, symptom hidden by downstream aggregation — requires `inspect` to trace
- **Intent-mismatch**: Multiple realistic mistakes (wrong approach, not just typos)

### Agent tools

The agent has 4 tools available via `<tool_call>` XML tags:

| Tool | Description |
|------|-------------|
| `run(code)` | Execute DSL code, see output or errors |
| `inspect(node_name)` | View intermediate output at a named variable |
| `read_docs(operation)` | Read documentation for a DSL operation |
| `submit(code)` | Submit corrected code (ends episode) |

### Local Development

For working outside Docker (reading code, running tests, exploring the environment):

```bash
uv sync                # creates .venv and installs everything
source .venv/bin/activate
dsl-debug test         # run unit tests
dsl-debug quickstart   # run an oracle agent episode
```

Or with pip: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[cli]"`

### Python API

The environment has zero dependencies and can be used standalone:

```python
from dsl_debug import DSLDebugEnv
from dsl_debug.data_generator import DataGenerator

gen = DataGenerator(seed=42)
example = gen.generate_example(difficulty="medium")

env = DSLDebugEnv(max_turns=8)
obs = env.reset(example)       # -> initial observation string
result = env.step(action_str)  # -> StepResult(observation, reward, done, info)
```

- `action_str` is the model's raw text output containing `<tool_call>` XML
- `reward` is 0.0 until `submit`, then 1.0 (correct) or 0.0 (wrong)
- `done` is True after `submit` or max turns reached

### HTTP Server

```bash
pip install -e ".[server]"   # installs fastapi + uvicorn
uvicorn server.app:app --port 8080
```

## Architecture

```
dsl-debug/
├── src/dsl_debug/           # Core package (zero deps)
│   ├── interpreter.py           # DAG-based dataflow DSL interpreter
│   ├── environment.py           # Gymnasium-style reset/step environment
│   ├── bug_injector.py          # Programmatic bug injection (5 categories)
│   ├── data_generator.py        # Random dataset + program generation
│   └── test_set.py              # Test set data loader
├── data/
│   ├── test_sets/               # Held-out evaluation splits (standard, nonlocal, intent_mismatch)
│   ├── rl/                      # GRPO training data
│   └── sft/                     # SFT training data
├── configs/                 # Training configs (SFT, GRPO)
├── tools/                   # verl tool classes + reward function
├── eval/                    # Evaluation harness
├── server/                  # FastAPI HTTP server + Docker
├── scripts/                 # CLI, vast.sh, build scripts
├── examples/                # Usage examples
└── tests/                   # Unit tests
```

## License

MIT
