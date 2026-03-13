# =============================================================================
# DSL Debug Environment — Training Image
# =============================================================================
#
# Everything needed to reproduce the paper results in one image:
#   - verl 0.7.0 + sglang 0.5.6 + torch 2.9.1 (CUDA 12.9)
#   - dsl-debug package, tool classes, reward function
#   - Training configs, scripts, `dsl-debug` CLI
#   - SFT + RL training data
#
# Models are downloaded on first use via `dsl-debug setup` (~14GB base model).
# This keeps the image small and avoids baking in weights that may change.
#
# Build:
#   docker buildx build --platform linux/amd64 -t andrewlngdn/dsl-debug-train --load .
#
# Use on Vast.ai (default image in vast.sh):
#   bash scripts/vast.sh create
#   vast.sh ssh
#   dsl-debug setup sft && dsl-debug train sft
#
# Use locally (if you have GPUs):
#   docker run --gpus all -it andrewlngdn/dsl-debug-train
#   dsl-debug setup sft && dsl-debug train sft
#
# Tested on: 2x A100-SXM4-80GB, NVIDIA driver >= 570

FROM verlai/verl:sgl056.latest

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# --- System deps ---
RUN apt-get update -qq && \
    apt-get install -y -qq rsync git && \
    rm -rf /var/lib/apt/lists/*

# --- verl 0.7.0 (base image already has torch, sglang, flashinfer) ---
RUN pip install --no-cache-dir "verl==0.7.0" --no-deps && \
    pip install --no-cache-dir cachetools

# Patch verl HTTP engine retry (TP>=2 needs longer sglang startup)
RUN VERL_DIR=$(python3 -c "import verl, os; print(os.path.dirname(verl.__file__))") && \
    sed -i 's/^DEFAULT_MAX_ATTEMPTS = 3$/DEFAULT_MAX_ATTEMPTS = 10/' \
        "${VERL_DIR}/workers/rollout/sglang_rollout/http_server_engine.py" && \
    sed -i 's/^DEFAULT_RETRY_DELAY = 2.0$/DEFAULT_RETRY_DELAY = 3.0/' \
        "${VERL_DIR}/workers/rollout/sglang_rollout/http_server_engine.py"

# --- Training + server deps ---
RUN pip install --no-cache-dir pandas pyarrow wandb huggingface_hub fastapi uvicorn click

# --- Project code ---
WORKDIR /workspace/dsl-debug
COPY . .
RUN pip install --no-cache-dir -e ".[cli]"

# --- PYTHONPATH so verl finds tool classes ---
ENV PYTHONPATH="/workspace/dsl-debug"

# --- Training data (SFT + RL parquets) ---
RUN mkdir -p /workspace/dsl_debug_data/sft /workspace/dsl_debug_data/rl && \
    cp data/sft/train.parquet /workspace/dsl_debug_data/sft/ && \
    cp data/sft/val.parquet /workspace/dsl_debug_data/sft/ && \
    cp data/rl/train.parquet /workspace/dsl_debug_data/rl/ && \
    cp data/rl/val.parquet /workspace/dsl_debug_data/rl/

WORKDIR /workspace

# --- Verify at build time ---
RUN python3 -c "\
import torch; print(f'torch {torch.__version__}'); \
import sglang; print(f'sglang {sglang.__version__}'); \
import verl; print(f'verl {verl.__version__}'); \
from verl.tools.base_tool import BaseTool; print('verl tools: OK'); \
import flash_attn; print(f'flash_attn {flash_attn.__version__}'); \
from dsl_debug import DSLDebugEnv, load_test_set_raw; print('dsl-debug: OK'); \
rows = load_test_set_raw('standard'); print(f'Test sets: {len(rows)} problems'); \
import pandas; df = pandas.read_parquet('/workspace/dsl_debug_data/rl/train.parquet'); \
print(f'RL data: {len(df)} examples'); \
print('All verified — run dsl-debug setup to download models')"

# --- Ensure SSH works on Vast.ai (sshd requires strict permissions) ---
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

CMD ["bash"]
