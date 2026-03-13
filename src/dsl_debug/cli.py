"""dsl-debug — Click CLI for DSL Debug Environment."""

import os
import subprocess
import sys
import time
from pathlib import Path

import click

# Project root: two levels up from src/dsl_debug/cli.py
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _load_dotenv():
    """Load .env and pod-style credential files into os.environ."""
    env_file = os.path.join(PROJECT_ROOT, ".env")
    if os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    # Pod-style credential files
    for path, var in [
        ("/workspace/.wandb_key", "WANDB_API_KEY"),
        ("/workspace/.hf_token", "HF_TOKEN"),
    ]:
        if os.path.isfile(path):
            os.environ.setdefault(var, Path(path).read_text().strip())


def _load_config(config_file: str):
    """Load a KEY=VALUE config file into os.environ (overwriting)."""
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key.strip()] = value.strip()


def _download_model(repo: str, local_dir: str):
    """Download a HuggingFace model if not already present."""
    if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "config.json")):
        click.echo(f"Model: {local_dir} (exists)")
        return
    click.echo(f"Downloading {repo} → {local_dir} ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=local_dir)
    click.echo("Model downloaded.")


def _health_check(port: int) -> bool:
    """Check if sglang server is healthy on the given port."""
    import urllib.request

    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
        return True
    except Exception:
        return False


def _kill_by_pattern(pattern: str):
    """Kill all processes matching pattern."""
    try:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    except FileNotFoundError:
        pass


def _process_running(pattern: str) -> bool:
    """Check if a process matching pattern is running."""
    try:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _env(key: str, default: str = "") -> str:
    """Get env var with default."""
    return os.environ.get(key, default)


def _logger_arg() -> str:
    """Return the verl logger list based on WANDB_API_KEY availability."""
    if os.environ.get("WANDB_API_KEY"):
        return '["console","wandb"]'
    click.echo("NOTE: WANDB_API_KEY not set — logging to console only")
    return '["console"]'


# =============================================================================
# CLI group
# =============================================================================


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """dsl-debug — DSL Debug Environment CLI."""
    _load_dotenv()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# =============================================================================
# setup
# =============================================================================


@cli.command()
@click.argument("mode", default="sft")
def setup(mode):
    """Download models and verify dependencies.

    MODE: base | sft | final | all
    """
    click.echo("=== DSL Debug Environment Setup ===")

    # GPU check
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if out.returncode == 0:
            click.echo("GPU:")
            click.echo(out.stdout.strip())
        else:
            click.echo("No GPU detected (CPU-only mode)")
    except FileNotFoundError:
        click.echo("No GPU detected (CPU-only mode)")

    # CUDA 12.9 compat fix
    compat_dir = Path("/usr/local/cuda-12.9/compat")
    if compat_dir.is_dir() and list(compat_dir.glob("*.so*")):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True,
            )
            driver_major = int(out.stdout.strip().split(".")[0])
            if driver_major >= 570:
                click.echo(f"Removing CUDA 12.9 compat libs (driver {driver_major}.x is native)")
                for f in compat_dir.iterdir():
                    f.unlink(missing_ok=True)
                subprocess.run(["ldconfig"], check=False)
        except Exception:
            pass

    # Download models
    if mode == "base":
        _download_model("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/Qwen2.5-7B-Instruct")
    elif mode == "sft":
        _download_model("andrewlngdn/dsl-debug-7b-sft-step100", "/workspace/models/sft_7b_step100")
    elif mode == "final":
        _download_model("andrewlngdn/dsl-debug-7b-sft-rl", "/workspace/models/dsl-debug-7b-sft-rl")
    elif mode == "all":
        _download_model("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/Qwen2.5-7B-Instruct")
        _download_model("andrewlngdn/dsl-debug-7b-sft-step100", "/workspace/models/sft_7b_step100")
        _download_model("andrewlngdn/dsl-debug-7b-sft-rl", "/workspace/models/dsl-debug-7b-sft-rl")
    else:
        click.echo("Usage: dsl-debug setup <base|sft|final|all>")
        click.echo("")
        click.echo("  base    Qwen2.5-7B-Instruct — starting point for SFT or RL-only training")
        click.echo("  sft     SFT checkpoint (step 100) — for SFT→RL training, or serve/eval")
        click.echo("  final   SFT→RL model (best) — for inference or evaluation")
        click.echo("  all     All of the above")
        sys.exit(1)

    # Verify deps via inline Python checks
    checks = []
    try:
        import torch

        checks.append(
            f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}"
        )
    except ImportError:
        checks.append("torch: MISSING")
        click.echo("\n".join(f"  {c}" for c in checks))
        sys.exit(1)
    try:
        import verl

        checks.append(f"verl {verl.__version__}")
    except ImportError:
        checks.append("verl: MISSING (install with: pip install verl==0.7.0 --no-deps)")
    try:
        import sglang

        checks.append(f"sglang {sglang.__version__}")
    except ImportError:
        checks.append("sglang: MISSING")
    try:
        from dsl_debug import DSLDebugEnv  # noqa: F401

        checks.append("dsl-debug: OK")
    except ImportError:
        checks.append("dsl-debug: MISSING")
    try:
        import flash_attn

        checks.append(f"flash_attn {flash_attn.__version__}")
    except ImportError:
        checks.append("flash_attn: MISSING (optional)")

    for c in checks:
        click.echo(f"  {c}")
    click.echo("Setup complete.")


# =============================================================================
# train
# =============================================================================


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("mode", required=False)
@click.option("--model", "model_override", default=None, help="Override model path (auto-merges FSDP checkpoints)")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def train(mode, model_override, extra_args):
    """Run SFT or GRPO training.

    MODE: sft | grpo | rl-only | sft-rl | <config.env>
    """
    if not mode:
        click.echo("Usage: dsl-debug train <mode> [--model <path>]")
        click.echo("")
        click.echo("Modes:")
        click.echo("  sft       SFT on debugging trajectories")
        click.echo("  grpo      GRPO RL-only from base model")
        click.echo("  sft-rl    GRPO from SFT checkpoint")
        click.echo("  <file>    Custom config file")
        click.echo("")
        click.echo("Options:")
        click.echo("  --model   Override model path (auto-merges FSDP checkpoints)")
        sys.exit(1)

    # Resolve config file
    config_map = {
        "sft": os.path.join(PROJECT_ROOT, "configs", "sft_7b.env"),
        "grpo": os.path.join(PROJECT_ROOT, "configs", "grpo_rl_only.env"),
        "rl-only": os.path.join(PROJECT_ROOT, "configs", "grpo_rl_only.env"),
        "sft-rl": os.path.join(PROJECT_ROOT, "configs", "grpo_sft_rl.env"),
    }

    if mode in config_map:
        config_file = config_map[mode]
    elif mode.endswith(".env"):
        config_file = mode
    else:
        click.echo(f"ERROR: Unknown mode '{mode}'")
        click.echo("Valid: sft, grpo, sft-rl, or a .env config file")
        sys.exit(1)

    if not os.path.isfile(config_file):
        click.echo(f"ERROR: Config not found: {config_file}")
        sys.exit(1)

    click.echo(f"Loading config: {config_file}")
    _load_config(config_file)

    # Handle --model override (auto-merge FSDP if needed)
    if model_override:
        fsdp_shard = os.path.join(model_override, "model_world_size_2_rank_0.pt")
        config_json = os.path.join(model_override, "config.json")
        if os.path.isfile(fsdp_shard) and not os.path.isfile(config_json):
            merged_dir = f"/workspace/models/{os.path.basename(model_override)}"
            click.echo("FSDP checkpoint detected — merging to HF format...")
            from dsl_debug.merge import merge_fsdp_checkpoint

            merge_fsdp_checkpoint(model_override, merged_dir)
            os.environ["MODEL_PATH"] = merged_dir
        else:
            os.environ["MODEL_PATH"] = model_override

    # Detect SFT vs GRPO
    is_sft = mode == "sft" or "sft_7b" in config_file
    if is_sft:
        _run_sft(extra_args)
    else:
        _run_grpo(extra_args)


def _run_sft(extra_args):
    """Launch SFT training via torchrun (replaces this process)."""
    model_path = _env("MODEL_PATH", "/workspace/models/Qwen2.5-7B-Instruct")
    train_files = _env("TRAIN_FILES", _env("DATA_DIR", "/workspace/dsl_debug_data/sft") + "/train.parquet")
    val_files = _env("VAL_FILES", _env("DATA_DIR", "/workspace/dsl_debug_data/sft") + "/val.parquet")

    if not os.path.isfile(train_files):
        click.echo(f"ERROR: Training data not found: {train_files}")
        sys.exit(1)
    if not os.path.isdir(model_path):
        click.echo(f"ERROR: Model not found: {model_path}. Run: dsl-debug setup sft")
        sys.exit(1)

    # Count GPUs
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
        num_gpus = len(out.stdout.strip().splitlines()) if out.returncode == 0 else 1
    except FileNotFoundError:
        num_gpus = 1

    exp_name = _env("EXP_NAME", f"sft-{time.strftime('%Y%m%d-%H%M')}")
    os.environ["EXP_NAME"] = exp_name
    log_dir = f"/workspace/logs/{exp_name}"
    ckpt_dir = _env("CKPT_DIR", "/workspace/checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = _logger_arg()

    click.echo("==============================================")
    click.echo("SFT Training")
    click.echo("==============================================")
    click.echo(f"Model:      {model_path}")
    click.echo(f"Data:       {train_files}")
    click.echo(f"Epochs:     {_env('TOTAL_EPOCHS', '3')}")
    click.echo(f"Batch:      {_env('TRAIN_BSZ', '16')} (micro={_env('MICRO_BSZ', '4')})")
    click.echo(f"LR:         {_env('LR', '2e-5')}")
    click.echo(f"GPUs:       {num_gpus}")
    click.echo("==============================================")

    cmd = [
        "torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={num_gpus}",
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={_env('TRAIN_BSZ', '16')}",
        "data.multiturn.enable=true",
        "data.multiturn.messages_key=messages",
        f"data.micro_batch_size_per_gpu={_env('MICRO_BSZ', '4')}",
        f"data.max_length={_env('MAX_LENGTH', '4096')}",
        "data.truncation=right",
        f"model.partial_pretrain={model_path}",
        "model.enable_gradient_checkpointing=true",
        "model.trust_remote_code=true",
        "model.lora_rank=0",
        "model.strategy=fsdp",
        "model.fsdp_config.model_dtype=bf16",
        f"optim.lr={_env('LR', '2e-5')}",
        f"trainer.default_local_dir={ckpt_dir}",
        "trainer.project_name=dsl-debug",
        f"trainer.experiment_name={exp_name}",
        f"trainer.logger={logger}",
        "trainer.default_hdfs_dir=null",
        f"trainer.total_epochs={_env('TOTAL_EPOCHS', '3')}",
        f"trainer.test_freq={_env('VAL_FREQ', '20')}",
        f"trainer.save_freq={_env('SAVE_FREQ', '20')}",
        "trainer.max_ckpt_to_keep=-1",
        *extra_args,
    ]

    os.execvp(cmd[0], cmd)


def _run_grpo(extra_args):
    """Launch GRPO training via verl (replaces this process)."""
    # Set environment variables for GRPO
    os.environ.update({
        "RAY_DEDUP_LOGS": "0",
        "HYDRA_FULL_ERROR": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONWARNINGS": (
            "ignore::FutureWarning,"
            "ignore::UserWarning:verl.utils.tokenizer,"
            "ignore::UserWarning:verl.utils.profiler,"
            "ignore::UserWarning:sglang.srt.layers.quantization"
        ),
        "TOKENIZERS_PARALLELISM": "false",
        "RAY_DISABLE_DOCKER_CPU_WARNING": "1",
        "VERL_LOGGING_LEVEL": "CRITICAL",
        "TRANSFORMERS_VERBOSITY": "error",
    })
    cuda_home = _env("CUDA_HOME", "/usr/local/cuda")
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Set ulimit (best-effort)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    except Exception:
        pass

    model_path = _env("MODEL_PATH", "/workspace/models/Qwen2.5-7B-Instruct")
    if not os.path.isdir(model_path):
        hint = "sft-rl" if "sft" in model_path else "grpo"
        click.echo(f"ERROR: Model not found: {model_path}. Run: dsl-debug setup {hint}")
        sys.exit(1)

    import random

    exp_name = _env("EXP_NAME", "grpo_7b")
    run_id = f"{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 999999):06d}"
    log_dir = f"/workspace/logs/{run_id}"
    data_dir = _env("DATA_DIR", "/workspace/dsl_debug_data/rl")
    ckpt_dir = f"/workspace/checkpoints/dsl-debug/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    os.environ.update({
        "EXPERIMENT_NAME": exp_name,
        "RUN_ID": run_id,
        "ROLLOUT_LOG_DIR": log_dir,
        "METRICS_LOG_DIR": f"{log_dir}/metrics",
        "MODEL_PATH": model_path,
        "PYTHONPATH": f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}",
    })
    if not os.environ.get("DIAG_LOG_DIR"):
        os.environ["DIAG_LOG_DIR"] = ""

    logger = _logger_arg()
    tool_config = os.path.join(PROJECT_ROOT, "configs", "tool_config.yaml")
    reward_fn = os.path.join(PROJECT_ROOT, "tools", "reward_fn.py")

    click.echo("==========================================")
    click.echo(f"GRPO Training — {run_id}")
    click.echo(f"Model: {model_path}")
    click.echo(f"GPUs: {_env('N_GPUS', '2')} (TP={_env('TP_SIZE', '1')})")
    click.echo(f"Batch: {_env('TRAIN_BATCH_SIZE', '512')} x {_env('N_ROLLOUTS', '8')} rollouts")
    click.echo(f"LR: {_env('LEARNING_RATE', '2e-7')}")
    click.echo(f"Data: {data_dir}")
    click.echo("==========================================")

    micro_batch = _env("MICRO_BATCH_SIZE", "1")
    log_prob_micro = _env("LOG_PROB_MICRO_BATCH_SIZE", micro_batch)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator={_env('ADV_ESTIMATOR', 'grpo')}",
        f"data.train_files={data_dir}/{_env('TRAIN_FILE', 'train.parquet')}",
        f"data.val_files={data_dir}/val.parquet",
        f"data.train_batch_size={_env('TRAIN_BATCH_SIZE', '512')}",
        f"data.val_batch_size={_env('VAL_BATCH_SIZE', '16')}",
        "data.max_prompt_length=6144",
        "data.max_response_length=4096",
        "data.trust_remote_code=True",
        "data.return_raw_chat=True",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",
        "++data.seed=42",
        "++data.validation_shuffle=False",
        "actor_rollout_ref.hybrid_engine=True",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.use_remove_padding={_env('USE_REMOVE_PADDING', 'False')}",
        f"actor_rollout_ref.actor.optim.lr={_env('LEARNING_RATE', '2e-7')}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={_env('PPO_MINI_BATCH_SIZE', '512')}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={_env('PPO_MAX_TOKEN_LEN', '12288')}",
        f"actor_rollout_ref.actor.use_kl_loss={_env('USE_KL_LOSS', 'False')}",
        f"actor_rollout_ref.actor.entropy_coeff={_env('ENTROPY_COEFF', '0.0')}",
        f"actor_rollout_ref.actor.entropy_from_logits_with_chunking={_env('ENTROPY_CHUNKING', 'False')}",
        f"actor_rollout_ref.actor.entropy_checkpointing={_env('ENTROPY_CHECKPOINTING', 'False')}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_env('ACTOR_PARAM_OFFLOAD', 'True')}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_env('OPTIMIZER_OFFLOAD', 'True')}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={log_prob_micro}",
        "actor_rollout_ref.rollout.name=sglang",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={_env('GPU_MEM_UTIL', '0.4')}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={_env('TP_SIZE', '1')}",
        "actor_rollout_ref.rollout.max_model_len=12288",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro}",
        f"actor_rollout_ref.rollout.n={_env('N_ROLLOUTS', '8')}",
        f"actor_rollout_ref.rollout.multi_stage_wake_up={_env('MULTI_STAGE_WAKE_UP', 'True')}",
        f"actor_rollout_ref.rollout.free_cache_engine={_env('FREE_CACHE_ENGINE', 'True')}",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={_env('MAX_TURNS', '8')}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={tool_config}",
        "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable",
        f"custom_reward_function.path={reward_fn}",
        "custom_reward_function.name=compute_score",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"algorithm.kl_ctrl.type={_env('KL_CTRL_TYPE', 'fixed')}",
        f"algorithm.kl_ctrl.kl_coef={_env('KL_COEF', '0.0')}",
        "trainer.critic_warmup=0",
        f"trainer.logger={logger}",
        "trainer.project_name=dsl-debug",
        f"trainer.experiment_name={exp_name}",
        f"trainer.n_gpus_per_node={_env('N_GPUS', '2')}",
        "trainer.nnodes=1",
        f"trainer.val_before_train={_env('VAL_BEFORE_TRAIN', 'True')}",
        f"trainer.test_freq={_env('TEST_FREQ', '5')}",
        f"trainer.save_freq={_env('SAVE_FREQ', '3')}",
        f"trainer.max_actor_ckpt_to_keep={_env('MAX_CKPT_TO_KEEP', '-1')}",
        f"trainer.total_epochs={_env('TOTAL_EPOCHS', '6')}",
        f"trainer.validation_data_dir={log_dir}/val_generations",
        f"trainer.rollout_data_dir={log_dir}/rollouts",
        f"trainer.resume_mode={_env('RESUME_MODE', 'disable')}",
        f"trainer.default_local_dir={ckpt_dir}",
    ]

    # Conditional args (only added if env var is set — mirrors bash ${VAR:+...} pattern)
    conditional = [
        ("LR_WARMUP_STEPS", "actor_rollout_ref.actor.optim.lr_warmup_steps"),
        ("LR_SCHEDULER_TYPE", "actor_rollout_ref.actor.optim.lr_scheduler_type"),
        ("MIN_LR_RATIO", "actor_rollout_ref.actor.optim.min_lr_ratio"),
        ("FORWARD_PREFETCH", "actor_rollout_ref.actor.fsdp_config.forward_prefetch"),
        ("FSDP_MODEL_DTYPE", "actor_rollout_ref.actor.fsdp_config.model_dtype"),
        ("TOTAL_TRAINING_STEPS", "trainer.total_training_steps"),
    ]
    for env_key, hydra_key in conditional:
        val = os.environ.get(env_key)
        if val:
            cmd.append(f"{hydra_key}={val}")

    # Conditional ++ prefixed args
    conditional_pp = [
        ("KL_LOSS_COEF", "actor_rollout_ref.actor.kl_loss_coef"),
        ("KL_LOSS_TYPE", "actor_rollout_ref.actor.kl_loss_type"),
        ("LOSS_AGG_MODE", "actor_rollout_ref.actor.loss_agg_mode"),
        ("DETERMINISTIC_INFERENCE", "actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference"),
        ("ROLLOUT_SEED", "actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed"),
    ]
    for env_key, hydra_key in conditional_pp:
        val = os.environ.get(env_key)
        if val:
            cmd.append(f"++{hydra_key}={val}")

    cmd.extend(extra_args)

    os.execvp(cmd[0], cmd)


# =============================================================================
# test
# =============================================================================


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def test(extra_args):
    """Run pytest test suite."""
    os.chdir(PROJECT_ROOT)
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", *extra_args]
        ).returncode
    )


# =============================================================================
# eval
# =============================================================================


@cli.command("eval", context_settings=dict(ignore_unknown_options=True))
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def eval_cmd(args):
    """Run evaluation (requires sglang server)."""
    # Extract port from --base-url if present
    port = 30000
    for i, arg in enumerate(args):
        if arg == "--base-url" and i + 1 < len(args):
            import re

            m = re.search(r":(\d+)/", args[i + 1])
            if m:
                port = int(m.group(1))
            break

    if not _health_check(port):
        click.echo(f"ERROR: No sglang server on port {port}. Start one with: dsl-debug sglang <model>")
        sys.exit(1)

    os.chdir(PROJECT_ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
    sys.exit(
        subprocess.run(
            [sys.executable, "eval/evaluate.py", *args], env=env
        ).returncode
    )


# =============================================================================
# sglang
# =============================================================================


@cli.command()
@click.argument("model_path", required=False, default=None)
@click.option("--port", default=None, type=int, help="Server port (default: 30000)")
@click.option("--tp", default=2, type=int, help="Tensor parallelism (default: 2)")
@click.option("--mem-fraction", default=0.9, type=float, help="GPU memory fraction (default: 0.9)")
def sglang(model_path, port, tp, mem_fraction):
    """Launch sglang inference server."""
    if port is None:
        port = int(os.environ.get("PORT", "30000"))

    # Default: download best checkpoint from HF
    if model_path is None:
        model_path = "/workspace/models/dsl-debug-7b-sft-rl"
        if not os.path.isfile(os.path.join(model_path, "config.json")):
            click.echo("No model specified. Downloading best checkpoint (SFT->RL) from HuggingFace...")
            from huggingface_hub import snapshot_download

            snapshot_download(
                "andrewlngdn/dsl-debug-7b-sft-rl",
                local_dir=model_path,
                ignore_patterns=["global_step_*"],
            )
            click.echo("Download complete.")

    # Check if already running
    if _health_check(port):
        click.echo(f"sglang already running on port {port}")
        return

    click.echo("Launching sglang server...")
    click.echo(f"  Model: {model_path}")
    click.echo(f"  Port: {port}, TP: {tp}")

    env = os.environ.copy()
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--tp", str(tp),
            "--port", str(port),
            "--host", "0.0.0.0",
            "--mem-fraction-static", str(mem_fraction),
            "--max-total-tokens", "12288",
            "--trust-remote-code",
        ],
        stdout=open("/tmp/sglang_server.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )

    click.echo(f"Waiting for server to start (PID {proc.pid})...")
    for i in range(1, 181):
        if _health_check(port):
            click.echo(f"Server ready after {i}s")
            return
        if proc.poll() is not None:
            click.echo("ERROR: sglang died during startup. Last 30 lines:")
            click.echo(Path("/tmp/sglang_server.log").read_text().splitlines()[-30:])
            sys.exit(1)
        time.sleep(1)

    click.echo("ERROR: Server failed to start within 180s")
    lines = Path("/tmp/sglang_server.log").read_text().splitlines()[-20:]
    click.echo("\n".join(lines))
    sys.exit(1)


# =============================================================================
# merge
# =============================================================================


@cli.command()
@click.argument("ckpt_dir")
@click.argument("output_dir")
@click.argument("base_model", default="/workspace/models/Qwen2.5-7B-Instruct")
def merge(ckpt_dir, output_dir, base_model):
    """Merge FSDP checkpoint to HuggingFace format."""
    click.echo("Merging FSDP checkpoint...")
    click.echo(f"  Checkpoint: {ckpt_dir}")
    click.echo(f"  Output: {output_dir}")
    click.echo(f"  Base model: {base_model}")

    from dsl_debug.merge import merge_fsdp_checkpoint

    merge_fsdp_checkpoint(ckpt_dir, output_dir, base_model)


# =============================================================================
# status
# =============================================================================


@cli.command()
@click.argument("port", default=30000, type=int, required=False)
def status(port):
    """Show GPU + process status."""
    click.echo("=== GPU Status ===")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        click.echo(out.stdout.strip() if out.returncode == 0 else "  nvidia-smi not available")
    except FileNotFoundError:
        click.echo("  nvidia-smi not available")

    click.echo("")
    click.echo("=== Processes ===")

    for label, pattern in [
        ("sglang server", "sglang.launch_server"),
        ("GRPO training", "verl.trainer.main_ppo"),
        ("SFT training", "fsdp_sft_trainer"),
    ]:
        state = "RUNNING" if _process_running(pattern) else "not running"
        click.echo(f"  {label}: {state}")

    if _health_check(port):
        click.echo("")
        click.echo(f"  sglang health: OK (port {port})")
    click.echo("")


# =============================================================================
# serve
# =============================================================================


@cli.command()
@click.argument("model_path", required=False, default=None)
@click.option("--port", default=None, type=int)
def serve(model_path, port):
    """Start model server for inference (downloads best model if none specified)."""
    if port is None:
        port = int(os.environ.get("PORT", "30000"))

    click.echo("=== DSL Debug — Model Server ===")
    click.echo("")
    click.echo("This starts an sglang inference server with an OpenAI-compatible API.")
    click.echo("Use the /v1/completions endpoint for text completions.")
    click.echo("")

    # Build sglang args
    ctx = click.get_current_context()
    ctx.invoke(sglang, model_path=model_path, port=port)

    click.echo("")
    click.echo(f"Server running at http://0.0.0.0:{port}/v1")
    click.echo(f"  Health check: curl http://localhost:{port}/health")
    click.echo("  Kill: dsl-debug kill")
    click.echo("")
    click.echo("To evaluate: dsl-debug eval --split standard")


# =============================================================================
# kill
# =============================================================================


@cli.command()
def kill():
    """Stop sglang/training, free GPU memory."""
    click.echo("Stopping processes...")
    _kill_by_pattern("sglang.launch_server")
    _kill_by_pattern("sglang.srt")
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True, check=False)
    except FileNotFoundError:
        pass

    # Kill GPU-holding processes
    try:
        has_gpu = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        has_gpu = False

    if has_gpu:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        for pid_str in out.stdout.strip().splitlines():
            pid_str = pid_str.strip()
            if pid_str:
                try:
                    os.kill(int(pid_str), 9)
                except (ProcessLookupError, PermissionError):
                    pass

        _kill_by_pattern("verl.trainer")
        _kill_by_pattern("torchrun")

        # Wait for GPU memory to free
        for _ in range(30):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if out.returncode == 0:
                max_used = max(int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip())
                if max_used < 500:
                    click.echo("GPU memory freed")
                    out2 = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                        capture_output=True, text=True,
                    )
                    click.echo(out2.stdout.strip())
                    return
            time.sleep(2)

        click.echo("WARNING: GPU memory not fully freed")
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        click.echo(out.stdout.strip())

    click.echo("Done.")


# =============================================================================
# quickstart
# =============================================================================


@cli.command()
def quickstart():
    """Run quickstart example."""
    os.chdir(PROJECT_ROOT)
    sys.exit(subprocess.run([sys.executable, "examples/quickstart.py"]).returncode)


@cli.command("help", hidden=True)
@click.pass_context
def help_cmd(ctx):
    """Show help."""
    click.echo(ctx.parent.get_help())


if __name__ == "__main__":
    cli()
