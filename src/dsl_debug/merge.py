"""FSDP checkpoint merge — converts verl FSDP shards to HuggingFace format."""

import gc
import os


def merge_fsdp_checkpoint(
    ckpt_dir: str,
    output_dir: str,
    base_model: str = "/workspace/models/Qwen2.5-7B-Instruct",
):
    """Merge 2-rank FSDP checkpoint shards into a single HuggingFace model.

    Loads torch/transformers lazily so that commands like ``dsl-debug status``
    never pay the CUDA initialisation cost.
    """
    import torch
    from torch.distributed._tensor import DTensor  # noqa: F401 — needed for unpickling

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"Loading shards from {ckpt_dir}...")
    s0 = torch.load(
        f"{ckpt_dir}/model_world_size_2_rank_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    s1 = torch.load(
        f"{ckpt_dir}/model_world_size_2_rank_1.pt",
        map_location="cpu",
        weights_only=False,
    )

    print(f"  {len(s0)} parameters, merging Shard(dim=0)...")
    merged = {}
    for key in list(s0.keys()):
        t0 = s0.pop(key)
        t1 = s1.pop(key)
        if hasattr(t0, "_local_tensor"):
            merged[key] = torch.cat([t0._local_tensor, t1._local_tensor], dim=0)
        else:
            merged[key] = t0
    del s0, s1

    hf_config_dir = os.path.join(ckpt_dir, "huggingface")
    print(f"Loading config from {hf_config_dir}...")
    config = AutoConfig.from_pretrained(hf_config_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if missing:
        print(f"  WARNING: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    del merged
    gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    del model

    # Copy tokenizer from base model (not verl's rewritten version which triggers warnings)
    print(f"  Loading tokenizer from {base_model}...")
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(
        output_dir
    )

    print("Verifying...")
    gc.collect()
    model2 = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
    print(f"Done! Parameters: {sum(p.numel() for p in model2.parameters()):,}")
