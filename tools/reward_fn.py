"""Custom reward function for DSL debugger multi-turn tool training.

In multi-turn rollouts, the SubmitTool executes the model's corrected code
and returns a distinctive observation:
  - Correct: "Correct! Your fix produces the expected output."
  - Incorrect: "Incorrect. Your output: ..."

Since the decoded response text includes tool observations (with loss_mask=0),
we can detect correctness by checking for the success marker in the response.

Reward: binary 1.0 (correct) / 0.0 (incorrect).

Optional DAPO group filtering: drops zero-variance GRPO groups before
computing advantages, so gradients only flow from informative groups.

Environment variables (set in .env config files):
  DAPO_FILTER - "true"/"false" (default: true)

Used via verl's custom_reward_function config:
    custom_reward_function.path = tools/reward_fn.py
    custom_reward_function.name = compute_score
"""

import os
from collections import defaultdict

import torch

# Marker text from DSLDebugEnv._handle_submit() on correct submission
_CORRECT_MARKER = "Correct! Your fix produces the expected output."

_DAPO_FILTER = os.environ.get("DAPO_FILTER", "true").lower() == "true"

print(f"[reward_fn] Config: dapo_filter={_DAPO_FILTER}")


# =============================================================================
# DAPO-style group filtering for GRPO (applied at module load time)
# =============================================================================
# GRPO computes per-group advantages: (reward - group_mean) / group_std.
# When all rollouts in a group get the same reward (binary 0/1 is common),
# the advantage is zero for every sample → zero gradient → wasted compute.
#
# DAPO (https://arxiv.org/abs/2503.14476) proposes filtering: "keep sampling
# until the batch is fully filled with samples whose accuracy is neither 0
# nor 1." We implement the simpler variant: after rollouts, drop groups with
# zero reward variance and train on the remaining informative groups.

def _patch_dapo_group_filter():
    """Patch compute_advantage to filter zero-variance GRPO groups."""
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from verl.trainer.ppo.ray_trainer import compute_advantage as _original_compute_advantage

    def _filtered_compute_advantage(data, adv_estimator, **kwargs):
        from verl.trainer.ppo.core_algos import AdvantageEstimator

        # Only filter for GRPO
        if adv_estimator != AdvantageEstimator.GRPO:
            return _original_compute_advantage(data, adv_estimator, **kwargs)

        if "uid" not in data.non_tensor_batch:
            return _original_compute_advantage(data, adv_estimator, **kwargs)

        # Compute per-sample scalar rewards
        if "token_level_scores" in data.batch:
            scores = data.batch["token_level_scores"].sum(dim=-1)
        else:
            scores = data.batch["token_level_rewards"].sum(dim=-1)

        uids = data.non_tensor_batch["uid"]

        # Group scores by uid
        uid_to_indices = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)

        # Find groups with reward variance > 0
        keep_indices = []
        n_total_groups = len(uid_to_indices)
        n_all_correct = 0
        n_all_wrong = 0

        for uid, indices in uid_to_indices.items():
            group_scores = scores[indices]
            if torch.all(group_scores == group_scores[0]):
                if group_scores[0] > 0:
                    n_all_correct += 1
                else:
                    n_all_wrong += 1
            else:
                keep_indices.extend(indices)

        n_filtered = n_all_correct + n_all_wrong
        n_kept = n_total_groups - n_filtered

        print(
            f"[dapo_filter] Groups: {n_total_groups} total, "
            f"{n_kept} kept ({n_kept/max(n_total_groups,1)*100:.1f}%), "
            f"{n_all_correct} all-correct, {n_all_wrong} all-wrong "
            f"({n_filtered/max(n_total_groups,1)*100:.1f}% filtered)"
        )

        # If ALL groups are zero-variance, skip filtering to avoid empty batch
        if not keep_indices:
            print("[dapo_filter] WARNING: All groups zero-variance, using full batch")
            return _original_compute_advantage(data, adv_estimator, **kwargs)

        # Filter the batch and compute advantages
        return _original_compute_advantage(data[keep_indices], adv_estimator, **kwargs)

    # Guard against double-patching (reward_fn is loaded twice: train + val)
    if not getattr(ray_trainer_module.compute_advantage, '_dapo_patched', False):
        _filtered_compute_advantage._dapo_patched = True
        ray_trainer_module.compute_advantage = _filtered_compute_advantage
        print("[reward_fn] Patched compute_advantage with DAPO group filtering")
    else:
        print("[reward_fn] DAPO group filtering already patched, skipping")


if _DAPO_FILTER:
    try:
        _patch_dapo_group_filter()
    except Exception as e:
        print(f"[reward_fn] WARNING: Failed to patch DAPO filter: {e}")
else:
    print("[reward_fn] DAPO group filtering DISABLED")


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Check if the multi-turn response contains a correct submission.

    Args:
        data_source: Dataset identifier (e.g. "dsl_debug_hard").
        solution_str: Decoded response text (includes tool observations).
        ground_truth: The correct DSL code (for logging, not used for scoring).
        extra_info: Optional dict with num_turns etc.

    Returns:
        dict with 'score' (1.0 correct, 0.0 incorrect) and 'acc'.
    """
    if _CORRECT_MARKER in solution_str:
        return {"score": 1.0, "acc": 1.0}
    return {"score": 0.0, "acc": 0.0}
