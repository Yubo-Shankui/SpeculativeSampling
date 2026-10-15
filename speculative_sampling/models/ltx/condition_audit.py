"""Audit the actual native MTSS query mask without changing its time policy."""
from __future__ import annotations

import hashlib


def tensor_fingerprint(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_condition(batch, task, timeline, extras, *, include_fingerprints=False):
    """Evaluate representative first-spatial-token queries for all K latents.

    Text visibility is time-only, so spatial tokens at a given video time have
    identical masks. This calls the *native* bias function, not another mask
    implementation. It reports delayed activation instead of labeling all K
    outputs as seeing the newest condition.
    """
    import torch
    from vforge.models.ltx.temporal_prompt import compute_hard_temporal_bias
    from .speculative_sampling_native import latent_ranges

    inputs = batch.model_inputs
    start, count = int(task["retain_until"]), int(task["chunk_latents"])
    ranges = latent_ranges(batch)[start:start + count]
    if len(ranges) != count:
        raise ValueError("condition audit exceeds source geometry")
    lengths = [segment["video_context"].shape[1] for segment in timeline]
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    chunk_seconds = float(extras.get("hard_chunk_seconds", 0))
    records = {}
    for name, positions, indices in (
        ("video", inputs.video_positions, [r[0][0] for r in ranges]),
        ("audio", inputs.audio_positions, [r[1][0] for r in ranges]),
    ):
        queries = positions[0, 0, indices].float().mean(-1).unsqueeze(0)
        bias = compute_hard_temporal_bias(
            queries, inputs.context_positions, torch.float32,
            chunk_seconds=chunk_seconds)
        visible = bias[0, 0] == 0
        active = [
            [int(segment["version"]) for j, segment in enumerate(timeline)
             if visible[i, boundaries[j]:boundaries[j+1]].any().item()]
            for i in range(count)
        ]
        newest = int(task["prompt_version"])
        first = next((start+i for i, versions in enumerate(active) if newest in versions), None)
        records[name] = {
            "query_midpoints_seconds": queries[0].cpu().tolist(),
            "active_condition_versions": active,
            "first_new_condition_latent": first,
            "new_condition_visible": first is not None,
        }
    result = {
        "condition_version": int(task["prompt_version"]), "retain_until": start,
        "native_hard_chunk_seconds": chunk_seconds,
        "activation_seconds": float(timeline[-1]["activation"]),
        "video": records["video"], "audio": records["audio"],
        "new_condition_reaches_both_modalities": all(r["new_condition_visible"] for r in records.values()),
    }
    if include_fingerprints:
        result["fingerprints"] = {
            name: tensor_fingerprint(getattr(inputs, name))
            for name in ("reference_video", "video_context", "audio_context", "context_positions")
        }
    return result
