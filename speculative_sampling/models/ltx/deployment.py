"""LTX K15 ReRoPE live-deployment declaration."""

from pathlib import Path

from speculative_sampling.core import ModelCapabilities, SamplingCapabilities
from speculative_sampling.runtime.deployment import load_deployment


SPEC = load_deployment(Path(__file__).with_name("configs") / "dmd_k15_rerope_usp2_b300.json")

CAPABILITIES = ModelCapabilities(
    model_id=SPEC.id,
    conditioning=dict(SPEC.document["conditioning"]),
    continuation_kind="retained_kv",
    execution="split",
    joint_chunk_size=15,
    supported_parallelism=(2,),
    source_geometry={"fps": 24, "resolution": [832, 448], "steady_frames_per_latent": 8},
    decoder_grouping={
        "kind": "upstream_zero_lookahead", "delivery_groups": 1,
        "history_video_latents": 2,
    },
    valid_branch_policy="accepted_prefix_clean_commit",
    can_fork_generator_state=True,
    can_fork_decoder_state=True,
    can_split_decode=True,
    condition_readiness="encoded_handle_required",
    initialization_policy="reference_once_then_steady",
    mapping_version="ltx24-ref2av-v1",
    sampling=SamplingCapabilities(
        mode="streaming", native_chunk_size=15, requires_refill=True,
        refill_chunk_size=3,
    ),
    evidence=("speculative_sampling/models/ltx/README.md",),
)

LIVE_POLICY = SPEC.policy

__all__ = ["CAPABILITIES", "LIVE_POLICY", "SPEC"]
