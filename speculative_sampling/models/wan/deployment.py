"""Wan CF++ T2V K3 live-deployment declaration."""

from pathlib import Path

from speculative_sampling.core import ModelCapabilities, SamplingCapabilities
from speculative_sampling.runtime.deployment import load_deployment


SPEC = load_deployment(Path(__file__).with_name("configs") / "cfpp_t2v_4step_k3_b300.json")

CAPABILITIES = ModelCapabilities(
    model_id=SPEC.id,
    conditioning=dict(SPEC.document["conditioning"]),
    continuation_kind="retained_kv",
    execution="monolithic",
    joint_chunk_size=3,
    supported_parallelism=(4,),
    source_geometry={"fps": 24, "resolution": [832, 480]},
    decoder_grouping={"kind": "native_causal", "delivery_groups": 1},
    valid_branch_policy="accepted_prefix_clean_commit",
    can_fork_generator_state=True,
    can_fork_decoder_state=True,
    can_split_decode=False,
    condition_readiness="encoded_handle_required",
    initialization_policy="text_only_then_steady",
    mapping_version="wan-cfpp-k3-v1",
    sampling=SamplingCapabilities(
        mode="streaming", native_chunk_size=3, requires_refill=True,
        refill_chunk_size=3,
    ),
    evidence=("speculative_sampling/models/wan/README.md",),
)

LIVE_POLICY = SPEC.policy

__all__ = ["CAPABILITIES", "LIVE_POLICY", "SPEC"]
