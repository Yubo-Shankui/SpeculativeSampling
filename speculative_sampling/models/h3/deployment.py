"""MiniMax H3 live-deployment declaration."""

from pathlib import Path

from speculative_sampling.core import ModelCapabilities, SamplingCapabilities
from speculative_sampling.runtime.deployment import load_deployment


SPEC = load_deployment(
    Path(__file__).with_name("configs") / "boundary_image_5s_b300_sol_h3_vae.json"
)

CAPABILITIES = ModelCapabilities(
    model_id=SPEC.id,
    conditioning=dict(SPEC.document["conditioning"]),
    continuation_kind="boundary_image",
    execution="monolithic",
    joint_chunk_size=1,
    supported_parallelism=(8,),
    source_geometry={"fps": 24, "resolution": [832, 480]},
    decoder_grouping={"kind": "native_monolithic", "delivery_groups": 1},
    valid_branch_policy="rgb_reencode_any_frame",
    can_fork_generator_state=False,
    can_fork_decoder_state=False,
    can_split_decode=False,
    condition_readiness="native_multimodal",
    initialization_policy="reference_once_then_boundary_image",
    mapping_version="h3-rgb-frame-v1",
    sampling=SamplingCapabilities(
        mode="bidirectional", native_chunk_size=None, requires_refill=False,
    ),
    evidence=("tests/test_h3_adapter.py", "speculative_sampling/models/h3/README.md"),
)

LIVE_POLICY = SPEC.policy

__all__ = ["CAPABILITIES", "LIVE_POLICY", "SPEC"]
