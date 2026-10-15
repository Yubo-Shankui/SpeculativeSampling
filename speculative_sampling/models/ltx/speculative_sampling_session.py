"""LTX retained-KV plugin configuration for the shared live session."""

from speculative_sampling.runtime.retained_kv_session import (
    NativeChunkBoundaryAdapter,
    NativeSafePointAdapter,
    RetainedKVLiveSession,
)
from .boundary_adapter import LtxLiveBoundaryAdapter
from .deployment import CAPABILITIES, LIVE_POLICY
from .output_mapping import LtxOutputMapper


def rgb_boundary(latent_groups: int) -> int:
    """RGB frontier after ``latent_groups`` causal LTX video latents."""
    return 0 if latent_groups <= 0 else 1 + 8 * latent_groups


def latent_boundary(rgb_frames: int) -> int:
    if rgb_frames < 0:
        raise ValueError("RGB boundary must be non-negative")
    return 0 if rgb_frames < 9 else (rgb_frames - 1) // 8


class LtxSpeculativeSamplingLiveSession(RetainedKVLiveSession):
    """LTX mapping and split-service hooks over the common retained-KV flow."""

    chunk_latents = CAPABILITIES.sampling.native_chunk_size
    capabilities = CAPABILITIES

    mapper_factory = LtxOutputMapper
    split_boundary_factory = LtxLiveBoundaryAdapter
    delivery_packet_frames = 9
    live_policy = LIVE_POLICY

    @staticmethod
    def rgb_boundary(latent_groups):
        return rgb_boundary(latent_groups)

    @staticmethod
    def latent_boundary(rgb_frames):
        return latent_boundary(rgb_frames)


__all__ = [
    "LtxSpeculativeSamplingLiveSession",
    "NativeChunkBoundaryAdapter",
    "NativeSafePointAdapter",
    "latent_boundary",
    "rgb_boundary",
]
