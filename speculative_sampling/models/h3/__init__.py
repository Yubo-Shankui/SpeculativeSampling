"""MiniMax H3 boundary-image plugin for SpeculativeSampling."""

from .adapter import (
    H3BoundaryImageAdapter,
    H3OutputMapper,
    H3ProposalBoundaryAdapter,
    boundary_image_shape,
)
from .session import H3SpeculativeSamplingLiveSession

__all__ = [
    "H3BoundaryImageAdapter",
    "H3ProposalBoundaryAdapter",
    "H3OutputMapper",
    "H3SpeculativeSamplingLiveSession",
    "boundary_image_shape",
]
