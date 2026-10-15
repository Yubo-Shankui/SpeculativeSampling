import pytest

from speculative_sampling import ContinuationKind
from speculative_sampling.models.h3 import (
    H3BoundaryImageAdapter,
    H3OutputMapper,
    H3ProposalBoundaryAdapter,
    boundary_image_shape,
)


def test_boundary_image_selects_first_safe_rgb_frame():
    adapter = H3BoundaryImageAdapter()
    assert adapter.continuation_kind is ContinuationKind.BOUNDARY_IMAGE
    assert adapter.select_retain_until(4, (0, 3, 5, 8), 8) == 5
    assert adapter.select_retain_until(9, (0, 3, 5, 8), 8) is None
    base = H3ProposalBoundaryAdapter()
    assert base.select_retain_until(4, (0, 3, 5, 8), 8) == 8


def test_boundary_image_uses_native_generation_geometry_not_transport_preview():
    generation = {"width": 832, "height": 480}
    assert boundary_image_shape(generation) == (480, 832, 3)
    with pytest.raises(ValueError):
        boundary_image_shape({"width": 0, "height": 480})


def test_five_second_bootstrap_and_continuation_geometry():
    mapper = H3OutputMapper()
    first = mapper.proposal(
        logical_start=0, output_frame_start=0,
        proposal_frames=124, continuation=False,
    )
    assert len(first.groups) == 124
    assert first.output_frame_end == 124
    assert first.source_spans[0] == (0, 1)

    following = mapper.proposal(
        logical_start=124, output_frame_start=124,
        proposal_frames=124, continuation=True,
    )
    assert len(following.groups) == 123
    assert following.output_frame_end == 247
    assert following.source_spans[0] == (1, 2)
    assert following.groups[0].start_seconds == 124 / 24
    assert following.groups[-1].end_seconds == 247 / 24


def test_only_native_frame_counts_are_accepted():
    with pytest.raises(ValueError):
        H3OutputMapper().proposal(
            logical_start=0, output_frame_start=0,
            proposal_frames=2, continuation=False,
        )
