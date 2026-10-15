"""Non-invasive MiniMax H3 adapter.

H3 continuation needs no retained model KV.  A branch is represented by the
last full-resolution decoded image at a native causal boundary.  The next I2V
proposal repeats that boundary image as its first frame, so the mapper removes
the repeated one-frame anchor and publishes only new media.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from speculative_sampling import ContinuationKind, OutputGroup


FPS = 24
_SPAN_PATTERN = (1, 4, 4, 4, 4)


def boundary_image_shape(generation: Mapping[str, int]) -> tuple[int, int, int]:
    """Return the native RGB geometry used for H3 boundary re-encoding.

    Transport previews may be scaled down, but continuation always consumes
    the full-resolution decoded boundary image retained by the model plugin.
    """
    height = int(generation["height"])
    width = int(generation["width"])
    if height <= 0 or width <= 0:
        raise ValueError("generation width and height must be positive")
    return height, width, 3


def native_spans(frame_count: int) -> tuple[int, ...]:
    """Return H3 causal decode groups that exactly cover ``frame_count``."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    spans: list[int] = []
    covered = 0
    while covered < frame_count:
        span = _SPAN_PATTERN[len(spans) % len(_SPAN_PATTERN)]
        if covered + span > frame_count:
            raise ValueError("frame_count does not end at an H3 causal boundary")
        spans.append(span)
        covered += span
    return tuple(spans)


@dataclass(frozen=True)
class H3BoundaryImageAdapter:
    """Choose the first decoded RGB frame boundary after a deadline."""

    continuation_kind: ContinuationKind = ContinuationKind.BOUNDARY_IMAGE

    def select_retain_until(
        self, requested: int, candidates: Sequence[int], horizon: int
    ) -> int | None:
        return next((value for value in candidates if requested <= value <= horizon), None)


@dataclass(frozen=True)
class H3ProposalBoundaryAdapter:
    """Compatibility policy that only appends at the existing horizon."""

    continuation_kind: ContinuationKind = ContinuationKind.BOUNDARY_IMAGE

    def select_retain_until(
        self, requested: int, candidates: Sequence[int], horizon: int
    ) -> int | None:
        return horizon if requested <= horizon and horizon in candidates else None


@dataclass(frozen=True)
class H3ProposalMapping:
    groups: tuple[OutputGroup, ...]
    source_spans: tuple[tuple[int, int], ...]
    output_frame_end: int


@dataclass(frozen=True)
class H3OutputMapper:
    """Map one native H3 proposal onto the replaceable logical timeline.

    RGB re-encode can condition on any decoded frame, so logical positions are
    individual RGB frames. Native latent grouping remains an internal decode
    detail and does not constrain SpeculativeSampling boundaries.
    """

    fps: int = FPS

    def proposal(
        self,
        *,
        logical_start: int,
        output_frame_start: int,
        proposal_frames: int,
        continuation: bool,
    ) -> H3ProposalMapping:
        spans = native_spans(proposal_frames)  # validate native proposal geometry
        source_start = spans[0] if continuation else 0
        if continuation and source_start != 1:
            raise ValueError("H3 continuation must begin with a one-frame anchor")

        groups: list[OutputGroup] = []
        source_spans: list[tuple[int, int]] = []
        source_cursor = source_start
        output_cursor = output_frame_start
        logical = logical_start
        for _ in range(source_start, proposal_frames):
            source_end = source_cursor + 1
            output_end = output_cursor + 1
            groups.append(OutputGroup(
                start=logical,
                end=logical + 1,
                source_start=source_cursor,
                source_end=source_end,
                start_seconds=output_cursor / self.fps,
                end_seconds=output_end / self.fps,
                requires_audio=True,
            ))
            source_spans.append((source_cursor, source_end))
            logical += 1
            source_cursor = source_end
            output_cursor = output_end
        return H3ProposalMapping(tuple(groups), tuple(source_spans), output_cursor)
