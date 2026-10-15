"""Typed boundary between the SpeculativeSampling controller and native generators.

Positions are logical, half-open delivery-group indices. Model adapters own the
mapping from these positions to native latent, frame, block, or KV indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence


class ContinuationKind(str, Enum):
    """Native history representation exposed by a generator adapter."""

    RETAINED_KV = "retained_kv"
    INCREMENTAL_CLEAN_PREFILL = "incremental_clean_prefill"
    BOUNDARY_IMAGE = "boundary_image"
    BOUNDARY_LATENT = "boundary_latent"


class GenerationReason(str, Enum):
    BOOTSTRAP = "bootstrap"
    UPDATE = "update"
    BRIDGE = "bridge"
    REFILL = "refill"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class GenerationCommand:
    """One complete native K-group sampling request."""

    job_id: int
    reason: GenerationReason
    condition_version: int
    condition: Any
    retain_until: int
    chunk_size: int
    seed: int
    deadline_plan: tuple[float, ...] = ()


@dataclass(frozen=True)
class SourceDependency:
    """A decoded output may depend on another proposal's still-valid source."""

    job_id: int
    start: int
    end: int

    def __post_init__(self):
        if self.job_id < 1 or not 0 <= self.start < self.end:
            raise ValueError("invalid source dependency")


@dataclass(frozen=True)
class PredictionGroup:
    """One q-sized, ready-to-deliver prediction interval."""

    start: int
    end: int
    job_id: int
    condition_version: int
    payload: Any = None
    continuation_state: Any = None
    # Exclusive logical source dependency boundary. A 3-latent decoder window
    # can protect more than the one delivery group being committed.
    dependency_end: int | None = None
    source_dependencies: tuple[SourceDependency, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("prediction groups must be positive half-open intervals")
        if self.dependency_end is not None and self.dependency_end < self.end:
            raise ValueError("dependency_end cannot precede output end")


@dataclass(frozen=True)
class OutputGroup:
    """Adapter-owned source/output/time mapping; M need not equal native K."""

    start: int
    end: int
    source_start: int
    source_end: int
    start_seconds: float
    end_seconds: float
    requires_audio: bool = False
    dependency_end: int | None = None

    def __post_init__(self):
        import math
        if not (0 <= self.start < self.end and 0 <= self.source_start < self.source_end):
            raise ValueError("invalid output/source span")
        if not (math.isfinite(self.start_seconds) and math.isfinite(self.end_seconds)
                and 0 <= self.start_seconds < self.end_seconds):
            raise ValueError("invalid playback interval")
        if self.dependency_end is not None and self.dependency_end < self.end:
            raise ValueError("dependency_end cannot precede output end")


@dataclass(frozen=True)
class BoundaryReadiness:
    """Adapter evidence for history, decoder state, cover and per-group deadlines.

    Durations are relative to the coordinator's dispatch time, never timestamps
    subtracted across hosts. Unknown evidence is not feasible.
    """

    context_ready: bool
    decoder_ready: bool
    cover_ready: bool
    ready_after_seconds: tuple[float, ...] = ()

    def meets(self, deadlines: tuple[float, ...]) -> bool:
        import math
        if not (self.context_ready and self.decoder_ready and self.cover_ready):
            return False
        if deadlines and len(deadlines) != len(self.ready_after_seconds):
            return False
        if any(not math.isfinite(value) or value < 0 for value in self.ready_after_seconds):
            return False
        return all(
            math.isfinite(ready) and math.isfinite(deadline)
            and 0 <= ready <= deadline
            for ready, deadline in zip(self.ready_after_seconds, deadlines)
        )


class GeneratorBoundaryAdapter(Protocol):
    """Model-specific mapping from logical positions to native branch points."""

    continuation_kind: ContinuationKind

    def select_retain_until(
        self,
        requested: int,
        candidates: Sequence[int],
        horizon: int,
    ) -> int | None:
        """Return the earliest valid native boundary at or after ``requested``."""


@dataclass(frozen=True)
class ExactBoundaryAdapter:
    """Use prediction-group boundaries directly.

    LTX/Wan adapters will replace this with native KV/block mappings. H3 can
    use a boundary-image adapter without changing the controller.
    """

    continuation_kind: ContinuationKind = ContinuationKind.RETAINED_KV

    def select_retain_until(
        self,
        requested: int,
        candidates: Sequence[int],
        horizon: int,
    ) -> int | None:
        return next((value for value in candidates if requested <= value <= horizon), None)
