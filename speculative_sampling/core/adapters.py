"""Small adapter contracts; no torch, transport, HTTP or model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .protocol import BoundaryReadiness, OutputGroup


@dataclass(frozen=True)
class SamplingCapabilities:
    """Model-native sampling semantics, independent of playback scheduling.

    ``requires_refill`` describes whether one native sampling request consumes
    retained streaming state.  It must not be used to decide whether a live
    player should extend its timeline: boundary-image models such as H3 do not
    refill model state, but still need just-in-time continuation for an
    unbounded session.
    """

    mode: str
    native_chunk_size: int | None
    requires_refill: bool
    refill_chunk_size: int | None = None

    def validate(self) -> None:
        if self.mode not in {"streaming", "bidirectional"}:
            raise ValueError("sampling mode must be streaming or bidirectional")
        if self.mode == "streaming" and self.native_chunk_size is None:
            raise ValueError("streaming sampling requires native_chunk_size")
        for name, value in (
            ("native_chunk_size", self.native_chunk_size),
            ("refill_chunk_size", self.refill_chunk_size),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when present")
        if self.mode == "bidirectional" and self.requires_refill:
            raise ValueError("bidirectional sampling cannot require retained-state refill")


@dataclass(frozen=True)
class LiveSchedulingPolicy:
    """Measured deployment policy, never a model capability declaration."""

    auto_extend_timeline: bool
    extension_mode: str
    condition_ready_to_first_commit_seconds: float
    continuation_to_first_commit_seconds: float
    ema_alpha: float
    ema_margin_seconds: float
    delivery_credit_frames: int
    rtc_jitter_target_seconds: float | None
    playhead_clock_mode: str = "acknowledged"
    playhead_feedback_timeout_seconds: float = 0.75
    rewrite_horizon_fallback: bool = False
    rewrite_readiness_window: int = 8
    rewrite_guard_frames: int = 1

    def validate(self) -> None:
        if self.extension_mode not in {"streaming_refill", "jit_continuation"}:
            raise ValueError("unknown live timeline extension mode")
        if min(self.condition_ready_to_first_commit_seconds,
               self.continuation_to_first_commit_seconds) <= 0:
            raise ValueError("measured first-commit latencies must be positive")
        if not 0 <= self.ema_alpha <= 1:
            raise ValueError("EMA alpha must be in [0, 1]; zero disables updates")
        if self.ema_margin_seconds < 0:
            raise ValueError("EMA margin must be non-negative")
        if self.delivery_credit_frames < 0:
            raise ValueError("delivery credit must be non-negative")
        if (self.rtc_jitter_target_seconds is not None
                and not 0 <= self.rtc_jitter_target_seconds <= 1):
            raise ValueError("RTC jitter target must be between zero and one second")
        if self.playhead_clock_mode not in {
            "acknowledged", "extrapolated", "transport_paced",
        }:
            raise ValueError(
                "playhead clock mode must be acknowledged, extrapolated, or transport_paced"
            )
        if not 0 < self.playhead_feedback_timeout_seconds <= 10:
            raise ValueError("playhead feedback timeout must be in (0, 10] seconds")
        if type(self.rewrite_horizon_fallback) is not bool:
            raise ValueError("rewrite horizon fallback must be a boolean")
        if type(self.rewrite_readiness_window) is not int or self.rewrite_readiness_window < 1:
            raise ValueError("rewrite readiness window must be a positive integer")
        if type(self.rewrite_guard_frames) is not int or self.rewrite_guard_frames < 0:
            raise ValueError("rewrite guard frames must be a non-negative integer")


@dataclass(frozen=True)
class ModelCapabilities:
    model_id: str
    conditioning: dict[str, bool]
    continuation_kind: str
    execution: str
    joint_chunk_size: int
    supported_parallelism: tuple[int, ...]
    source_geometry: dict[str, Any]
    decoder_grouping: dict[str, Any]
    valid_branch_policy: str
    can_fork_generator_state: bool
    can_fork_decoder_state: bool
    can_split_decode: bool
    condition_readiness: str
    initialization_policy: str
    mapping_version: str
    sampling: SamplingCapabilities
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.conditioning or any(
            not isinstance(name, str) or type(enabled) is not bool
            for name, enabled in self.conditioning.items()
        ):
            raise ValueError("conditioning capabilities must be explicit booleans")
        if self.joint_chunk_size < 1:
            raise ValueError("joint_chunk_size must be positive")
        if not self.supported_parallelism or any(value < 1 for value in self.supported_parallelism):
            raise ValueError("supported_parallelism must contain positive degrees")
        if self.execution == "split" and not self.can_split_decode:
            raise ValueError("split execution requires can_split_decode")
        self.sampling.validate()
        if not self.evidence:
            raise ValueError("capability declarations require evidence references")


class ConditionProvider(Protocol):
    def encode_condition(self, value: Any, version: int) -> Any:
        ...

    def is_ready(self, version: int) -> bool:
        ...


class GeneratorSession(Protocol):
    """Resident iterator owned by one live model session.

    The controller supplies only the accepted logical prefix and a ready
    condition. Native KV/cache promotion, discarded speculative suffixes,
    positions, ReRoPE, and RNG remain internal across ``advance`` calls.
    """

    capabilities: ModelCapabilities

    def query_boundary(self, requested: int, horizon: int) -> BoundaryReadiness:
        ...

    def advance(self, *, retain_until: int, condition: Any,
                condition_version: int, seed: int) -> Any:
        ...

    def close(self) -> None:
        ...


class DecoderSession(Protocol):
    def query_boundary(self, retain_until: int) -> BoundaryReadiness:
        ...

    def fork(self) -> "DecoderSession":
        ...

    def decode(self, source: Any, mapping: Sequence[OutputGroup]) -> Any:
        ...

    def promote(self) -> None:
        ...

    def invalidate(self, start: int) -> None:
        ...


class OutputMapper(Protocol):
    def map_native_output(self, source: Any, logical_start: int, logical_end: int) -> tuple[OutputGroup, ...]:
        ...


class ServingAdapter(Protocol):
    capabilities: ModelCapabilities
    condition_provider: ConditionProvider
    generator: GeneratorSession
    decoder: DecoderSession
    mapper: OutputMapper
