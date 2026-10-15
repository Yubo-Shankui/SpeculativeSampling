"""Public, model-agnostic SpeculativeSampling API.

This package is deliberately free of model frameworks, networking, media
code, and deployment concerns.  Model integrations implement the protocols
exported here; applications compose those integrations with ``live_runtime``.
"""

from .core import (
    SpeculativeSamplingController,
    ConditionProvider,
    ControllerConfig,
    ControllerSnapshot,
    DecoderSession,
    GeneratorSession,
    ModelCapabilities,
    SamplingCapabilities,
    LiveSchedulingPolicy,
    OutputMapper,
    ServingAdapter,
    ContinuationKind,
    ExactBoundaryAdapter,
    GenerationCommand,
    GenerationReason,
    PredictionGroup,
    OutputGroup,
    BoundaryReadiness,
    SourceDependency,
)

__all__ = [
    "SpeculativeSamplingController",
    "ContinuationKind",
    "ControllerConfig",
    "ControllerSnapshot",
    "ExactBoundaryAdapter",
    "GenerationCommand",
    "GenerationReason",
    "PredictionGroup",
    "OutputGroup",
    "BoundaryReadiness",
    "SourceDependency",
    "ConditionProvider",
    "DecoderSession",
    "GeneratorSession",
    "ModelCapabilities",
    "SamplingCapabilities",
    "LiveSchedulingPolicy",
    "OutputMapper",
    "ServingAdapter",
]
