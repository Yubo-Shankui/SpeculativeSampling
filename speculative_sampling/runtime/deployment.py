"""Declarative model-variant configuration for the shared live runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from speculative_sampling.core import LiveSchedulingPolicy


REQUIRED_SECTIONS = {
    "model", "adapter", "conditioning", "sampling", "continuation",
    "decoder", "media", "deployment", "live_scheduling",
}

_POLICY_FIELDS = {
    "auto_extend_timeline",
    "extension_mode",
    "condition_ready_to_first_commit_seconds",
    "continuation_to_first_commit_seconds",
    "ema_alpha",
    "ema_margin_seconds",
    "delivery_credit_frames",
    "rtc_jitter_target_seconds",
    "playhead_clock_mode",
    "playhead_feedback_timeout_seconds",
    "rewrite_horizon_fallback",
    "rewrite_readiness_window",
    "rewrite_guard_frames",
}

_POLICY_DEFAULTS = {
    "playhead_clock_mode": "acknowledged",
    "playhead_feedback_timeout_seconds": 0.75,
    "rewrite_horizon_fallback": False,
    "rewrite_readiness_window": 8,
    "rewrite_guard_frames": 1,
}


@dataclass(frozen=True)
class LiveDeploymentSpec:
    path: Path
    document: dict[str, Any]
    policy: LiveSchedulingPolicy

    @property
    def id(self) -> str:
        return self.document["id"]

    @property
    def session_class(self) -> str:
        return self.document["adapter"]["session_class"]

    @property
    def worker_factory(self) -> str:
        return self.document["adapter"]["worker_factory"]

    @property
    def default_initial_budget_seconds(self) -> float:
        return float(self.document["live_scheduling"]["default_initial_budget_seconds"])

    def accepts(self, condition: str) -> bool:
        return bool(self.document["conditioning"].get(condition, False))


def load_deployment(path: str | Path) -> LiveDeploymentSpec:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("unsupported live deployment schema")
    if not isinstance(data.get("id"), str) or not data["id"]:
        raise ValueError("deployment id must be a non-empty string")
    missing = REQUIRED_SECTIONS - set(data)
    if missing:
        raise ValueError(f"deployment missing sections: {sorted(missing)}")
    adapter = data["adapter"]
    if not all(isinstance(adapter.get(key), str) and adapter[key]
               for key in ("session_class", "worker_factory")):
        raise ValueError("deployment adapter requires session_class and worker_factory")
    conditioning = data["conditioning"]
    if not all(type(value) is bool for value in conditioning.values()):
        raise ValueError("conditioning capabilities must be booleans")
    sampling = data["sampling"]
    chunk = sampling.get("native_chunk_latents")
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("native_chunk_latents must be positive")
    schedule = sampling.get("denoise_schedule")
    if schedule is not None and (not isinstance(schedule, list) or not schedule):
        raise ValueError("denoise_schedule must be a non-empty list")
    forwards = sampling.get("denoise_forwards")
    clean = sampling.get("clean_commit_forwards")
    steady = sampling.get("steady_dit_forwards")
    if any(type(value) is not int or value < 0 for value in (forwards, clean, steady)):
        raise ValueError("forward counts must be non-negative integers")
    if forwards + clean != steady:
        raise ValueError("steady_dit_forwards must equal denoise plus clean forwards")
    deployment = data["deployment"]
    if (type(deployment.get("gpu_count")) is not int
            or deployment["gpu_count"] < 1):
        raise ValueError("deployment gpu_count must be a positive integer")
    if not isinstance(deployment.get("roles"), dict) or not deployment["roles"]:
        raise ValueError("deployment requires explicit worker roles")
    decoder = data["decoder"]
    backend = decoder.get("backend", "original")
    if backend not in {"original", "sol-h3-vae"}:
        raise ValueError("decoder backend must be 'original' or 'sol-h3-vae'")
    available_backends = decoder.get("available_backends", [backend])
    if (not isinstance(available_backends, list) or not available_backends
            or any(item not in {"original", "sol-h3-vae"} for item in available_backends)
            or len(set(available_backends)) != len(available_backends)):
        raise ValueError("decoder available_backends must be a unique non-empty backend list")
    if backend not in available_backends:
        raise ValueError("default decoder backend must be listed in available_backends")
    if "sol-h3-vae" in available_backends:
        sol = decoder.get("sol_h3_vae")
        if not isinstance(sol, dict):
            raise ValueError("sol-h3-vae backend requires decoder.sol_h3_vae")
        for key in ("tae_weights", "causal_decoder_weights"):
            if not isinstance(sol.get(key), str) or not sol[key]:
                raise ValueError(f"sol-h3-vae requires a non-empty {key}")
        if "fa4_root" in sol and (not isinstance(sol["fa4_root"], str)
                                   or not sol["fa4_root"]):
            raise ValueError("sol-h3-vae fa4_root must be a non-empty string")
        if "cuda_graph" in sol and type(sol["cuda_graph"]) is not bool:
            raise ValueError("sol-h3-vae cuda_graph must be a boolean")
    live = data["live_scheduling"]
    initial_budget = live.get("default_initial_budget_seconds")
    if not isinstance(initial_budget, (int, float)) or initial_budget <= 0:
        raise ValueError("default_initial_budget_seconds must be positive")
    policy = LiveSchedulingPolicy(**{
        key: (live.get(key, _POLICY_DEFAULTS[key])
              if key in _POLICY_DEFAULTS else live[key])
        for key in _POLICY_FIELDS
    })
    policy.validate()
    return LiveDeploymentSpec(path.resolve(), data, policy)


def load_live_policy(path: str | Path) -> LiveSchedulingPolicy:
    return load_deployment(path).policy


def live_scheduling_status(policy: LiveSchedulingPolicy) -> dict[str, Any]:
    """Expose the same model-independent policy fields for every live app."""
    return {key: getattr(policy, key) for key in sorted(_POLICY_FIELDS)}
