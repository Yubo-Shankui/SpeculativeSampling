"""H3 web-session facade backed by the shared SpeculativeSampling controller."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, replace

from speculative_sampling import SpeculativeSamplingController, ControllerConfig, PredictionGroup
from speculative_sampling.runtime.estimator import ProductionEstimate
from .adapter import (
    H3BoundaryImageAdapter,
    H3OutputMapper,
    H3ProposalBoundaryAdapter,
    native_spans,
)


FPS = 24


@dataclass(frozen=True)
class LiveUnit:
    start: int
    end: int
    proposal: int
    latent_index: int
    source_start: int
    source_end: int
    prompt_version: int


class H3SpeculativeSamplingLiveSession:
    """Translate the H3 boundary-image protocol to SpeculativeSampling commands.

    The model remains a stateless I2V producer.  The only continuation state is
    the full-resolution final image at the selected causal output boundary.
    """

    # MiniMax-H3's VAE accepts 17*n+5 pixel frames.  Twenty-two frames is the
    # shortest non-degenerate native request and is the short-proposal study
    # arm; production defaults remain the validated 124/243-frame profiles.
    proposal_frames_allowed = (22, 124, 243)
    delivery_packet_frames = 8
    from .deployment import CAPABILITIES as capabilities

    def __init__(
        self, prompt: str, reference: str, *, seed: int = 42,
        mode: str = "image", initial_budget: float = 3.5,
        delivery_ahead_frames: int = 0,
        auto_extend_timeline: bool = True,
        extension_mode: str = "jit_continuation",
        condition_ready_to_first_commit_seconds: float = 1.85,
        continuation_to_first_commit_seconds: float | None = None,
        ema_alpha: float = 0.2, ema_margin_seconds: float = 0.0,
        playhead_clock_mode: str = "acknowledged",
        playhead_feedback_timeout_seconds: float = 0.75,
        rewrite_horizon_fallback: bool = False,
        rewrite_readiness_window: int = 8,
        rewrite_guard_frames: int = 1,
        proposal_frames: int = 124, serving_policy: str = "co", emit=None,
    ):
        if mode != "image":
            raise ValueError("the shared H3 adapter supports boundary-image mode only")
        if proposal_frames not in self.proposal_frames_allowed:
            raise ValueError(f"expected one of {self.proposal_frames_allowed} frames")
        if serving_policy not in ("base", "co"):
            raise ValueError("serving_policy must be base or co")
        if extension_mode not in ("streaming_refill", "jit_continuation"):
            raise ValueError("unknown timeline extension mode")
        if playhead_clock_mode not in ("acknowledged", "extrapolated", "committed"):
            raise ValueError("unknown playhead clock mode")
        if not 0 < playhead_feedback_timeout_seconds <= 10:
            raise ValueError("playhead feedback timeout must be in (0, 10] seconds")
        if type(rewrite_horizon_fallback) is not bool:
            raise ValueError("rewrite horizon fallback must be a boolean")
        if type(rewrite_readiness_window) is not int or rewrite_readiness_window < 1:
            raise ValueError("rewrite readiness window must be positive")
        if type(rewrite_guard_frames) is not int or rewrite_guard_frames < 0:
            raise ValueError("rewrite guard frames must be non-negative")
        if delivery_ahead_frames != 0:
            raise ValueError("H3 RTC commit is backend-paced; delivery credit must be zero")
        if condition_ready_to_first_commit_seconds <= 0 or (
            continuation_to_first_commit_seconds is not None
            and continuation_to_first_commit_seconds <= 0
        ):
            raise ValueError("first-commit latencies must be positive")
        if continuation_to_first_commit_seconds is None:
            continuation_to_first_commit_seconds = condition_ready_to_first_commit_seconds
        self.id = uuid.uuid4().hex
        self.reference = reference
        self.seed = int(seed)
        self.mode = mode
        self.serving_policy = serving_policy
        self.proposal_frames = int(proposal_frames)
        self.native_groups = len(native_spans(self.proposal_frames))
        self.mapper = H3OutputMapper()
        self.prompt = self._prompt(prompt)
        self.version = 1
        self.prompts = {1: self.prompt}
        self.next_prompts = {1: (self.prompt,)}
        now = time.monotonic()
        self.prompt_received_at = {1: now}
        self.condition_ready_at = {1: now}
        lead = self._groups_for_seconds(initial_budget)
        self.controller = SpeculativeSamplingController(
            self.prompt,
            ControllerConfig(
                chunk_size=self.proposal_frames,
                delivery_size=1,
                compute_lead=lead,
                refill_lead=lead,
                seed=self.seed,
            ),
            boundary_adapter=(
                H3BoundaryImageAdapter()
                if serving_policy == "co"
                else H3ProposalBoundaryAdapter()
            ),
            # The durable trace is written by ``self.event``.  This internal
            # list is only a bounded diagnostic cache; retaining every frame
            # commit makes status work grow with session age.
            event_history_limit=512,
        )
        self.initial_budget = float(initial_budget)
        self.delivery_ahead_frames = int(delivery_ahead_frames)
        self.auto_extend_timeline = bool(auto_extend_timeline)
        self.extension_mode = extension_mode
        self.condition_ready_to_first_commit_seconds = float(
            condition_ready_to_first_commit_seconds
        )
        self.continuation_to_first_commit_seconds = float(
            continuation_to_first_commit_seconds
        )
        self.condition_estimate = ProductionEstimate(
            self.condition_ready_to_first_commit_seconds,
            float(ema_margin_seconds), float(ema_alpha),
        )
        self.continuation_estimate = ProductionEstimate(
            self.continuation_to_first_commit_seconds,
            float(ema_margin_seconds), float(ema_alpha),
        )
        self.playhead_clock_mode = playhead_clock_mode
        self.playhead_feedback_timeout_seconds = float(
            playhead_feedback_timeout_seconds
        )
        self.rewrite_horizon_fallback = rewrite_horizon_fallback
        self.rewrite_readiness_window = rewrite_readiness_window
        self.rewrite_guard_frames = rewrite_guard_frames
        self.playhead_feedback_at = now
        # Freeze the projected cut when a condition becomes ready.  Dispatch
        # may be delayed by an already-running continuation; recomputing from
        # the later commit frontier would add that delay to prompt latency.
        self.rewrite_target_frame: int | None = None
        self.rewrite_boundary: int | None = None
        self.transport_rtt_s = 0.0
        self.durations: list[float] = []
        self.units: list[LiveUnit] = []
        self.busy: dict | None = None
        self.proposal_metrics = {}
        self.played = 0
        self.stopped = False
        self.error = None
        self.last_seen = now
        self.emit = emit or (lambda _event: None)
        self._mapping = {}
        self._published_source = {}
        self.event("session_start", mode=mode, prompt=self.prompt, seed=seed)

    @staticmethod
    def _prompt(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 5000:
            raise ValueError("prompt must contain 1–5000 characters")
        return value.strip()

    def _groups_for_seconds(self, seconds: float) -> int:
        return max(1, math.ceil(float(seconds) * FPS))

    def _frame_at(self, logical: int) -> int:
        if logical == 0:
            return 0
        # Proposals are immutable audit records, so an invalidated suffix can
        # share logical positions with its replacement while carrying the old
        # playback timestamps.  Only predictions describe the live branch.
        for group in self.controller.predictions:
            if group.end == logical:
                return group.payload.end
        if logical == self.controller.archived_until:
            return self.played
        raise ValueError(f"no H3 frame mapping for logical boundary {logical}")

    def _logical_at_or_before(self, frame: int) -> int:
        logical = self.controller.archived_until
        for group in self.controller.predictions:
            if group.payload.end <= frame:
                logical = max(logical, group.end)
        return logical

    def event(self, event_name: str, **fields):
        self.emit({
            "event": event_name, "session": self.id, "wall_time": time.time(),
            "monotonic": time.monotonic(), "played_frame": self.played,
            "committed_frame": self.committed, **fields,
        })

    @property
    def horizon(self):
        # Browser-visible cover is media-ready, not merely denoised source.
        logical = self.controller.snapshot().media_ready
        return self._frame_at(logical) if logical else 0

    @property
    def committed(self):
        return self._frame_at(self.controller.committed_until) if self.controller.committed_until else 0

    @property
    def applied_version(self):
        return self.controller.applied_condition_version

    @property
    def budget(self):
        margin = max(.45, self.transport_rtt_s + .20)
        if not self.durations:
            return self.initial_budget
        return max(1.0, max(self.durations[-8:]) * 1.15 + margin)

    def update_prompt(self, value: str, next_prompts=None):
        value = self._prompt(value)
        if next_prompts is None:
            next_prompts = (value,)
        if not isinstance(next_prompts, (list, tuple)) or not 1 <= len(next_prompts) <= 8:
            raise ValueError("next_prompts must contain 1–8 prompts")
        next_prompts = tuple(self._prompt(prompt) for prompt in next_prompts)
        if self.stopped:
            raise ValueError("session is stopped")
        self.last_seen = time.monotonic()
        if value == self.prompt:
            self.next_prompts[self.version] = next_prompts
            return self.version
        self.version = self.controller.update_condition(value)
        self.prompt = value
        self.prompts[self.version] = value
        self.next_prompts[self.version] = next_prompts
        now = time.monotonic()
        self.prompt_received_at[self.version] = now
        self.condition_ready_at[self.version] = now
        if self.serving_policy == "co":
            desired_target = self.committed + self._groups_for_seconds(
                self.condition_estimate.lead_seconds
            )
            self.rewrite_target_frame = self._feasible_rewrite_target(
                desired_target
            )
            self.rewrite_boundary = self._available_boundary_at_or_after(
                self.rewrite_target_frame
            )
        else:
            self.rewrite_target_frame = None
            self.rewrite_boundary = None
        self.event(
            "condition_ready", prompt_version=self.version,
            desired_target_frame=(desired_target if self.serving_policy == "co" else None),
            target_frame=self.rewrite_target_frame,
            reserved_boundary=self.rewrite_boundary,
        )
        self.event("prompt_received", prompt_version=self.version, prompt=value)
        return self.version

    def set_followup_prompt(self, version: int, value: str):
        """Compatibility alias for the legacy single-next-prompt API."""
        if self.stopped:
            raise ValueError("session is stopped")
        if version not in self.prompts:
            raise ValueError("prompt version does not exist")
        self.next_prompts[int(version)] = (self._prompt(value),)

    def _scheduling_playhead(self, now: float) -> int:
        """Estimate playback for refill scheduling without changing commits.

        Exact browser feedback remains authoritative for irreversible history
        and compaction. Between fresh feedback samples, advance a bounded
        24 FPS estimate and cap it at media already taken by the sender.
        """
        if self.playhead_clock_mode == "committed":
            return self.committed
        if self.playhead_clock_mode != "extrapolated":
            return self.played
        feedback_age = max(0.0, now - self.playhead_feedback_at)
        if feedback_age > self.playhead_feedback_timeout_seconds:
            return self.played
        # The acknowledged frame was sampled before its heartbeat traversed
        # the control path. Half the measured RTT estimates that transport age.
        feedback_age += min(0.5, max(0.0, self.transport_rtt_s) / 2)
        predicted = self.played + math.floor(feedback_age * FPS)
        return min(self.committed, predicted)

    def update_scheduling_lead(
        self, seconds: float | None = None, *,
        condition_seconds: float | None = None,
        continuation_seconds: float | None = None,
    ) -> dict:
        """Reset either or both H3 scheduling leads at runtime.

        ``lead_seconds`` remains the one-control compatibility path and writes
        both independent values.  The named values let a future UI expose two
        controls without changing or restarting the backend.
        """
        if seconds is not None:
            if condition_seconds is not None or continuation_seconds is not None:
                raise ValueError("lead_seconds cannot be combined with split lead values")
            condition_seconds = continuation_seconds = seconds
        if condition_seconds is None and continuation_seconds is None:
            raise ValueError("at least one scheduling lead is required")

        def validate(name: str, estimate: ProductionEstimate, raw: float | None):
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number") from exc
            margin = estimate.margin_seconds
            if not math.isfinite(value) or not 0.1 <= value <= 10.0:
                raise ValueError(f"{name} must be between 0.1 and 10.0")
            if value <= margin:
                raise ValueError(f"{name} must be greater than the EMA margin")
            return value - margin

        condition_value = validate(
            "condition_lead_seconds", self.condition_estimate, condition_seconds,
        )
        continuation_value = validate(
            "continuation_lead_seconds", self.continuation_estimate,
            continuation_seconds,
        )

        def reset(estimate: ProductionEstimate, value: float | None):
            if value is None:
                return estimate.lead_seconds
            estimate.initial_seconds = value
            estimate.value = value
            estimate.count = 0
            return estimate.lead_seconds

        previous_condition = self.condition_estimate.lead_seconds
        previous_continuation = self.continuation_estimate.lead_seconds
        condition_lead = reset(self.condition_estimate, condition_value)
        continuation_lead = reset(self.continuation_estimate, continuation_value)
        self.condition_ready_to_first_commit_seconds = self.condition_estimate.value
        self.continuation_to_first_commit_seconds = self.continuation_estimate.value
        self.event(
            "scheduling_lead_updated",
            previous_condition_lead_seconds=previous_condition,
            previous_continuation_lead_seconds=previous_continuation,
            condition_lead_seconds=condition_lead,
            continuation_lead_seconds=continuation_lead,
            effective_from="next_decision",
        )
        return {
            "lead_seconds": condition_lead if math.isclose(
                condition_lead, continuation_lead
            ) else None,
            "condition_lead_seconds": condition_lead,
            "continuation_lead_seconds": continuation_lead,
            "margin_seconds": self.condition_estimate.margin_seconds,
            "condition_observations": self.condition_estimate.count,
            "continuation_observations": self.continuation_estimate.count,
            "effective_from": "next_decision",
        }

    def _available_boundary_at_or_after(self, desired: int) -> int | None:
        """Resolve an RGB boundary from ready or currently-produced output."""
        eventual_horizon = self.horizon
        if self.busy is not None:
            eventual_horizon = max(
                eventual_horizon,
                self.busy["boundary"]
                + self.busy["proposal_frames"]
                - self.busy["trim_frames"],
            )
        return desired if desired <= eventual_horizon else None

    def _recent_first_ready_requirement(self) -> float | None:
        """Conservative warm first-media time used only for horizon fallback.

        The operator lead remains the normal target.  This measurement merely
        decides whether a prompt arriving one scheduler tick late can safely
        use the already-ready horizon instead of producing a full old-prompt
        bridge.  One video frame is retained as timing/rounding slack.
        """
        samples = [
            float(metric["first_ready_s"])
            for metric in list(self.proposal_metrics.values())[
                -self.rewrite_readiness_window:
            ]
            if metric.get("action") in ("update", "continuation", "refill")
            and metric.get("first_ready_s") is not None
        ]
        if not samples:
            return None
        return max(samples) + self.rewrite_guard_frames / FPS

    def _feasible_rewrite_target(self, desired: int) -> int:
        """Clamp a late rewrite to ready media when it can finish in time."""
        if (not self.rewrite_horizon_fallback or desired <= self.horizon
                or self.busy is not None):
            return desired
        remaining_seconds = max(0, self.horizon - self.committed) / FPS
        required_seconds = self._recent_first_ready_requirement()
        if (self.horizon > self.committed and required_seconds is not None
                and remaining_seconds >= required_seconds):
            self.event(
                "rewrite_target_clamped",
                desired_target_frame=desired,
                target_frame=self.horizon,
                remaining_seconds=remaining_seconds,
                required_seconds=required_seconds,
            )
            return self.horizon
        return desired

    def should_cancel_inflight(self, task: dict) -> bool:
        """Whether a superseded continuation is unnecessary for the safe cut.

        A continuation that is already generating the projected cut remains a
        useful bridge.  Cancelling it would force two serial generations and
        leave the media clock without cover.  An unpublished continuation
        starting at or after the projected cut can be cancelled immediately.
        """
        if self.busy != task or task.get("action") != "continuation":
            return False
        if self.version == task.get("prompt_version"):
            return False
        if self._published_source.get(task["proposal"], 0):
            return False
        target = self.rewrite_target_frame
        return target is None or target <= task["boundary"]

    def heartbeat(self, played: int):
        if not 0 <= played <= self.committed:
            raise ValueError("playhead is outside delivered content")
        self.played = max(self.played, int(played))
        logical = self._logical_at_or_before(self.played)
        if logical > self.controller.played_until:
            self.controller.advance_playback(logical)
        # Keep one second behind acknowledged browser playback for transient
        # WebSocket reconnects, then compact immutable coordinator history.
        # Without this, every status/frontier query scans the entire session.
        compact_until = self._logical_at_or_before(max(0, self.played - FPS))
        if compact_until > self.controller.archived_until:
            removed = set(self.controller.compact_played(compact_until))
            if removed:
                self._mapping = {
                    job: value for job, value in self._mapping.items()
                    if job not in removed
                }
                self._published_source = {
                    job: value for job, value in self._published_source.items()
                    if job not in removed
                }
            self.units = [group.payload for group in self.controller.predictions]
            retained_versions = set(self.controller.conditions)
            self.prompts = {
                version: value for version, value in self.prompts.items()
                if version in retained_versions
            }
            self.prompt_received_at = {
                version: value for version, value in self.prompt_received_at.items()
                if version in retained_versions
            }
            self.condition_ready_at = {
                version: value for version, value in self.condition_ready_at.items()
                if version in retained_versions
            }
            self.next_prompts = {
                version: value for version, value in self.next_prompts.items()
                if version in retained_versions
            }
        now = time.monotonic()
        self.playhead_feedback_at = now
        self.last_seen = now

    @property
    def delivery_resume_cursor(self) -> int:
        """Oldest retained frame from which a WebSocket can safely resume."""
        return self._frame_at(self.controller.archived_until)

    def plan(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        if self.busy or self.stopped or self.error or now - self.last_seen > 10:
            return None
        changed = self.controller.pending_condition_version is not None
        lead = self._groups_for_seconds(self.continuation_estimate.lead_seconds)
        refill_prompts = None
        refill_source_version = None
        if self.controller.horizon and not changed:
            if not self.auto_extend_timeline:
                return None
            if self.extension_mode == "jit_continuation":
                # Delivery to the transport is an irreversible commit, but it
                # is not playback.  Scheduling from committed_until makes a
                # fast WebSocket sender look empty and continuously generates
                # future media.  Refill from the browser playhead so the
                # configured lead is the actual unplayed coverage.
                # Use the identical integer frame budget as prompt rewrites:
                # rewrite chooses ``committed + lead_frames`` and continuation
                # starts at ``horizon - clock <= lead_frames``.
                continuation_lead_frames = self._groups_for_seconds(
                    self.continuation_estimate.lead_seconds
                )
                if (self.horizon - self._scheduling_playhead(now)
                        > continuation_lead_frames):
                    return None
                # Trigger exactly one low-water continuation. A completed H3
                # window adds ~5 seconds, so the exact frame gate closes again.
                lead = self.controller.horizon - self.controller.committed_until
            refill_source_version = self.applied_version
            refill_prompts = self.next_prompts.get(
                refill_source_version,
                (self.prompts[refill_source_version],),
            )
        # A prompt update only has to branch after browser-committed media.
        # Reusing the refill budget here needlessly bakes rare slow proposals
        # into every interaction (for example, a 3.5 s refill budget delays an
        # otherwise 1 s H3 update by 3.5 s).  The delivery window is the
        # irreversibility window for SpeculativeSampling; keep the conservative dynamic
        # budget exclusively for refill scheduling.
        interactive_update = changed and self.applied_version > 0
        if interactive_update and self.serving_policy == "co":
            if (self.rewrite_target_frame is None
                    or self.rewrite_target_frame < self.committed):
                desired_target = self.committed + self._groups_for_seconds(
                    self.condition_estimate.lead_seconds
                )
                self.rewrite_target_frame = self._feasible_rewrite_target(
                    desired_target
                )
                self.rewrite_boundary = self._available_boundary_at_or_after(
                    self.rewrite_target_frame
                )
            compute_lead = max(0, self.rewrite_target_frame - self.committed)
        else:
            compute_lead = lead
        if self.controller.horizon and self.serving_policy == "base" and changed:
            if (self.horizon - self.played) / FPS > self.budget + .5:
                return None
            # Base cannot rewrite buffered output. Once normal refill is due,
            # append at the existing horizon even if the ideal lead was missed.
            lead = min(
                lead, self.controller.horizon - self.controller.played_until,
            )
            compute_lead = lead
        self.controller.config = replace(
            self.controller.config,
            compute_lead=compute_lead,
            refill_lead=lead,
        )
        command = self.controller.dispatch(
            refill_condition=(refill_prompts[0] if refill_prompts else None),
        )
        if command is None:
            return None
        if refill_prompts is not None:
            refill_prompt = refill_prompts[0]
            remaining = refill_prompts[1:] or (refill_prompt,)
            if refill_source_version != command.condition_version:
                self.next_prompts.pop(refill_source_version, None)
            self.prompts[command.condition_version] = refill_prompt
            self.next_prompts[command.condition_version] = remaining
            self.prompt_received_at.setdefault(command.condition_version, now)
            self.condition_ready_at.setdefault(command.condition_version, now)
            self.version = max(self.version, command.condition_version)
            self.prompt = refill_prompt
        boundary_frame = self._frame_at(command.retain_until)
        if interactive_update and self.serving_policy == "co":
            self.rewrite_boundary = boundary_frame
        continuation = command.retain_until > 0
        mapping = self.mapper.proposal(
            logical_start=command.retain_until,
            output_frame_start=boundary_frame,
            proposal_frames=self.proposal_frames,
            continuation=continuation,
        )
        self._mapping[command.job_id] = mapping
        anchor = None
        if continuation:
            unit = next(
                group.payload for group in self.controller.predictions
                if group.end == command.retain_until
            )
            anchor = {
                "proposal": unit.proposal,
                "latent_index": unit.latent_index,
                "source_frame": unit.source_end - 1,
            }
        keep = {
            unit.proposal for unit in self.units
            if unit.end > max(0, self.played - FPS)
        }
        if anchor:
            keep.add(anchor["proposal"])
        task = {
            "session": self.id,
            "proposal": command.job_id,
            "action": (
                "continuation"
                if command.reason.value == "refill" and self.extension_mode == "jit_continuation"
                else command.reason.value
            ),
            "boundary": boundary_frame,
            "retain_until": command.retain_until,
            "anchor": anchor,
            "prompt": self.prompts[command.condition_version],
            "prompt_version": command.condition_version,
            "mode": "image",
            "seed": command.seed,
            "reference": self.reference,
            "keep_proposals": sorted(keep),
            "trim_frames": 1 if continuation else 0,
            "budget_s": self.budget,
            "transport_rtt_s": self.transport_rtt_s,
            "started_monotonic": now,
            "proposal_frames": self.proposal_frames,
            "serving_policy": self.serving_policy,
        }
        self.busy = task
        self.proposal_metrics[command.job_id] = {
            "proposal": command.job_id,
            "prompt_version": command.condition_version,
            "action": (
                "continuation"
                if command.reason.value == "refill" and self.extension_mode == "jit_continuation"
                else command.reason.value
            ),
            "started_monotonic": now,
            "queue_s": max(0, now - self.prompt_received_at[command.condition_version]),
            "first_ready_s": None,
            "first_commit_s": None,
            "total_s": None,
            "stages": {},
        }
        # Metrics are UI observability, not scheduling state.  Keep the same
        # bounded history as the legacy H3 runtime so long sessions do not
        # grow every 4 Hz status message indefinitely.
        for old in sorted(self.proposal_metrics)[:-24]:
            del self.proposal_metrics[old]
        self.units = [group.payload for group in self.controller.predictions]
        self.event("production_start", **{k: v for k, v in task.items() if k != "session"})
        return dict(task)

    def publish(self, task: dict, available_source: int):
        if self.stopped or self.busy != task:
            return False
        if not 0 <= available_source <= self.proposal_frames:
            raise ValueError("source availability is outside the H3 proposal")
        previous = self._published_source.get(task["proposal"], 0)
        if available_source < previous:
            raise ValueError("source availability must advance monotonically")
        mapping = self._mapping[task["proposal"]]
        proposal = self.controller.proposals[task["proposal"]]
        if not proposal.generated:
            self.controller.latents_ready(
                task["proposal"], mapping.groups, native_count=self.proposal_frames,
            )
        ready = []
        skipped = 1 if task["anchor"] else 0
        for offset, (group, source) in enumerate(zip(mapping.groups, mapping.source_spans)):
            if source[1] > available_source or source[1] <= previous:
                continue
            unit = LiveUnit(
                round(group.start_seconds * FPS), round(group.end_seconds * FPS),
                task["proposal"], offset + skipped,
                source[0], source[1], task["prompt_version"],
            )
            ready.append(PredictionGroup(
                group.start, group.end, task["proposal"], task["prompt_version"], payload=unit,
            ))
        self.controller.media_ready(task["proposal"], ready)
        self._published_source[task["proposal"]] = available_source
        self.units = [group.payload for group in self.controller.predictions]
        metric = self.proposal_metrics[task["proposal"]]
        if ready and metric["first_ready_s"] is None:
            metric["first_ready_s"] = time.monotonic() - task["started_monotonic"]
        self.event(
            "production_chunk", proposal=task["proposal"],
            boundary_frame=task["boundary"], source_end=available_source,
            horizon_frame=self.horizon,
            elapsed_s=time.monotonic() - task["started_monotonic"],
        )
        return True

    def complete(self, task: dict, elapsed: float):
        if self.stopped or self.busy != task:
            return False
        self.publish(task, self.proposal_frames)
        self.controller.finish(task["proposal"])
        self.durations.append(float(elapsed))
        del self.durations[:-16]
        metric = self.proposal_metrics[task["proposal"]]
        metric["total_s"] = time.monotonic() - task["started_monotonic"]
        observation = metric.get("first_ready_s")
        if observation is not None and task["action"] == "update":
            self.condition_estimate.observe(observation)
        elif observation is not None and task["action"] in ("continuation", "refill"):
            self.continuation_estimate.observe(observation)
        self.busy = None
        if self.applied_version == self.version:
            self.rewrite_target_frame = None
            self.rewrite_boundary = None
        self.event(
            "production_ready", proposal=task["proposal"],
            boundary_frame=task["boundary"], prompt_version=self.applied_version,
            elapsed_s=elapsed, horizon_frame=self.horizon,
        )
        return True

    def cancel(self, task: dict, reason: str = "superseded"):
        """Terminate one uncommitted JIT continuation without failing the session."""
        if self.busy != task or task.get("action") != "continuation":
            return False
        job_id = task["proposal"]
        self.controller.cancel_active(job_id, reason)
        self.busy = None
        self.units = [group.payload for group in self.controller.predictions]
        self.event(
            "production_cancelled", proposal=job_id, reason=reason,
            reserved_boundary=self.rewrite_boundary,
        )
        return True

    def record_metrics(self, task, stages, engine_elapsed=None, mock=False):
        metric = self.proposal_metrics.get(task["proposal"])
        if metric is not None:
            metric.update(stages=stages, engine_elapsed_s=engine_elapsed, mock=mock)
            self.event("production_metrics", **metric)

    def deliver(self, cursor: int, max_frames: int = 12):
        """Advance the backend-owned media frontier without browser gating."""
        if self.stopped or self.error:
            return []
        if not 0 <= cursor <= self.committed:
            raise ValueError("cursor must resume at or before the delivered frontier")
        boundaries = {0, *(unit.end for unit in self.units)}
        if cursor not in boundaries:
            raise ValueError("cursor must be a native causal boundary")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        limit = self.horizon
        replacement_publishing = bool(
            self.busy
            and self.busy["prompt_version"] == self.version
            and self._published_source.get(self.busy["proposal"], 0) > 0
        )
        if self.rewrite_boundary is not None and not replacement_publishing:
            limit = min(limit, self.rewrite_boundary)
        selected = []
        position = cursor
        while True:
            group = next(
                (g for g in self.controller.predictions if g.payload.start == position), None
            )
            if group is None or group.payload.end > limit:
                break
            if group.payload.end - cursor > max_frames:
                break
            if group.start == self.controller.committed_until:
                self.controller.deliver()
                metric = self.proposal_metrics.get(group.job_id)
                if metric and metric["first_commit_s"] is None:
                    metric["first_commit_s"] = time.monotonic() - metric["started_monotonic"]
            selected.append(group.payload)
            position = group.payload.end
        return selected

    def stop(self):
        self.stopped = True
        self.event("session_stop")

    def fail(self, message: str):
        self.error = str(message)
        if self.controller.active_job is not None:
            self.controller.job_failed(self.controller.active_job.job_id, message)
        self.busy = None
        self.event("error", message=str(message))

    def snapshot(self):
        snapshot_now = time.monotonic()
        controller_snapshot = self.controller.snapshot()
        # The live branch can contain hundreds of PredictionGroups.  Sending
        # all of them to the browser four times per second blocks its main
        # thread and, in turn, corrupts heartbeat RTT and video smoothness.
        # Detailed groups remain in traces/controller state for diagnostics.
        controller = asdict(replace(controller_snapshot, predictions=()))
        controller.pop("predictions")
        controller["prediction_groups"] = len(controller_snapshot.predictions)
        return {
            "session": self.id,
            "mode": self.mode,
            "stopped": self.stopped,
            "proposal_frames": self.proposal_frames,
            "serving_policy": self.serving_policy,
            "sampling_requires_refill": False,
            "timeline_extension": {
                "enabled": self.auto_extend_timeline,
                "mode": self.extension_mode,
                "condition_ready_to_first_commit_seconds":
                    self.condition_estimate.lead_seconds,
                "continuation_to_first_commit_seconds":
                    self.continuation_estimate.lead_seconds,
                "ema": {
                    "mode": "disabled" if (
                        self.condition_estimate.alpha == 0
                        and self.continuation_estimate.alpha == 0
                    ) else "separate",
                    "alpha": self.condition_estimate.alpha,
                    "margin_seconds": self.condition_estimate.margin_seconds,
                    "condition_value_seconds": self.condition_estimate.value,
                    "condition_observations": self.condition_estimate.count,
                    "continuation_value_seconds": self.continuation_estimate.value,
                    "continuation_observations": self.continuation_estimate.count,
                },
                "delivery_credit_frames": self.delivery_ahead_frames,
                "rewrite_horizon_fallback": self.rewrite_horizon_fallback,
                "rewrite_readiness_window": self.rewrite_readiness_window,
                "rewrite_guard_frames": self.rewrite_guard_frames,
                "playhead_clock": {
                    "mode": self.playhead_clock_mode,
                    "feedback_timeout_seconds":
                        self.playhead_feedback_timeout_seconds,
                    "acknowledged_frame": self.played,
                    "scheduling_frame": self._scheduling_playhead(snapshot_now),
                    "feedback_age_seconds": max(
                        0.0, snapshot_now - self.playhead_feedback_at
                    ),
                },
            },
            "error": self.error,
            "played_frame": self.played,
            "committed_frame": self.committed,
            "horizon_frame": self.horizon,
            "buffer_s": (self.horizon - self.played) / FPS,
            "budget_s": self.budget,
            "latest_prompt_version": self.version,
            "transport_rtt_s": self.transport_rtt_s,
            "generated_prompt_version": self.applied_version,
            "prompts": self.prompts,
            "inflight": self.busy,
            "reserved_rewrite_boundary": self.rewrite_boundary,
            "projected_rewrite_target_frame": self.rewrite_target_frame,
            "client_online": snapshot_now - self.last_seen < 10,
            "proposal_metrics": list(self.proposal_metrics.values()),
            "controller": controller,
        }
