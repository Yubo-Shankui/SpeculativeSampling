"""Shared retained-KV web-session facade for SpeculativeSampling model plugins."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import time
import uuid

from speculative_sampling import (
    SpeculativeSamplingController,
    ContinuationKind,
    ControllerConfig,
    PredictionGroup,
)

from .media_types import LiveUnit
from .estimator import ProductionEstimate

FPS = 24


class NativeSafePointAdapter:
    """Select the exact SpeculativeSampling safe point accepted by the generator.

    K controls proposal length, not branch alignment. The LTX generator owns
    incremental clean-commit and sink/recent-KV retention; this adapter only
    chooses the logical boundary.
    """

    continuation_kind = ContinuationKind.RETAINED_KV

    def select_retain_until(self, requested, candidates, horizon):
        return next((value for value in candidates if requested <= value <= horizon), None)


class NativeChunkBoundaryAdapter:
    """Base serving: a generated native chunk is indivisible for replacement."""

    continuation_kind = ContinuationKind.RETAINED_KV

    def __init__(self, chunk_size):
        self.chunk_size = int(chunk_size)

    def select_retain_until(self, requested, candidates, horizon):
        return next(
            (
                value for value in candidates
                if requested <= value <= horizon and value % self.chunk_size == 0
            ),
            None,
        )


class RetainedKVLiveSession:
    """Translate browser delivery to model-owned retained-KV boundaries.

    Native generation is always one complete K-latent joint proposal.  This
    class does not own or edit model KV; its tasks carry ``retain_until`` to
    the persistent vForge generator session.
    """

    backend_delivery_pacing = True

    chunk_latents = None
    capabilities = None
    mapper_factory = None
    split_boundary_factory = None
    split_services = False
    # None preserves the historical split-services contract; model subclasses
    # may require encoded handles without also using split media services.
    condition_handles_required = None
    # The reference-once first logical group is 9 RGB frames. Transport must
    # accept a complete group; its packet credit is independent of model K.
    delivery_packet_frames = 9
    # Set only from a validated serving config by the launcher. None means the
    # explicit development fallback, not an allegedly profiled deployment.
    serving_policy_config = None
    live_policy = None

    def __init__(
        self, prompt, reference, *, seed=42, mode="latent", initial_budget=3.5,
        delivery_ahead_frames=None, proposal_frames=363, serving_policy="co", emit=None,
        chunk_latents=None, condition_ready_to_first_commit_seconds=None,
        continuation_to_first_commit_seconds=None,
    ):
        if mode not in ("latent", "image", "relay"):
            raise ValueError("mode must be latent, image or relay")
        if serving_policy not in ("base", "co"):
            raise ValueError("serving_policy must be base or co")
        chunk_latents = int(self.chunk_latents if chunk_latents is None else chunk_latents)
        if chunk_latents < 1:
            raise ValueError("chunk_latents must be positive")
        self.id = uuid.uuid4().hex
        if self.mapper_factory is None or self.live_policy is None:
            raise ValueError("retained-KV plugin must declare mapper and live policy")
        self.mapper = self.mapper_factory()
        if self.split_services:
            self.rgb_boundary = self.mapper.rgb_boundary
            self.latent_boundary = self.mapper.latent_boundary
        self.reference = reference
        self.seed = int(seed)
        self.mode = mode
        self.serving_policy = serving_policy
        self.condition_handles_required = (
            self.split_services if self.condition_handles_required is None
            else bool(self.condition_handles_required)
        )
        self.prompt = self._prompt(prompt)
        self.version = 1
        self.prompts = {1: self.prompt}
        # One-shot UI actions may nominate the condition that takes over only
        # after the action itself has been dispatched.
        self.followup_prompts = {}
        now = time.monotonic()
        self.prompt_received_at = {1: now}
        self.condition_ready_at = ({1: now} if not self.condition_handles_required else {})
        condition_initial = float(
            self.live_policy.condition_ready_to_first_commit_seconds
            if condition_ready_to_first_commit_seconds is None
            else condition_ready_to_first_commit_seconds
        )
        continuation_initial = float(
            self.live_policy.continuation_to_first_commit_seconds
            if continuation_to_first_commit_seconds is None
            else continuation_to_first_commit_seconds
        )
        if condition_initial <= 0 or continuation_initial <= 0:
            raise ValueError("first-commit timing estimates must be positive")
        if not math.isclose(condition_initial, continuation_initial):
            raise ValueError(
                "retained-KV condition and continuation latencies must share one estimate"
            )
        compute_lead = max(1, self.mapper.distance_for_seconds(condition_initial))
        refill_lead = max(1, self.mapper.distance_for_seconds(continuation_initial))
        self.controller = SpeculativeSamplingController(
            self.prompt,
            ControllerConfig(
                chunk_size=chunk_latents,
                delivery_size=1,
                compute_lead=compute_lead,
                refill_lead=refill_lead,
                seed=self.seed,
            ),
            boundary_adapter=(
                NativeSafePointAdapter()
                if serving_policy == "co"
                else NativeChunkBoundaryAdapter(chunk_latents)
            ),
        )
        if self.condition_handles_required:
            # Initial text also has to be encoded before acquiring the DiT slot.
            self.controller.condition_ready.clear()
        self.proposal_frames = self.rgb_boundary(chunk_latents)
        self.delivery_ahead_frames = int(
            self.live_policy.delivery_credit_frames
            if delivery_ahead_frames is None else delivery_ahead_frames
        )
        if self.delivery_ahead_frames != 0:
            raise ValueError("backend-paced SpeculativeSampling requires delivery credit zero")
        self.initial_budget = float(initial_budget)
        margin = self.live_policy.ema_margin_seconds
        alpha = self.live_policy.ema_alpha
        self.production_estimate = ProductionEstimate(condition_initial, margin, alpha)
        self.condition_estimate = self.production_estimate
        self.continuation_estimate = self.production_estimate
        self.readiness_curve = ()
        self.decode_window_seconds = 1.0  # uncalibrated bootstrap fallback
        if self.serving_policy_config is not None:
            policy = self.serving_policy_config
            estimator = policy["estimator"]
            calibrated = ProductionEstimate(
                estimator["initial_dit_production_seconds"],
                estimator["calibrated_margin_seconds"], estimator["alpha"])
            self.condition_estimate = calibrated
            self.continuation_estimate = calibrated
            self.production_estimate = calibrated
            self.readiness_curve = tuple(estimator["readiness_curve"])
            if len(self.readiness_curve) != chunk_latents:
                raise ValueError("readiness curve must cover each new delivery group")
            if chunk_latents > self.mapper.group_latents:
                # Last RGB group's PCM can depend on the next proposal.
                # Its readiness includes refill scheduling, not VAE window
                # service time; do not turn that wait into a decode-time floor.
                self.decode_window_seconds = max((
                    self.readiness_curve[i] - self.readiness_curve[i-self.mapper.group_latents]
                    for i in range(self.mapper.group_latents, chunk_latents - 1)), default=0.)
        if self.split_services:
            if self.split_boundary_factory is None:
                raise ValueError("split retained-KV plugin requires a boundary factory")
            self.controller.boundary_adapter = self.split_boundary_factory(self)
        self.durations = []
        self.transport_rtt_s = 0.0
        self.units = []
        self.busy = None
        # Freeze the projected cut when encoded conditioning becomes ready.
        # Dispatch may be delayed by an old refill, but that delay must not
        # move the prompt's safe point farther into the future.
        self.rewrite_target = None
        self.rewrite_boundary = None
        self.rewrite_frontier_seconds = None
        self.rewrite_dispatch_delay_seconds = 0.0
        self.proposal_metrics = {}
        self.counter = 0
        self.played = 0
        self.committed = 0
        # WebSocket delivery uses a backend-owned token bucket.  Browser
        # heartbeat remains telemetry only: it must neither stall delivery nor
        # grant permission to irrevocably commit a large future buffer.
        self._delivery_pace_credit = None
        self._delivery_pace_last = None
        self._delivery_pace_capacity = None
        self.stopped = False
        self.error = None
        self.recovering_workers = {}
        self.last_seen = now
        self.emit = emit or (lambda _event: None)
        if self.split_services:
            self.controller.event_history_limit = 256
            self.controller.event_sink = lambda record: self.event(
                "controller_event", controller_event=record)
        self.event("session_start", mode=mode, prompt=self.prompt, seed=self.seed)

    @staticmethod
    def _prompt(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 5000:
            raise ValueError("prompt must contain 1–5000 characters")
        return value.strip()

    def event(self, event_name, **fields):
        self.emit({
            "event": event_name, "session": self.id, "wall_time": time.time(),
            "monotonic": time.monotonic(), "played_frame": self.played,
            "committed_frame": self.committed, **fields,
        })

    @property
    def horizon(self):
        return self.rgb_boundary(self.controller.horizon)

    @property
    def applied_version(self):
        return self.controller.applied_condition_version

    @property
    def budget(self):
        return self.condition_budget

    @property
    def condition_budget(self):
        # Rewrite safety ends at first media ready on the coordinator. Browser
        # RTT is downstream of SpeculativeSampling and must not move the branch point.
        return self.production_estimate.lead_seconds

    @property
    def continuation_budget(self):
        # Refill protects against underrun and may remain conservative, but it
        # learns from the same first-media-ready distribution as rewrites.
        return self.production_estimate.lead_seconds

    def _reserve_rewrite_boundary(self):
        if self.serving_policy != "co" or self.applied_version <= 0:
            self.rewrite_target = None
            self.rewrite_boundary = None
            self.rewrite_frontier_seconds = None
            self.rewrite_dispatch_delay_seconds = 0.0
            return
        frontier = self.controller.committed_until
        protected = max(
            frontier, self.controller.locked_until,
            self.controller.decoder_protected_until,
        )
        self.rewrite_frontier_seconds = self.mapper.seconds(frontier)
        self.rewrite_dispatch_delay_seconds = self._inflight_dispatch_delay()
        effective_lead = (
            self.production_estimate.lead_seconds
            + self.rewrite_dispatch_delay_seconds
        )
        # Match H3: the shared production clock chooses q exactly once. LTX
        # positions are already individual latent boundaries, so the adapter
        # may validate/alignment-map that target but may not search farther
        # based on decoder or media readiness.
        lead = max(1, self.mapper.distance_for_seconds(effective_lead))
        target = max(frontier + lead, protected)
        boundary = self.controller.boundary_adapter.select_retain_until(
            target, self.controller.valid_boundaries, self.controller.horizon)
        self.rewrite_target = target
        self.rewrite_boundary = target if boundary is None else boundary
        self.event(
            "rewrite_reserved",
            prompt_version=self.controller.pending_condition_version,
            target=self.rewrite_target,
            boundary_frame=self.rgb_boundary(self.rewrite_boundary),
            dispatch_delay_s=self.rewrite_dispatch_delay_seconds,
        )

    def _inflight_dispatch_delay(self):
        """Estimate only time for the current native call to release its slot."""
        if self.busy is None:
            return 0.0
        elapsed = max(0.0, time.monotonic() - self.busy["started_monotonic"])
        completed = [
            metric for metric in self.proposal_metrics.values()
            if metric.get("finished_successfully") and metric.get("total_s") is not None
            and metric.get("action") in ("update", "refill", "continuation")
        ][-8:]
        if completed:
            slot_seconds = max(float(metric["total_s"]) for metric in completed)
        else:
            # First media is handed off before the sole clean-prefill returns.
            # Until one steady proposal has completed, reserve a small tail for
            # that slot-release work rather than learning from cold bootstrap.
            slot_seconds = self.production_estimate.lead_seconds + 0.3
        return max(0.0, slot_seconds - elapsed)

    def mark_condition_ready(self, version, handle=None, *, condition_encode_seconds=None):
        """Install an encoded condition and freeze its SpeculativeSampling cut now."""
        self.controller.mark_condition_ready(version, handle)
        self.condition_ready_at[version] = time.monotonic()
        if version == self.controller.pending_condition_version:
            self._reserve_rewrite_boundary()
        fields = {
            "prompt_version": version,
            "target": self.rewrite_target,
            "reserved_boundary": self.rewrite_boundary,
        }
        if condition_encode_seconds is not None:
            fields["condition_encode_seconds"] = float(condition_encode_seconds)
        self.event("condition_ready", **fields)

    def update_prompt(self, value):
        value = self._prompt(value)
        if self.stopped:
            raise ValueError("session is stopped")
        self.last_seen = time.monotonic()
        # The controller stores an opaque handle after encoding; compare UI
        # input here rather than comparing a string with an encoded handle.
        if value == self.prompt:
            return self.version
        version = self.controller.update_condition(
            value, ready=not self.condition_handles_required)
        if version != self.version:
            self.version = version
            self.prompt = value
            self.prompts[version] = value
            self.prompt_received_at[version] = time.monotonic()
            if not self.condition_handles_required:
                self.condition_ready_at[version] = self.prompt_received_at[version]
                self._reserve_rewrite_boundary()
                self.event(
                    "condition_ready",
                    prompt_version=version,
                    target=self.rewrite_target,
                    reserved_boundary=self.rewrite_boundary,
                )
            self.event("prompt_received", prompt_version=version, prompt=value)
        return version

    def set_followup_prompt(self, version, value):
        if self.stopped:
            raise ValueError("session is stopped")
        if version not in self.prompts:
            raise ValueError("prompt version does not exist")
        self.followup_prompts = {int(version): self._prompt(value)}

    def heartbeat(self, played):
        if not 0 <= played <= self.committed:
            raise ValueError("playhead is outside delivered content")
        self.played = max(self.played, int(played))
        self.controller.advance_playback(self.latent_boundary(self.played))
        if self.split_services:
            # Preserve one native decode window for transport retry. Worker
            # snapshots have their own release protocol and are not touched.
            until = max(self.controller.archived_until,
                        self.controller.played_until - self.mapper.group_latents)
            self.controller.compact_played(until)
            self.units = [group.payload for group in self.controller.predictions]
        self.last_seen = time.monotonic()

    def plan(self, now=None):
        now = time.monotonic() if now is None else now
        if (self.busy or self.stopped or self.error or self.recovering_workers
                or now - self.last_seen > 10):
            return None
        changed = self.controller.pending_condition_version is not None
        interactive_update = changed and self.applied_version > 0
        condition_lead = max(1, self.mapper.distance_for_seconds(self.condition_budget))
        if interactive_update and self.serving_policy == "co":
            pending = self.controller.pending_condition_version
            if (pending in self.controller.condition_ready
                    and (self.rewrite_target is None
                         or self.rewrite_target < self.controller.committed_until)):
                # Compatibility for direct controller users; the production
                # ConditionPump reserves this at condition-ready time.
                self._reserve_rewrite_boundary()
            if self.rewrite_target is not None:
                condition_lead = max(
                    0, self.rewrite_target - self.controller.committed_until
                )
        continuation_lead = max(
            1, self.mapper.distance_for_seconds(self.continuation_budget)
        )
        self.controller.config = replace(
            self.controller.config, compute_lead=condition_lead,
            # Refill uses the measured first-media lead directly. The native
            # A/V tail is validated at the boundary; reserving another latent
            # here duplicated the ceil-to-latent allowance and started LTX
            # continuation about one frame group too early.
            refill_lead=continuation_lead,
        )
        if self.split_services:
            adapter = self.controller.boundary_adapter
            # Deadline planning starts at the backend-owned irrevocable
            # frontier. Browser playback remains telemetry/compaction only.
            adapter.commit_frontier_seconds = (
                self.rewrite_frontier_seconds
                if interactive_update and self.rewrite_frontier_seconds is not None
                else self.mapper.seconds(self.controller.committed_until)
            )
            command = self.controller.dispatch(
                deadline_plan=adapter.deadlines,
                fixed_update_boundary=(
                    self.rewrite_boundary
                    if interactive_update and self.serving_policy == "co"
                    else None
                ),
            )
        else:
            command = self.controller.dispatch()
        if command is None:
            return None
        if interactive_update and self.serving_policy == "co":
            # Native cache/decoder constraints may move the exact branch
            # forward from the time projection. Once dispatch is atomic this
            # becomes the only old-branch delivery fence.
            self.rewrite_boundary = command.retain_until
        source_frames = (
            self.rgb_boundary(command.retain_until + command.chunk_size)
            - self.rgb_boundary(command.retain_until)
        )
        task = {
            "session": self.id, "proposal": command.job_id, "action": command.reason.value,
            "boundary": self.rgb_boundary(command.retain_until),
            "retain_until": command.retain_until,
            "chunk_latents": command.chunk_size,
            "prompt": self.prompts[command.condition_version],
            "prompt_version": command.condition_version,
            "seed": command.seed, "reference": self.reference,
            "trim_frames": 0, "proposal_frames": source_frames,
            "serving_policy": self.serving_policy,
            "budget_s": (
                self.continuation_budget
                if command.reason.value in ("refill", "continuation")
                else self.condition_budget
            ),
            "transport_rtt_s": self.transport_rtt_s, "started_monotonic": now,
        }
        if self.condition_handles_required:
            task["condition_handle"] = command.condition
        self.busy = task
        self.proposal_metrics[command.job_id] = {
            "proposal": command.job_id, "prompt_version": command.condition_version,
            "action": command.reason.value, "started_monotonic": now,
            "queue_s": max(0.0, now - self.condition_ready_at.get(
                command.condition_version,
                self.prompt_received_at[command.condition_version],
            )),
            "first_ready_s": None, "first_commit_s": None, "total_s": None, "stages": {},
            "finished_successfully": False, "estimate_observed": False,
        }
        self.event("production_start", **{key: value for key, value in task.items() if key != "session"})
        followup = self.followup_prompts.pop(task["prompt_version"], None)
        if followup is not None and self.version == task["prompt_version"]:
            followup_version = self.update_prompt(followup)
            self.event(
                "one_shot_followup_queued",
                source_prompt_version=task["prompt_version"],
                prompt_version=followup_version,
            )
        return dict(task)

    def publish(self, task, available_source):
        if self.split_services:
            return self._publish_split(task, available_source)
        if self.stopped or self.busy != task:
            return False
        if available_source < task["proposal_frames"]:
            return True
        if self.controller.horizon > task["retain_until"]:
            return True
        source = 0
        groups = []
        live = []
        for offset in range(task["chunk_latents"]):
            logical = task["retain_until"] + offset
            start = self.rgb_boundary(logical)
            end = self.rgb_boundary(logical + 1)
            span = end - start
            unit = LiveUnit(
                start, end, task["proposal"], offset,
                source, source + span, task["prompt_version"],
            )
            live.append(unit)
            groups.append(PredictionGroup(
                logical, logical + 1, task["proposal"], task["prompt_version"], payload=unit,
            ))
            source += span
        if source != task["proposal_frames"]:
            raise RuntimeError(f"LTX source geometry mismatch: {source} != {task['proposal_frames']}")
        self.controller.publish(task["proposal"], groups)
        self.units = [group.payload for group in self.controller.predictions]
        metric = self.proposal_metrics[task["proposal"]]
        metric["first_ready_s"] = time.monotonic() - task["started_monotonic"]
        self.event("production_chunk", proposal=task["proposal"], boundary_frame=task["boundary"],
                   source_end=source, horizon_frame=self.horizon,
                   elapsed_s=time.monotonic() - task["started_monotonic"])
        return True

    def latents_ready(self, task):
        if self.stopped:
            return False
        mapping = self.mapper.mapping(task["retain_until"], task["chunk_latents"])
        installed = self.controller.latents_ready(
            task["proposal"], mapping, native_count=task["chunk_latents"])
        if installed:
            metric = self.proposal_metrics.get(task["proposal"])
            if metric is not None and metric.get("latent_ready_s") is None:
                metric["latent_ready_s"] = time.monotonic() - task["started_monotonic"]
            self.event("latents_ready", proposal=task["proposal"],
                       generated_horizon=self.controller.horizon)
        return installed

    def _publish_split(self, task, available_source, *, source_start=0, dependency_end=None,
                       kind="media", source_dependencies=()):
        if self.stopped:
            return False
        proposal = self.controller.proposals.get(task["proposal"])
        if proposal is None or not proposal.generated:
            return False
        groups = []
        for item in proposal.mapping:
            group_source_start = self.rgb_boundary(item.start) - task["boundary"]
            source_end = self.rgb_boundary(item.end) - task["boundary"]
            if source_end > available_source:
                break
            if group_source_start < source_start:
                continue
            if item.end > proposal.valid_end:
                continue
            unit = LiveUnit(self.rgb_boundary(item.start), self.rgb_boundary(item.end),
                            task["proposal"], item.start-task["retain_until"],
                            group_source_start, source_end, task["prompt_version"])
            groups.append(PredictionGroup(
                item.start, item.end, task["proposal"], task["prompt_version"],
                payload=unit, dependency_end=(
                    item.dependency_end if dependency_end is None else dependency_end),
                source_dependencies=source_dependencies))
        result = self.controller.media_ready(task["proposal"], groups, kind=kind)
        self.units = [group.payload for group in self.controller.predictions]
        metric = self.proposal_metrics.get(task["proposal"])
        if (result and metric and metric["first_ready_s"] is None
                and any(g.job_id == task["proposal"] for g in self.controller.predictions)):
            metric["first_ready_s"] = time.monotonic() - task["started_monotonic"]
            self._observe_first_media_ready(metric)
        return result

    def _observe_first_media_ready(self, metric):
        """Train the shared EMA once media and a successful finish both exist."""
        production = metric.get("first_ready_s")
        if (metric.get("estimate_observed") or not metric.get("finished_successfully")
                or production is None
                or metric.get("action") not in ("update", "refill", "continuation")):
            return False
        self.production_estimate.observe(production)
        metric["production_observation_s"] = production
        metric["estimate_observed"] = True
        return True

    def publish_media_window(self, task, source_start, source_end, dependency_end, *,
                             kind="media", source_dependencies=()):
        """Install a specific decoded window with its actual dependencies."""
        return self._publish_split(task, source_end, source_start=source_start,
                                   dependency_end=dependency_end, kind=kind,
                                   source_dependencies=source_dependencies)

    def complete(self, task, elapsed):
        if self.busy != task:
            return False
        if self.stopped:
            # Stop cancels delivery, not the in-flight native call. Its normal
            # return is the acknowledgement that remote state may be released.
            self.controller.job_failed(task["proposal"], "session stopped")
            self.worker_terminated(task["proposal"])
            self.busy = None
            self.event("stopped_generation_finished", proposal=task["proposal"])
            return False
        if self.error:
            # The rank group returned: only now is it safe to acknowledge that
            # the failed job no longer owns the generator. Do not train EMA
            # from work whose media/condition worker failed concurrently.
            self.worker_terminated(task["proposal"])
            return False
        if not self.split_services:
            self.publish(task, task["proposal_frames"])
        self.controller.finish(task["proposal"])
        metric = self.proposal_metrics[task["proposal"]]
        # The safe cut predicts first committable media, not latent handoff or
        # slot release after speculative clean-prefill. A proposal with no
        # media observation must not train the scheduling estimator.
        production = metric.get("first_ready_s")
        metric["finished_successfully"] = True
        self._observe_first_media_ready(metric)
        self.durations.append(float(elapsed))
        metric.setdefault("production_observation_s", production)
        metric["total_s"] = time.monotonic() - task["started_monotonic"]
        self.busy = None
        if self.applied_version == self.version:
            self.rewrite_target = None
            self.rewrite_boundary = None
            self.rewrite_frontier_seconds = None
            self.rewrite_dispatch_delay_seconds = 0.0
        self.event("production_ready", proposal=task["proposal"], boundary_frame=task["boundary"],
                   prompt_version=task["prompt_version"], elapsed_s=elapsed,
                   horizon_frame=self.horizon)
        return True

    def record_metrics(self, task, stages, engine_elapsed=None, mock=False):
        metric = self.proposal_metrics.get(task["proposal"])
        if metric is not None:
            metric.update(stages=stages, engine_elapsed_s=engine_elapsed, mock=mock)
            self.event("production_metrics", **metric)

    def record_decode_window(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid decode duration")
        # First decode is a separately recorded cold bucket. Do not let a
        # single optimistic steady window erase observed slow service.
        if not hasattr(self, "_decode_observations"):
            self._decode_observations = []
            self.event("decode_cold_observation", elapsed_s=float(seconds))
            return
        self._decode_observations.append(float(seconds))
        self._decode_observations = self._decode_observations[-32:]
        profile_floor = 0.
        if len(self.readiness_curve) > self.mapper.group_latents:
            profile_floor = max((
                self.readiness_curve[i] - self.readiness_curve[i-self.mapper.group_latents]
                for i in range(self.mapper.group_latents, len(self.readiness_curve) - 1)), default=0.)
        self.decode_window_seconds = max(profile_floor, max(self._decode_observations))

    def deliver(self, cursor, max_frames=12, *, paced=False):
        if self.stopped or self.error:
            return []
        if cursor < 0 or cursor > self.committed:
            raise ValueError("cursor must resume at or before the delivered frontier")
        if max_frames < 1:
            raise ValueError("packet frame budget must be positive")
        if cursor < self.committed:
            previous = next((g.payload for g in self.controller.predictions
                             if g.payload.start <= cursor < g.payload.end
                             and g.payload.end <= self.committed), None)
            if previous is None:
                raise ValueError("resume cursor is not retained committed media")
            end = min(previous.end, cursor + max_frames)
            offset = cursor - previous.start
            return [replace(
                previous, start=cursor, end=end,
                source_start=previous.source_start + offset,
                source_end=previous.source_start + offset + end - cursor,
            )]
        next_group = next(
            (group for group in self.controller.predictions
             if group.start == self.controller.committed_until),
            None,
        )
        if next_group is None:
            return []
        if (self.serving_policy == "co" and self.rewrite_boundary is not None
                and next_group.end > self.rewrite_boundary
                and next_group.condition_version != self.version):
            # Old future is replaceable. Stop committing it at the cut frozen
            # when conditioning became ready; browser feedback never controls
            # this backend-owned fence.
            return []
        if paced:
            now = time.monotonic()
            capacity = float(max_frames + self.delivery_ahead_frames)
            if self._delivery_pace_credit is None:
                self._delivery_pace_credit = capacity
                self._delivery_pace_last = now
                self._delivery_pace_capacity = capacity
            elif self._delivery_pace_capacity != capacity:
                raise RuntimeError("delivery pacing packet size changed during a session")
            else:
                elapsed = max(0.0, now - self._delivery_pace_last)
                self._delivery_pace_credit = min(
                    capacity, self._delivery_pace_credit + elapsed * FPS)
                self._delivery_pace_last = now
            span = next_group.payload.end - next_group.payload.start
            if span > self._delivery_pace_credit + 1e-9:
                return []
        ready = self.controller.deliver()
        units = [group.payload for group in ready]
        if units:
            if paced:
                self._delivery_pace_credit = max(
                    0.0, self._delivery_pace_credit
                    - sum(unit.end - unit.start for unit in units)
                )
            self.committed = units[-1].end
            metric = self.proposal_metrics.get(units[0].proposal)
            if metric and metric["first_commit_s"] is None:
                metric["first_commit_s"] = time.monotonic() - metric["started_monotonic"]
            self.event("commit", units=[asdict(unit) for unit in units])
            unit = units[0]
            end = min(unit.end, cursor + max_frames)
            return [replace(
                unit, end=end,
                source_end=unit.source_start + end - cursor,
            )]
        return []

    def stop(self):
        self.stopped = True
        self.event("session_stop")

    def fail(self, message):
        self.error = str(message)
        active = self.controller.active_job
        if active is not None:
            self.controller.job_failed(active.job_id, self.error)
            # Keep in-flight identity until the rank-group completion or
            # supervisor confirms termination. Error is not a slot release.
        else:
            self.busy = None
        self.event("error", message=self.error)

    def begin_recovery(self, worker, message):
        if self.stopped or self.error:
            return False
        self.recovering_workers[str(worker)] = str(message)
        self.event("worker_recovery_started", worker=str(worker), message=str(message))
        return True

    def finish_recovery(self, worker, report):
        worker = str(worker)
        if worker not in self.recovering_workers or self.stopped or self.error:
            return False
        del self.recovering_workers[worker]
        self.event("worker_recovery_finished", worker=worker, report=report)
        return True

    def worker_terminated(self, job_id):
        acknowledged = self.controller.worker_terminated(job_id)
        if acknowledged and self.busy and self.busy["proposal"] == job_id:
            self.busy = None
        return acknowledged

    def snapshot(self):
        return {
            "session": self.id, "mode": self.mode, "stopped": self.stopped,
            "proposal_frames": self.proposal_frames, "serving_policy": self.serving_policy,
            "error": self.error, "played_frame": self.played,
            "committed_frame": self.committed, "horizon_frame": self.horizon,
            "buffer_s": (self.horizon - self.committed) / FPS,
            "budget_s": self.condition_budget,
            "scheduling_estimates": {
                "condition_seconds": self.condition_estimate.value,
                "continuation_seconds": self.continuation_estimate.value,
                "margin_seconds": self.live_policy.ema_margin_seconds,
                "mode": "shared_first_media_ready",
            },
            "latest_prompt_version": self.version, "transport_rtt_s": self.transport_rtt_s,
            "generated_prompt_version": self.applied_version, "prompts": self.prompts,
            "inflight": self.busy, "client_online": time.monotonic() - self.last_seen < 10,
            "proposal_metrics": list(self.proposal_metrics.values()),
            "generator_kv_latent_frontier": self.controller.locked_until,
            "controller_state": self.controller.state,
            "generated_horizon": self.controller.horizon,
            "media_ready_horizon": self.controller.snapshot().media_ready,
            "video_ready_horizon": self.controller.snapshot().video_ready,
            "audio_ready_horizon": self.controller.snapshot().audio_ready,
            "decoder_protected_until": self.controller.decoder_protected_until,
            "reserved_rewrite_target": self.rewrite_target,
            "reserved_rewrite_boundary": self.rewrite_boundary,
            "reserved_rewrite_frontier_seconds": self.rewrite_frontier_seconds,
            "reserved_rewrite_dispatch_delay_seconds": self.rewrite_dispatch_delay_seconds,
            "profile_calibrated": self.serving_policy_config is not None,
            "recovering_workers": dict(self.recovering_workers),
        }
