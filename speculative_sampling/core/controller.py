"""Pure, coordinator-serialized SpeculativeSampling state machine.

No tensor, FPS, network or model imports. Native generation and media completion
are separate: a completed proposal remains addressable until its ranges expire.
Legacy ``publish`` is a monolithic adapter bridge, not the split runtime API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

from .protocol import (
    BoundaryReadiness, ExactBoundaryAdapter, GenerationCommand, GenerationReason,
    GeneratorBoundaryAdapter, OutputGroup, PredictionGroup,
)


@dataclass(frozen=True)
class ControllerConfig:
    chunk_size: int
    delivery_size: int = 1
    compute_lead: int = 1
    refill_lead: int = 1
    seed: int = 42

    def __post_init__(self):
        if self.chunk_size < 1 or self.delivery_size < 1:
            raise ValueError("chunk_size and delivery_size must be positive")
        if min(self.compute_lead, self.refill_lead) < 0:
            raise ValueError("lead distances must be non-negative")


@dataclass
class Proposal:
    command: GenerationCommand
    valid_end: int | None = None
    generated: bool = False
    finished: bool = False
    failed: bool = False
    mapping: tuple[OutputGroup, ...] = ()
    media: dict[int, dict[str, PredictionGroup]] = field(default_factory=dict)
    stable_state: Any = None


@dataclass(frozen=True)
class ControllerSnapshot:
    state: str
    played_until: int
    committed_until: int
    locked_until: int
    horizon: int
    latest_condition_version: int
    applied_condition_version: int
    pending_condition_version: int | None
    active_job: GenerationCommand | None
    predictions: tuple[PredictionGroup, ...]
    discarded_groups: int
    generated_horizon: int = 0
    video_ready: int = 0
    audio_ready: int = 0
    media_ready: int = 0
    decoder_protected_until: int = 0
    recovery: bool = False


class SpeculativeSamplingController:
    def __init__(self, initial_condition: Any, config: ControllerConfig, *,
                 boundary_adapter: GeneratorBoundaryAdapter | None = None,
                 event_sink: Callable[[dict], None] | None = None,
                 event_history_limit: int | None = None):
        if event_history_limit is not None and (
            type(event_history_limit) is not int or event_history_limit < 0
        ):
            raise ValueError("event history limit must be nonnegative")
        self.event_sink = event_sink
        self.event_history_limit = event_history_limit
        self.config = config
        self.boundary_adapter = boundary_adapter or ExactBoundaryAdapter()
        self.conditions = {1: self._condition(initial_condition)}
        self.condition_ready = {1}
        self.latest_condition_version = 1
        self.applied_condition_version = 0
        self.pending_condition_version: int | None = 1
        self.predictions: list[PredictionGroup] = []
        self.proposals: dict[int, Proposal] = {}
        self.played_until = self.committed_until = self.locked_until = 0
        self.decoder_protected_until = 0
        self.active_job: GenerationCommand | None = None
        self.next_job_id = 1
        self.discarded_groups = 0
        self.recovery = False
        self._bridge_progress: tuple[int, int] | None = None
        self.events: list[dict] = []
        self.archived_until = 0

    @staticmethod
    def _condition(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("condition must be a non-empty opaque handle")
        return value.strip() if isinstance(value, str) else value

    def _event(self, name, **fields):
        record = {"event": name, **fields}
        self.events.append(record)
        if self.event_history_limit is not None:
            del self.events[:max(0, len(self.events) - self.event_history_limit)]
        if self.event_sink is not None:
            self.event_sink(dict(record))

    @property
    def horizon(self):
        return self._frontier("generated")

    @property
    def state(self):
        if self.active_job is not None:
            return "Terminating" if self.proposals[self.active_job.job_id].failed else "Produce"
        return "Consume"

    def _valid_mappings(self):
        return sorted(
            ((group, proposal) for proposal in self.proposals.values()
             for group in proposal.mapping
             if proposal.valid_end is not None and self.archived_until < group.end <= proposal.valid_end),
            key=lambda item: item[0].start,
        )

    def _frontier(self, kind):
        # Only acknowledged playback can be compacted. It has already passed
        # every readiness/commit check, so it is an immutable covered prefix.
        position = self.archived_until
        for group, proposal in self._valid_mappings():
            if group.start != position:
                break
            media = proposal.media.get(group.start, {})
            if kind == "media":
                ready = "media" in media or (
                    "video" in media and (not group.requires_audio or "audio" in media)
                )
            elif kind == "audio":
                ready = not group.requires_audio or "audio" in media or "media" in media
            else:
                ready = kind == "generated" or kind in media or "media" in media
            if not ready:
                break
            position = group.end
        return position

    def _dependencies_valid(self, group):
        for dependency in group.source_dependencies:
            source = self.proposals.get(dependency.job_id)
            if (source is None or not source.generated or source.failed
                    or source.valid_end is None
                    or dependency.start < source.command.retain_until
                    or dependency.end > source.valid_end):
                return False
        return True

    @property
    def valid_boundaries(self):
        return (self.archived_until, *(group.end for group, _ in self._valid_mappings()))

    def update_condition(self, value, *, ready=True):
        condition = self._condition(value)
        if condition == self.conditions[self.latest_condition_version]:
            return self.latest_condition_version
        if self.pending_condition_version is not None:
            self._event("ConditionSuperseded", version=self.pending_condition_version)
        self.latest_condition_version += 1
        version = self.latest_condition_version
        self.conditions[version] = condition
        self.pending_condition_version = version
        self._bridge_progress = None
        if ready:
            self.condition_ready.add(version)
        self._event("ConditionAccepted", version=version)
        return version

    def mark_condition_ready(self, version, handle=None):
        if version not in self.conditions:
            raise ValueError("unknown condition version")
        if self.active_job is not None and self.active_job.condition_version == version:
            raise ValueError("cannot mutate the condition of an active job")
        if handle is not None:
            self.conditions[version] = self._condition(handle)
        self.condition_ready.add(version)

    def _feasible(self, boundary, deadlines):
        deadlines = self._deadlines(boundary, deadlines)
        check = getattr(self.boundary_adapter, "boundary_readiness", None)
        if check is not None:
            evidence = check(boundary)
            return isinstance(evidence, BoundaryReadiness) and evidence.meets(deadlines)
        # Compatibility adapters have no timing evidence. They cannot certify
        # a deadline plan; already-ready old prefix is their only Cover proof.
        return not deadlines and self._frontier("media") >= boundary

    @staticmethod
    def _deadlines(boundary, plan):
        """Resolve mapper-owned deadlines separately for each candidate.

        A fixed tuple remains a compatibility option. For mapped timelines the
        caller supplies a function: moving a branch changes all its deadlines.
        Values are durations on the coordinator clock, not remote timestamps.
        """
        import math
        values = tuple(plan(boundary) if callable(plan) else plan)
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in values):
            raise ValueError("deadline durations must be finite numbers")
        return values

    def _projected_update_frontier(self):
        return max(
            self.committed_until + self.config.compute_lead,
            self.locked_until,
            self.decoder_protected_until,
        )

    def _choose_update_boundary(self, deadlines, *, requested=None):
        # SpeculativeSampling projects the irreversible frontier forward from condition
        # readiness.  While the replacement takes ``compute_lead`` output
        # groups to produce its first committable media, the old branch keeps
        # committing at the fixed playback rate.  Branch at the first native
        # safe point at/after that projected frontier.
        requested = (
            self._projected_update_frontier() if requested is None else int(requested)
        )
        if requested < max(self.committed_until, self.locked_until,
                           self.decoder_protected_until):
            raise ValueError("requested update boundary precedes protected state")
        candidates = self.valid_boundaries
        while requested <= self.horizon:
            boundary = self.boundary_adapter.select_retain_until(
                requested, candidates, self.horizon)
            if boundary is None:
                return None
            if boundary < requested or boundary not in candidates or boundary > self.horizon:
                raise ValueError("adapter returned an invalid boundary")
            if self._feasible(boundary, deadlines):
                return boundary
            requested = boundary + 1
        return None

    def _recovery_needed(self):
        # Production protects the backend-owned irrevocable frontier. Browser
        # playback is telemetry and compaction evidence, not refill pacing.
        if self.horizon <= self.committed_until:
            return True
        # Some native mappings need the next proposal's audio to finish the
        # final video group. Exhaustion of playable media, not exhaustion of
        # already sampled latent indices, determines recovery in this case.
        check = getattr(self.boundary_adapter, "recovery_needed", None)
        return bool(check()) if check is not None else False

    def dispatch(self, *, deadline_plan: Iterable[float] | Callable[[int], Iterable[float]] = (),
                 fixed_update_boundary: int | None = None,
                 refill_condition: Any | None = None):
        if self.active_job is not None:
            return None
        deadlines = deadline_plan if callable(deadline_plan) else tuple(deadline_plan)
        if self.next_job_id == 1:
            reason, boundary = GenerationReason.BOOTSTRAP, 0
            version = self.pending_condition_version or self.latest_condition_version
        elif self.pending_condition_version is not None:
            version = self.pending_condition_version
            reason = GenerationReason.UPDATE
            fixed_update = fixed_update_boundary is not None
            if fixed_update:
                requested = int(fixed_update_boundary)
                protected = max(
                    self.committed_until, self.locked_until,
                    self.decoder_protected_until,
                )
                selected = self.boundary_adapter.select_retain_until(
                    requested, self.valid_boundaries, self.horizon)
                boundary = selected if selected == requested and requested >= protected else None
            else:
                boundary = self._choose_update_boundary(deadlines)
            if boundary is None:
                # Never disguise insufficient cover as a normal switch. Only
                # once exhausted use an explicit recovery at the existing tail.
                if not self._recovery_needed():
                    # A short native proposal (for example Wan K3) may not yet
                    # cover the projected update point. Keep extending the old,
                    # replaceable branch just in time; the pending condition is
                    # deliberately left pending until enough cover exists.
                    if self.horizon - self.committed_until > self.config.refill_lead:
                        return None
                    requested = (
                        int(fixed_update_boundary)
                        if fixed_update_boundary is not None
                        else self._projected_update_frontier()
                    )
                    gap = max(0, requested - self.horizon)
                    previous = self._bridge_progress
                    boundary = self.boundary_adapter.select_retain_until(
                        self.horizon, self.valid_boundaries, self.horizon)
                    if boundary is None:
                        return None
                    if (previous is not None and previous[0] == version
                            and gap >= previous[1]):
                        # A producer running at or below playback rate cannot
                        # chase a projected frontier forever. Preserve the
                        # current valid tail, switch the pending condition
                        # there, and make the resulting underrun explicit.
                        reason = GenerationReason.RECOVERY
                        self._event(
                            "BridgeStalled", version=version,
                            previous_gap=previous[1], gap=gap, boundary=boundary,
                        )
                    else:
                        self._bridge_progress = (version, gap)
                        version = self.applied_condition_version
                        reason = GenerationReason.BRIDGE
                        if not version:
                            return None
                else:
                    boundary, reason = self.horizon, GenerationReason.RECOVERY
        elif self.horizon - self.committed_until <= self.config.refill_lead or self._recovery_needed():
            version = self.applied_condition_version or self.latest_condition_version
            boundary = self.boundary_adapter.select_retain_until(
                self.horizon, self.valid_boundaries, self.horizon)
            reason = (GenerationReason.RECOVERY if self._recovery_needed()
                      else GenerationReason.REFILL)
            if boundary is None:
                return None
            if refill_condition is not None:
                condition = self._condition(refill_condition)
                if condition != self.conditions[version]:
                    self.latest_condition_version += 1
                    version = self.latest_condition_version
                    self.conditions[version] = condition
                    self.condition_ready.add(version)
                    self._event(
                        "RefillConditionAccepted", version=version,
                        previous_version=self.applied_condition_version,
                    )
        else:
            return None
        if version not in self.condition_ready:
            return None
        if reason not in (GenerationReason.BOOTSTRAP, GenerationReason.RECOVERY):
            if (reason == GenerationReason.UPDATE
                    and fixed_update_boundary is not None):
                # SpeculativeSampling already fixed q from the production clock. Model
                # evidence may reject an invalid history/decoder state, but it
                # must not search forward for a more convenient q. Missing A/V
                # cover is prepared after atomic suffix invalidation.
                check = getattr(self.boundary_adapter, "boundary_readiness", None)
                if check is not None:
                    evidence = check(boundary)
                    if not (isinstance(evidence, BoundaryReadiness)
                            and evidence.context_ready and evidence.decoder_ready):
                        return None
            elif not self._feasible(boundary, deadlines):
                return None
        if reason == GenerationReason.RECOVERY:
            # Recovery relaxes coverage deadlines, not native history validity.
            check = getattr(self.boundary_adapter, "boundary_readiness", None)
            if check is not None:
                evidence = check(boundary)
                if not evidence.context_ready or not evidence.decoder_ready:
                    return None
            if boundary < max(self.committed_until, self.locked_until,
                              self.decoder_protected_until):
                return None
            self.recovery = True
            self._event("Underrun", boundary=boundary)
        resolved_deadlines = self._deadlines(boundary, deadlines)
        command = GenerationCommand(
            self.next_job_id, reason, version, self.conditions[version], boundary,
            self.config.chunk_size,
            (self.config.seed + (self.next_job_id - 1) * 104729) % (2**63 - 1),
            resolved_deadlines)
        # All feasibility checks precede this atomic invalidation/dispatch.
        for proposal in self.proposals.values():
            if proposal.valid_end is not None and proposal.valid_end > boundary:
                start = max(boundary, proposal.command.retain_until)
                self.discarded_groups += proposal.valid_end - start
                self._event("ProposalInvalidated", job_id=proposal.command.job_id,
                            start=start, end=proposal.valid_end)
                proposal.valid_end = start
                # The replacement decoder may replay a shorter final window.
                # Old callbacks still carry their original dependency end and
                # are rejected; replay callbacks carry the newly valid source
                # range. Never relabel old pixels as independent of the suffix.
                proposal.mapping = tuple(
                    replace(group, dependency_end=min(group.dependency_end or group.end, start))
                    if group.end <= start else group
                    for group in proposal.mapping
                )
        self.predictions = [
            group for group in self.predictions
            if group.end <= boundary and (group.dependency_end or group.end) <= boundary
            and self._dependencies_valid(group)
        ]
        for proposal in self.proposals.values():
            for start, kinds in list(proposal.media.items()):
                proposal.media[start] = {
                    kind: group for kind, group in kinds.items()
                    if (group.dependency_end or group.end) <= boundary
                    and self._dependencies_valid(group)
                }
        self.next_job_id += 1
        self.active_job = command
        self.proposals[command.job_id] = Proposal(command)
        self.locked_until = boundary
        if self.pending_condition_version == version:
            self.pending_condition_version = None
            self._bridge_progress = None
        self._event("GenerationStarted", job_id=command.job_id, version=version)
        return command

    def latents_ready(self, job_id, mapping: Iterable[OutputGroup], *, native_count):
        """Publish only a complete jointly sampled native chunk, once."""
        proposal = self.proposals.get(job_id)
        if proposal is None or proposal.failed or proposal.finished:
            return False
        groups = tuple(mapping)
        if proposal.generated:
            return groups == proposal.mapping  # idempotent duplicate
        if native_count != proposal.command.chunk_size or not groups:
            raise ValueError("LatentsReady requires a complete native K and output mapping")
        expected = proposal.command.retain_until
        previous_time = None
        for group in groups:
            if group.start != expected or (previous_time is not None
                                          and group.start_seconds != previous_time):
                raise ValueError("mapping must be contiguous in logical and playback time")
            if group.end - group.start != self.config.delivery_size:
                raise ValueError("mapping must expose delivery-sized logical groups")
            expected, previous_time = group.end, group.end_seconds
        proposal.mapping, proposal.valid_end, proposal.generated = groups, expected, True
        self._event("LatentsReady", job_id=job_id, end=expected, native_count=native_count)
        return True

    def media_ready(self, job_id, groups: Iterable[PredictionGroup], *, kind="media"):
        if kind not in ("media", "video", "audio"):
            raise ValueError("unknown media kind")
        proposal = self.proposals.get(job_id)
        if proposal is None or proposal.failed or not proposal.generated:
            return False
        incoming = list(groups)
        mappings = {group.start: group for group in proposal.mapping}
        accepted = []
        seen = set()
        for group in incoming:
            if group.end <= self.archived_until:
                continue  # delayed duplicate of irrevocably played/archived output
            mapping = mappings.get(group.start)
            if group.job_id != job_id or group.condition_version != proposal.command.condition_version:
                raise ValueError("media callback provenance mismatch")
            if mapping is None or mapping.end != group.end:
                raise ValueError("media callback does not match output mapping")
            dependency = max(group.dependency_end or group.end,
                             mapping.dependency_end or mapping.end)
            if (group.end > proposal.valid_end or dependency > proposal.valid_end
                    or not self._dependencies_valid(group)):
                continue  # invalid suffix must not invalidate a valid late prefix
            if group.start in seen or kind in proposal.media.get(group.start, {}):
                continue  # first result wins; duplicates cannot overwrite payload
            seen.add(group.start)
            accepted.append((group, dependency))
        for group, dependency in accepted:
            from dataclasses import replace
            group = replace(group, dependency_end=dependency)
            media = proposal.media.setdefault(group.start, {})
            media[kind] = group
            mapping = mappings[group.start]
            if "media" in media or ("video" in media and
                                    (not mapping.requires_audio or "audio" in media)):
                result = media.get("media", media.get("video"))
                # An AV group protects every decoded source dependency, not
                # only the video's (audio/vocoder can require more history).
                result = replace(result, dependency_end=max(
                    entry.dependency_end or entry.end for entry in media.values()),
                    source_dependencies=tuple(dict.fromkeys(
                        dependency for entry in media.values()
                        for dependency in entry.source_dependencies)))
                if not any(old.start == group.start for old in self.predictions):
                    self.predictions.append(result)
                    self._event("MediaReady", job_id=job_id, start=result.start,
                                end=result.end, condition_version=result.condition_version)
        self.predictions.sort(key=lambda group: group.start)
        return bool(accepted) or not incoming

    def publish(self, job_id, groups: Iterable[PredictionGroup]):
        """Legacy monolithic bridge; partial callbacks do not release DiT."""
        proposal = self.proposals.get(job_id)
        if proposal is None or proposal.failed:
            return False
        incoming = tuple(groups)
        if proposal.generated:
            return self.media_ready(job_id, incoming)
        if self.active_job is None or self.active_job.job_id != job_id:
            return False
        expected = proposal.mapping[-1].end if proposal.mapping else proposal.command.retain_until
        mappings = []
        for group in incoming:
            if group.job_id != job_id or group.condition_version != proposal.command.condition_version:
                raise ValueError("prediction callback version mismatch")
            if group.start != expected or group.end - group.start != self.config.delivery_size:
                raise ValueError("prediction callbacks must extend contiguously by delivery_size")
            mappings.append(OutputGroup(group.start, group.end, group.start, group.end,
                                        float(group.start), float(group.end),
                                        dependency_end=group.dependency_end))
            expected = group.end
        if expected - proposal.command.retain_until > proposal.command.chunk_size:
            raise ValueError("published more than one native chunk")
        proposal.mapping += tuple(mappings)
        proposal.valid_end = expected
        proposal.generated = expected - proposal.command.retain_until == proposal.command.chunk_size
        for group in incoming:
            proposal.media[group.start] = {"media": group}
        self.predictions.extend(incoming)
        return True

    def finish(self, job_id, *, stable_state=None):
        if self.active_job is None or self.active_job.job_id != job_id:
            return False
        proposal = self.proposals[job_id]
        if proposal.failed:
            return False
        if not proposal.generated:
            raise ValueError("generator job finished without one complete native chunk")
        proposal.finished, proposal.stable_state = True, stable_state
        self.applied_condition_version = self.active_job.condition_version
        self.active_job = None
        self._event("GenerationFinished", job_id=job_id)
        return True

    def job_failed(self, job_id, reason):
        if self.active_job is None or self.active_job.job_id != job_id:
            return False
        self.proposals[job_id].failed = True
        self._event("JobFailed", job_id=job_id, reason=str(reason))
        return True  # slot held until all ranks are confirmed terminated

    def cancel_active(self, job_id, reason="superseded"):
        """Release an active proposal that stopped before publishing native output.

        Unlike worker-fault recovery, cooperative cancellation is acknowledged
        by every rank at a model safe point. It may therefore release the
        speculative lock, but only while no generated mapping/media exists.
        """
        if self.active_job is None or self.active_job.job_id != job_id:
            return False
        proposal = self.proposals[job_id]
        if proposal.generated or proposal.mapping or proposal.media:
            raise ValueError("cannot cooperatively cancel published proposal output")
        proposal.failed = True
        proposal.valid_end = proposal.command.retain_until
        self.locked_until = max(self.committed_until, self.decoder_protected_until)
        self.active_job = None
        self.pending_condition_version = (
            self.pending_condition_version or self.latest_condition_version
        )
        self.recovery = False
        self._event("GenerationCancelled", job_id=job_id, reason=str(reason))
        return True

    def worker_terminated(self, job_id):
        if self.active_job is None or self.active_job.job_id != job_id:
            return False
        proposal = self.proposals[job_id]
        if not proposal.failed:
            raise ValueError("termination acknowledgement requires failed job")
        # Never retract already committed output on a worker fault.
        proposal.valid_end = max(self.committed_until, proposal.command.retain_until)
        self.predictions = [g for g in self.predictions
                            if g.job_id != job_id or g.end <= proposal.valid_end]
        self.pending_condition_version = self.pending_condition_version or self.latest_condition_version
        self.active_job = None
        self.recovery = True
        self._event("WorkerTerminated", job_id=job_id)
        return True

    def deliver(self):
        group = next((g for g in self.predictions if g.start == self.committed_until), None)
        if group is None:
            return ()
        self.committed_until = group.end
        self.decoder_protected_until = max(
            self.decoder_protected_until, group.dependency_end or group.end,
            *(dependency.end for dependency in group.source_dependencies))
        self._event("OutputCommitted", job_id=group.job_id, end=group.end)
        return (group,)

    def advance_playback(self, position):
        if not self.played_until <= position <= self.committed_until:
            raise ValueError("playback must advance monotonically within committed output")
        self.played_until = position

    def compact_played(self, until):
        """Drop played coordinator payloads without inspecting native state.

        The runtime chooses its retry-retention boundary and separately owns
        worker releases. Source records referenced by still-retained media stay
        alive even when their own mapping has been compacted.
        """
        if type(until) is not int or not self.archived_until <= until <= self.played_until:
            raise ValueError("compaction must stay within acknowledged playback")
        if until != self.archived_until and until not in self.valid_boundaries:
            raise ValueError("compaction must end at an output group boundary")
        self.archived_until = until
        self.predictions = [g for g in self.predictions if g.end > until]
        for proposal in self.proposals.values():
            proposal.mapping = tuple(g for g in proposal.mapping if g.end > until)
            proposal.media = {
                start: kinds for start, kinds in proposal.media.items()
                if any(g.end > until for g in kinds.values())
            }
        referenced = {
            dependency.job_id
            for proposal in self.proposals.values()
            for kinds in proposal.media.values()
            for group in kinds.values()
            for dependency in group.source_dependencies
        }
        active_id = self.active_job.job_id if self.active_job is not None else None
        removed = []
        for job_id, proposal in list(self.proposals.items()):
            if (job_id != active_id and job_id not in referenced
                    and proposal.valid_end is not None and proposal.valid_end <= until):
                del self.proposals[job_id]
                removed.append(job_id)
        versions = {
            self.latest_condition_version, self.applied_condition_version,
            self.pending_condition_version,
            *(p.command.condition_version for p in self.proposals.values()),
        }
        self.conditions = {v: handle for v, handle in self.conditions.items() if v in versions}
        self.condition_ready.intersection_update(versions)
        return tuple(removed)

    def snapshot(self):
        return ControllerSnapshot(
            self.state, self.played_until, self.committed_until, self.locked_until,
            self.horizon, self.latest_condition_version, self.applied_condition_version,
            self.pending_condition_version, self.active_job, tuple(self.predictions),
            self.discarded_groups, self.horizon, self._frontier("video"),
            self._frontier("audio"), self._frontier("media"),
            self.decoder_protected_until, self.recovery)
