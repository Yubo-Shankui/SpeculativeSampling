"""Native LTX boundary/deadline evidence for the shared controller.

No sampling lives here. Readiness is a coordinator-clock estimate, initialized
from profiling when supplied and conservatively updated with actual decode
windows. Unsupported/missing state never gets rounded into a 'safe' boundary.
"""
import math

from speculative_sampling import BoundaryReadiness, ContinuationKind


class LtxLiveBoundaryAdapter:
    continuation_kind = ContinuationKind.RETAINED_KV

    def __init__(self, session):
        self.session = session
        self.commit_frontier_seconds = 0.

    def select_retain_until(self, requested, candidates, horizon):
        s = self.session
        return next((p for p in candidates if requested <= p <= horizon
                     and (s.serving_policy != "base" or p % s.controller.config.chunk_size == 0)), None)

    def readiness(self, count):
        s = self.session
        curve = s.readiness_curve
        # If conditioning became ready while an uncancellable native refill was
        # already in flight, the replacement cannot start until that rank group
        # releases its slot. This delay is frozen with the rewrite; it is not a
        # second boundary calculation at dispatch time.
        lead = s.budget + getattr(s, "rewrite_dispatch_delay_seconds", 0.0)
        if curve:
            # Keep the measured curve relative to its first group, updating
            # the DiT component online without erasing decoder tail costs.
            return tuple(lead + max(0., curve[i] - curve[0]) for i in range(count))
        # Development/bootstrap fallback only, explicitly uncalibrated.
        return tuple(lead + (i // s.mapper.group_latents) * s.decode_window_seconds
                     for i in range(count))

    def deadlines(self, boundary):
        s = self.session
        return tuple(s.mapper.seconds(boundary + i) - self.commit_frontier_seconds
                     for i in range(s.controller.config.chunk_size))

    def recovery_needed(self):
        s, c = self.session, self.session.controller
        ready = c.snapshot().media_ready
        # Only recover automatically for a demonstrable native audio tail gap:
        # video is decoded through the sampled horizon, playback exhausted AV,
        # and the mapped PCM ends before the last RGB interval.
        return (ready <= c.committed_until < c.horizon
                and c.snapshot().video_ready >= c.horizon
                and s.mapper.pcm_boundary(c.horizon, 48000) <
                s.mapper.rgb_boundary(c.horizon) * 48000 // s.mapper.fps)

    def boundary_readiness(self, boundary):
        s, c = self.session, self.session.controller
        context_ready = c.locked_until <= boundary <= c.horizon and not s.error
        decoder_ready = context_ready
        new_ready = self.readiness(c.config.chunk_size)
        cover_ready = True
        # Invalidate dependencies exactly as dispatch would. A retained group
        # crossing the cut needs native partial-window replay. If its PCM tail
        # falls after native PCM(boundary), it also needs the replacement's
        # first window, so its EARLIER deadline must be checked, not just j.
        for group, proposal in c._valid_mappings():
            if group.end > boundary or group.end <= c.committed_until:
                continue
            kinds = proposal.media.get(group.start, {})
            combined = kinds.get("media")
            ready_entries = ([combined] if combined is not None else
                             [kinds.get("video"), kinds.get("audio")])
            if all(entry is not None and (entry.dependency_end or entry.end) <= boundary
                   and all(d.end <= boundary for d in entry.source_dependencies)
                   for entry in ready_entries):
                continue
            # Native context is recoverable, but it is not yet ready media.
            # Count the missing retained windows serially before new output.
            deadline = s.mapper.seconds(group.start) - self.commit_frontier_seconds
            replay_start = max(c.committed_until, proposal.command.retain_until,
                               (group.start - proposal.command.retain_until)
                               // s.mapper.group_latents * s.mapper.group_latents
                               + proposal.command.retain_until)
            windows = max(1, math.ceil((boundary - replay_start) / s.mapper.group_latents))
            old_ready = windows * s.decode_window_seconds
            pcm_end = s.mapper.pcm_boundary(boundary, 48000)
            needs_replacement_audio = pcm_end < (
                s.mapper.rgb_boundary(group.end) * 48000 // s.mapper.fps
            )
            if needs_replacement_audio:
                old_ready = max(old_ready, new_ready[0])
                # If retained video is already decoded and only its final PCM
                # samples come from the replacement, those samples are not due
                # at the *start* of the RGB group.  They are due when native
                # PCM actually runs out, normally only 5--20 ms before the
                # group ends.  Using group.start here rounded that tiny tail
                # dependency into a whole 8-frame latent of refill lead.
                if (combined is None and kinds.get("video") is not None
                        and kinds.get("audio") is None):
                    deadline = pcm_end / 48000 - self.commit_frontier_seconds
            if old_ready > deadline:
                cover_ready = False
                break
        return BoundaryReadiness(bool(context_ready), bool(decoder_ready),
                                 cover_ready, new_ready)
