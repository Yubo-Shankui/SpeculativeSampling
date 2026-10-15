"""Bounded coordinator decode pump, independent of the DiT rank loop."""
from __future__ import annotations

import threading
import time

from speculative_sampling.runtime.recovery_journal import MediaCheckpointJournal
from speculative_sampling.runtime.rpc import RemoteCallError, RemoteWorkerRestarted, TensorClient
from .output_mapping import LtxOutputMapper
from .audio_timeline import AudioTimeline, AudioWindow


class MediaPump:
    def __init__(self, demo, endpoint, token, *, ahead_seconds=3, output=None,
                 recovery_timeout=600):
        self.demo = demo
        self.client = TensorClient(endpoint, token)
        # Pin the initial process epoch before the first mutating decode. If a
        # bootstrap decode loses its response and the service restarts, an
        # otherwise-uninitialized client cannot distinguish the fresh empty
        # service from its original owner and would receive a semantic KeyError
        # instead of the required epoch transition.
        health = self.client.call("health")
        if health.get("role") != "media" or health.get("ready") is not True:
            raise RuntimeError("media service is not ready")
        # Cleanup retries must not take the decoding client's lock or stall
        # first-media readiness when an old worker/session is unavailable.
        self.cleanup_client = TensorClient(endpoint, token, timeout=5)
        self.mapper = LtxOutputMapper()
        self.ahead_seconds = ahead_seconds
        self.recovery_timeout = float(recovery_timeout)
        self.journal = MediaCheckpointJournal(output) if output is not None else None
        self.pending = []
        self.retired = {}
        self.audio_timeline = None
        self.audio_session_id = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.run, name="ltx-media-pump", daemon=True)
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="ltx-media-cleanup", daemon=True)
        self.thread.start()
        self.cleanup_thread.start()

    def submit(self, task, *, prefetched=None):
        with self.lock:
            self.pending.append({
                "task": dict(task), "builder": None, "source": 0, "audio": 0,
                "prefetched": [] if prefetched is None else [prefetched],
            })

    def _valid(self, item):
        task = item["task"]
        session = self.demo.session
        if session is None or session.id != task["session"] or session.stopped or session.error:
            return False
        proposal = session.controller.proposals.get(task["proposal"])
        # A completed preview can still need a partial final-window replay
        # after an update. Keep it while any output is retained and unplayed.
        return (proposal is not None and proposal.valid_end is not None
                and proposal.valid_end > max(task["retain_until"],
                                            session.controller.archived_until))

    def _request(self, item):
        """Earliest missing valid group, translated to a native decode window."""
        session = self.demo.session
        task = item["task"]
        proposal = session.controller.proposals[task["proposal"]]
        ready = {start for start, kinds in proposal.media.items()
                 if "video" in kinds or "media" in kinds}
        missing = next((group for group in proposal.mapping
                        if group.end <= proposal.valid_end and group.start not in ready
                        and group.end > session.controller.committed_until), None)
        if missing is None:
            return None
        start = task["retain_until"] + (
            (missing.start - task["retain_until"]) // self.mapper.group_latents
        ) * self.mapper.group_latents
        return {"session": task["session"], "proposal_id": task["proposal"],
                "requested_start": start, "end_limit": proposal.valid_end}

    def _retire(self, item):
        """Called under both coordinator locks; never performs network I/O."""
        task = item["task"]
        builder = item["builder"]
        if builder is not None:
            builder.close()
            # Proposal IDs restart in a new session. Do not remove a new
            # session's preview just because an old item has the same ID.
            if self.demo.previews.get(task["proposal"]) is builder.preview:
                self.demo.previews.pop(task["proposal"], None)
        self.pending.remove(item)
        self.retired[(task["session"], task["proposal"])] = 0.0

    def _release_retired(self):
        """One idempotent release per tick, outside playback/dispatch locks.

        A played proposal may still be the native continuation source. A
        worker refusal remains queued until a later prepare promotes it.
        Session shutdown instead releases the complete worker session.
        """
        now = time.monotonic()
        with self.lock, self.demo.lock:
            entry = next((key for key, retry in self.retired.items() if retry <= now), None)
            if entry is None:
                return
            session_id, proposal_id = entry
            session = self.demo.session
            closed = (session is None or session.id != session_id
                      or session.stopped or session.error)
            self.retired[entry] = now + 1.0
            if (closed and session is not None and session.id == session_id
                    and (session.busy is not None or session.controller.active_job is not None)):
                # A stopped/failed session can still receive its last latent
                # handoff. Keep the remote session until the model group has
                # returned or a supervisor acknowledges its termination.
                return
        try:
            result = (self.cleanup_client.call("close", session=session_id) if closed else
                      self.cleanup_client.call("release", session=session_id, proposal_id=proposal_id))
        except RemoteWorkerRestarted as restart:
            # Cleanup identities are idempotent and a restored/new worker may
            # legitimately no longer own them. Adopt its epoch, then retry.
            self.cleanup_client.accept_restart(restart)
            with self.lock:
                if entry in self.retired:
                    self.retired[entry] = time.monotonic()
            return
        except Exception:
            # Keep the identity for a retry; a decode error still takes the
            # normal session failure path. Never label failed cleanup success.
            with self.lock:
                if entry in self.retired:
                    self.retired[entry] = time.monotonic() + 5.0
            return
        if closed or result.get("released") is True:
            with self.lock:
                for key in list(self.retired):
                    if key == entry or (closed and key[0] == session_id):
                        del self.retired[key]

    def _cleanup_loop(self):
        while not self.stopping.wait(.5):
            self._release_retired()

    def run(self):
        while not self.stopping.wait(.01):
            item = None
            request = None
            with self.lock, self.demo.lock:
                for candidate in list(self.pending):
                    if not self._valid(candidate):
                        self._retire(candidate)
                        continue
                    if candidate["prefetched"]:
                        item = candidate
                        result = candidate["prefetched"].pop(0)
                        break
                    session = self.demo.session
                    ready = session.controller.snapshot().media_ready
                    # Decode enough cover for the current guard, not merely a
                    # smaller fixed watermark that makes every update infeasible.
                    target = max(self.ahead_seconds, session.budget)
                    if (session.mapper.seconds(ready) - session.played / 24 >= target
                            and candidate["builder"] is not None):
                        # Cover is already healthy, so throttle additional old
                        # windows but keep scanning. Every newly handed-off
                        # proposal still gets its first window immediately;
                        # otherwise refill decode throttling pollutes the shared
                        # first-media EMA and can delay a later branch.
                        continue
                    request = self._request(candidate)
                    if request is None:
                        continue
                    item = candidate  # retained prefix precedes new proposals
                    break
            if item is None:
                continue
            task = item["task"]
            try:
                if request is not None:
                    result = self._decode_with_recovery(task, request)
                if result["done"]:
                    continue
                self._publish(item, result)
            except Exception as exc:
                with self.demo.lock:
                    session = self.demo.session
                    if session is not None and session.id == task["session"]:
                        session.fail(f"media worker: {type(exc).__name__}: {exc}")

    def _recovery_event(self, task, begin, value):
        with self.demo.lock:
            session = self.demo.session
            if session is None or session.id != task["session"]:
                return False
            if begin:
                return session.begin_recovery("media", value)
            return session.finish_recovery("media", value)

    def _restore(self, task, restart):
        if self.journal is None:
            raise RuntimeError("media recovery requires a local checkpoint journal")
        self.client.accept_restart(restart)
        checkpoint = self.journal.load(task["session"])
        report = self.client.call("restore_checkpoint", checkpoint=checkpoint)
        if report.get("restored") is not True:
            raise RuntimeError("media worker did not confirm checkpoint restore")
        return report

    def _decode_with_recovery(self, task, request):
        deadline = time.monotonic() + self.recovery_timeout
        recovering = False
        while True:
            try:
                result = self.client.call("decode_next", **request)
                if recovering:
                    self._recovery_event(task, False, {
                        "recovered": True, "same_epoch_retry": True,
                        "epoch": self.client.epoch})
                return result
            except RemoteWorkerRestarted as restart:
                if not recovering:
                    self._recovery_event(task, True, str(restart))
                recovering = True
                report = self._restore(task, restart)
                self._recovery_event(task, False, report)
                recovering = False
            except RemoteCallError:
                raise
            except Exception as exc:
                if not recovering:
                    self._recovery_event(
                        task, True, f"{type(exc).__name__}: {exc}")
                recovering = True
                if time.monotonic() >= deadline:
                    raise TimeoutError("media worker did not recover before deadline") from exc
                time.sleep(.5)

    def _publish(self, item, result):
        import torch
        from speculative_sampling.runtime.media import PreviewBuilder
        task = item["task"]
        source_start = self.mapper.rgb_boundary(result["start"]) - task["boundary"]
        source_end = self.mapper.rgb_boundary(result["end"]) - task["boundary"]
        # Serialize provenance validation, preview mutation and publication with
        # dispatch/commit. Never overwrite retained preview bytes using an
        # in-flight callback that depended on a now-rejected suffix.
        with self.demo.lock:
            if self.stopping.is_set():
                return  # timed-out close must not resurrect preview/media state
            session = self.demo.session
            if session is None or session.id != task["session"] or session.stopped or session.error:
                return
            proposal = session.controller.proposals.get(task["proposal"])
            if proposal is None or result["end"] > (proposal.valid_end or 0):
                session.event("stale_media_window", proposal=task["proposal"],
                              start=result["start"], end=result["end"])
                return
            self._install_window(item, result, source_start, source_end, session, PreviewBuilder, torch)

    def _install_window(self, item, result, source_start, source_end, session, PreviewBuilder, torch):
        task = item["task"]
        if item["builder"] is None:
            item["builder"] = PreviewBuilder(
                task["output"],
                width=(self.demo.args.rtc_width if task["transport"] == "rtc"
                       else self.demo.args.preview_width),
                transport=task["transport"], proposal_frames=task["proposal_frames"])
            item["builder"].audio(torch.zeros(2, 0), result["sample_rate"])
        builder = item["builder"]
        if source_start < item["source"]:
            if task["boundary"] + source_start < session.committed:
                raise RuntimeError("decoder replay would mutate committed preview bytes")
            preview = builder.preview
            if preview.raw_frames is not None:
                del preview.raw_frames[source_start:]
            else:
                del preview.frames[source_start:]
            item["source"] = source_start
            item["audio"] = self.mapper.pcm_boundary(
                result["start"], result["sample_rate"]
            ) - self.mapper.pcm_boundary(task["retain_until"], result["sample_rate"])
        if source_start != item["source"]:
            raise RuntimeError("decoder window would leave a preview gap")
        if getattr(self, "audio_session_id", None) != session.id:
            self.audio_session_id = session.id
            self.audio_timeline = AudioTimeline(result["sample_rate"])
        if self.audio_timeline.sample_rate != result["sample_rate"]:
            raise ValueError("PCM sample rate changed inside session")
        self.audio_timeline.append(AudioWindow(
            task["proposal"], result["start"], result["end"],
            self.mapper.pcm_boundary(result["start"], result["sample_rate"]),
            result["audio"][0]), session.controller)
        item["audio"] = self.mapper.pcm_boundary(result["end"], result["sample_rate"]) - (
            self.mapper.pcm_boundary(task["retain_until"], result["sample_rate"]))
        video = ((result["video"] + 1) / 2).clamp(0, 1)[0].permute(1, 0, 2, 3)
        preview = builder.video(item["source"], video)
        item["source"] += video.shape[0]
        if item["source"] != source_end:
            raise RuntimeError("decoder RGB output does not match its source interval")
        self.demo.previews[task["proposal"]] = preview
        session.publish_media_window(
            task, source_start, source_end, result["end"], kind="video")
        # A replacement's initial PCM can finish the retained prefix's last
        # RGB group. Join the exact intervals and attach cross-proposal
        # dependencies, so committing that group also protects its audio source.
        self._publish_audio(session, item)
        session.event(
            "media_window_ready", proposal=task["proposal"],
            start=result["start"], end=result["end"],
            decode_seconds=result["wall_seconds"],
            audio_samples=item["audio"], video_frames=item["source"])
        session.record_decode_window(result["wall_seconds"])
        if item["source"] == task["proposal_frames"]:
            # Do not close the builder/thread pool here: an uncommitted final
            # window can be replaced by a shorter replay on a later branch.
            session.event("media_proposal_decoded", proposal=task["proposal"])

    def _publish_audio(self, session, current):
        sr = self.audio_timeline.sample_rate
        items = list(getattr(self, "pending", []))
        if not any(item is current for item in items):
            items.append(current)
        for item in items:
            task = item["task"]
            if task["session"] != session.id or item["builder"] is None:
                continue
            proposal = session.controller.proposals.get(task["proposal"])
            if proposal is None:
                continue
            for group in proposal.mapping:
                if group.end > (proposal.valid_end or 0) or group.end <= session.controller.committed_until:
                    continue
                kinds = proposal.media.get(group.start, {})
                if "video" not in kinds or "audio" in kinds or "media" in kinds:
                    continue
                rgb_start = self.mapper.rgb_boundary(group.start)
                rgb_end = self.mapper.rgb_boundary(group.end)
                joined = self.audio_timeline.read(
                    rgb_start * sr // self.mapper.fps,
                    rgb_end * sr // self.mapper.fps, session.controller)
                if joined is None:
                    continue
                pcm, dependencies = joined
                item["builder"].audio_chunk(
                    (rgb_start - task["boundary"]) * sr // self.mapper.fps, pcm, sr)
                session.publish_media_window(
                    task, rgb_start-task["boundary"], rgb_end-task["boundary"],
                    group.end, kind="audio", source_dependencies=dependencies)
        self.audio_timeline.prune(session.played * sr // self.mapper.fps, session.controller)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        self.cleanup_thread.join(timeout=6)
        with self.lock:
            for item in self.pending:
                if item["builder"]:
                    item["builder"].close()
            self.pending.clear()
