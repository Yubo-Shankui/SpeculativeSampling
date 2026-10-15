"""Dedicated LTX text/media services; neither joins the DiT process group."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

from .decoder_state import streaming_decoder_class
from .decoder_checkpoint import (
    checkpoint_fingerprint as decoder_checkpoint_fingerprint,
    export_decoder_state,
    restore_decoder_state,
)
from .output_mapping import LtxOutputMapper


class TextService:
    operations = {"health", "encode", "encode_handle", "resolve"}

    def __init__(self, base_path, gemma_path, *, device, max_length, segment_length=256):
        from vforge.models.ltx.text_encoder import PromptEncoder
        self.encoder = PromptEncoder(base_path, gemma_path, device=device, max_length=max_length)
        self.segment_length = int(segment_length)
        self.cache = {}
        self.handle_prompts = {}
        self.encode("warmup")

    def health(self):
        return {"ready": True, "role": "text"}

    def encode(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 5000:
            raise ValueError("invalid prompt")
        started = time.perf_counter()
        if prompt not in self.cache:
            result = self.encoder.encode_timed_segment(
                prompt, segment_length=self.segment_length
            )
            self.cache[prompt] = {key: value.detach().cpu() if hasattr(value, "detach") else value
                                  for key, value in result.items()}
            while len(self.cache) > 128:
                del self.cache[next(iter(self.cache))]
        return {"context": self.cache[prompt], "wall_seconds": time.perf_counter() - started}

    def encode_handle(self, prompt):
        import hashlib
        import json
        encoded = self.encode(prompt)
        digest = hashlib.sha256()
        for name, value in sorted(encoded["context"].items()):
            digest.update(name.encode())
            if hasattr(value, "detach"):
                import torch
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
        fingerprint = digest.hexdigest()
        key = hashlib.sha256(prompt.encode()).hexdigest()
        # Keep handle metadata bounded to the actual context cache as well.
        self.handle_prompts = {k: v for k, v in self.handle_prompts.items() if v in self.cache}
        self.handle_prompts[key] = prompt
        return {"key": key, "fingerprint": fingerprint}

    def resolve(self, key):
        prompt = self.handle_prompts.get(key)
        if prompt is None or prompt not in self.cache:
            raise ValueError("condition handle expired; encode again before dispatch")
        return self.cache[prompt]


@dataclass
class MediaSession:
    reference: object
    official: object
    official_until: int = 0
    proposals: dict = field(default_factory=dict)
    reference_identity: dict | None = None
    reference_fingerprint: str | None = None


class MediaService:
    operations = {"health", "warm_reference", "encode_reference", "prepare", "install", "install_nccl", "decode_next", "close",
                  "verify_decoder_forks", "release", "state_summary", "checkpoint",
                  "restore_checkpoint"}

    def __init__(self, base_path, upstream_repo, *, device, chunk_latents=45):
        import torch
        from vforge.models.ltx.decoding import LtxDecoder
        self.torch = torch
        self.device = torch.device(device)
        self.base_path = base_path
        self.upstream_repo = Path(upstream_repo)
        loader = LtxDecoder(base_path, self.device)
        loader._load()
        self.models = tuple(m.to(self.device).eval().requires_grad_(False)
                            for m in loader._models)
        self.decoder_type = streaming_decoder_class(upstream_repo)
        self.mapper = LtxOutputMapper()
        self.chunk_latents = int(chunk_latents)
        if self.chunk_latents not in (3, 15, 45):
            raise ValueError("unsupported native latent chunk size")
        self.tensor_channel = None
        self.sessions = {}
        # The browser demo uses one immutable reference image.  Keep its VAE
        # latent resident across short-lived browser sessions so a reconnect
        # does not pay the tiled VAE encode again.
        self.reference_cache = {}

    def _tensor_fingerprint(self, *items):
        digest = hashlib.sha256()
        for name, value in items:
            value = value.detach().cpu().contiguous()
            digest.update(str(name).encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def health(self):
        return {"ready": True, "role": "media"}

    def state_summary(self, session):
        record = self.sessions.get(session)
        if record is None:
            return {"present": False, "proposals": 0, "snapshots": 0}
        return {
            "present": True, "official_until": record.official_until,
            "proposals": len(record.proposals),
            "snapshots": sum(len(p["snapshots"]) for p in record.proposals.values()),
            "proposal_ids": sorted(record.proposals),
        }

    @staticmethod
    def _reference_identity(path, width, height):
        return {
            "path": str(Path(path).expanduser().resolve()),
            "width": int(width), "height": int(height),
        }

    @staticmethod
    def _reference_cache_key(identity):
        # The web client round-trips the configured reference through a canvas
        # and the coordinator saves it under a per-session path. Keying by the
        # path (or encoded PNG bytes) therefore misses even though the RGB
        # image is identical.
        from PIL import Image
        with Image.open(identity["path"]) as source:
            image = source.convert("RGB")
            digest = hashlib.sha256()
            digest.update(str(image.size).encode())
            digest.update(image.tobytes())
        return (digest.hexdigest(), identity["width"], identity["height"])

    def _load_reference(self, identity):
        key = self._reference_cache_key(identity)
        cached = self.reference_cache.get(key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        import sys
        root = self.upstream_repo
        for source in (root / "src", *(root / "packages" / name / "src" for name in (
            "ltx-core", "ltx-pipelines", "ltx-tf", "ltx-trainer", "ltx-dmd"
        ))):
            if str(source) not in sys.path:
                sys.path.append(str(source))
        from ltx_interactive.cli import encode_reference_image
        with self.torch.inference_mode():
            latent, _ = encode_reference_image(
                Path(identity["path"]), self.base_path,
                identity["width"], identity["height"], self.device)
        # The upstream helper deliberately returns CPU latents. Retain the
        # immutable reference on the media GPU for all later sessions.
        latent = latent.to(self.device, dtype=next(self.models[0].parameters()).dtype)
        cached = {
            "latent": latent,
            "fingerprint": self._tensor_fingerprint(("reference", latent)),
            "wall_seconds": time.perf_counter() - started,
        }
        self.reference_cache[key] = cached
        return cached

    def warm_reference(self, path, width, height):
        identity = self._reference_identity(path, width, height)
        started = time.perf_counter()
        cached = self._load_reference(identity)
        return {
            "ready": True,
            "fingerprint": cached["fingerprint"],
            "wall_seconds": time.perf_counter() - started,
            "encode_wall_seconds": cached["wall_seconds"],
        }

    def encode_reference(self, session, path, width, height):
        identity = self._reference_identity(path, width, height)
        if session in self.sessions:
            record = self.sessions[session]
            if record.reference_identity == identity:
                # An uncertain RPC completion or a restarted DiT group may
                # repeat reference acquisition. Return the already-bound
                # tensor; never encode or replace it under the same session.
                return record.reference.detach().cpu()
            raise ValueError("session reference is immutable; close before replacing")
        if self.sessions:
            raise RuntimeError("media admission is limited to one session")
        cached = self._load_reference(identity)
        decoder = self.decoder_type(*self.models)
        latent = cached["latent"]
        self.sessions[session] = MediaSession(
            latent, decoder, reference_identity=identity,
            reference_fingerprint=cached["fingerprint"])
        return latent.detach().cpu()

    def _feed(self, state, proposal, start, end):
        """One native <=3-new-latent window, with one-time reference inclusion."""
        torch = self.torch
        a0 = self.mapper.audio_boundary(start) - self.mapper.audio_boundary(proposal["start"])
        a1 = self.mapper.audio_boundary(end) - self.mapper.audio_boundary(proposal["start"])
        video = proposal["video"][:, :, start - proposal["start"]:end - proposal["start"]]
        audio = proposal["audio"][:, :, a0:a1]
        if start == 0:
            video = torch.cat((proposal["reference"], video), dim=2)
        with torch.inference_mode():
            video_out, audio_out, event = state.feed_variable(video, audio)
        expected = self.mapper.rgb_boundary(end) - self.mapper.rgb_boundary(start)
        if video_out.shape[2] != expected:
            raise RuntimeError(f"native decoder emitted {video_out.shape[2]} != {expected} frames")
        return video_out, audio_out, event

    def _proposal_fingerprint(self, start, video, audio):
        return self._tensor_fingerprint(
            (f"video:{int(start)}", video), (f"audio:{int(start)}", audio))

    @staticmethod
    def _media_checkpoint_fingerprint(checkpoint, decoder_fingerprint):
        metadata = {
            "schema_version": checkpoint["schema_version"],
            "session": checkpoint["session"],
            "reference_identity": checkpoint["reference_identity"],
            "reference_fingerprint": checkpoint["reference_fingerprint"],
            "official_until": checkpoint["official_until"],
            "official_decoder_fingerprint": decoder_fingerprint,
            "proposals": [{
                key: proposal[key] for key in (
                    "proposal_id", "start", "end", "valid_end", "cursor",
                    "content_fingerprint")
            } for proposal in checkpoint["proposals"]],
        }
        return hashlib.sha256(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _is_forward_checkpoint(current, saved, current_decoder_fingerprint,
                               saved_decoder_fingerprint):
        """Prove that another client restored ``saved`` and only decoded ahead.

        The coordinator decode pump and distributed DiT service have separate
        RPC clients.  After one media restart they can observe the epoch in
        either order.  The later observer must not rewind decoder state, but it
        may safely adopt a live state whose immutable bytes and official
        decoder are identical and whose speculative cursors only moved
        forward.
        """
        for name in ("schema_version", "session", "reference_identity",
                     "reference_fingerprint", "official_until"):
            if current[name] != saved[name]:
                return False
        if current_decoder_fingerprint != saved_decoder_fingerprint:
            return False
        current_proposals = {item["proposal_id"]: item
                             for item in current["proposals"]}
        saved_proposals = {item["proposal_id"]: item
                           for item in saved["proposals"]}
        if set(current_proposals) != set(saved_proposals):
            return False
        immutable = ("proposal_id", "start", "end", "valid_end",
                     "content_fingerprint")
        for proposal_id, before in saved_proposals.items():
            after = current_proposals[proposal_id]
            if any(after[name] != before[name] for name in immutable):
                return False
            if after["cursor"] < before["cursor"]:
                return False
        return True

    def checkpoint(self, session):
        """Return bounded CPU state sufficient to reconstruct media history."""
        record = self.sessions[session]
        official = export_decoder_state(record.official)
        proposals = []
        for proposal_id, proposal in sorted(record.proposals.items()):
            if proposal["valid_end"] <= record.official_until:
                continue
            if proposal["start"] != record.official_until:
                raise RuntimeError("live proposal does not start at official media frontier")
            proposals.append({
                "proposal_id": int(proposal_id),
                "start": int(proposal["start"]),
                "end": int(proposal["end"]),
                "valid_end": int(proposal["valid_end"]),
                "cursor": int(proposal["cursor"]),
                "content_fingerprint": proposal["content_fingerprint"],
                "video": proposal["video"].detach().cpu().clone(),
                "audio": proposal["audio"].detach().cpu().clone(),
            })
        result = {
            "schema_version": 1,
            "session": session,
            "reference_identity": dict(record.reference_identity),
            "reference_fingerprint": record.reference_fingerprint,
            "reference": record.reference.detach().cpu().clone(),
            "official_until": int(record.official_until),
            "official_decoder": official,
            "proposals": proposals,
        }
        decoder_fingerprint = decoder_checkpoint_fingerprint(official, self.torch)
        result["checkpoint_fingerprint"] = self._media_checkpoint_fingerprint(
            result, decoder_fingerprint)
        return result

    def restore_checkpoint(self, checkpoint):
        """Rebuild one session after a media process restart and verify bytes."""
        required = {
            "schema_version", "session", "reference_identity", "reference_fingerprint",
            "reference", "official_until", "official_decoder", "proposals",
            "checkpoint_fingerprint",
        }
        if not isinstance(checkpoint, dict) or set(checkpoint) != required:
            raise ValueError("invalid media checkpoint structure")
        if checkpoint["schema_version"] != 1:
            raise ValueError("unsupported media checkpoint schema")
        session = checkpoint["session"]
        if not isinstance(session, str) or not session:
            raise ValueError("media checkpoint requires a session identity")
        decoder_fingerprint = decoder_checkpoint_fingerprint(
            checkpoint["official_decoder"], self.torch)
        expected = self._media_checkpoint_fingerprint(checkpoint, decoder_fingerprint)
        if checkpoint["checkpoint_fingerprint"] != expected:
            raise ValueError("media checkpoint fingerprint mismatch")
        reference_fingerprint = self._tensor_fingerprint(
            ("reference", checkpoint["reference"]))
        if reference_fingerprint != checkpoint["reference_fingerprint"]:
            raise ValueError("media checkpoint reference bytes changed")
        if self.sessions:
            existing = self.sessions.get(session)
            if existing is None:
                raise RuntimeError("media admission is limited to one session")
            current = self.checkpoint(session)
            if current["checkpoint_fingerprint"] == expected:
                return {"restored": True, "idempotent": True,
                        "checkpoint_fingerprint": expected}
            current_decoder_fingerprint = decoder_checkpoint_fingerprint(
                current["official_decoder"], self.torch)
            if self._is_forward_checkpoint(
                    current, checkpoint, current_decoder_fingerprint,
                    decoder_fingerprint):
                return {
                    "restored": True,
                    "idempotent": True,
                    "advanced": True,
                    "checkpoint_fingerprint": current["checkpoint_fingerprint"],
                    "restored_from_checkpoint_fingerprint": expected,
                }
            raise ValueError("live media session differs from restore checkpoint")
        official_until = checkpoint["official_until"]
        if type(official_until) is not int or official_until < 0:
            raise ValueError("invalid official media frontier")
        official = restore_decoder_state(
            self.decoder_type, self.models, checkpoint["official_decoder"],
            device=self.device)
        reference = checkpoint["reference"].to(
            self.device, dtype=next(self.models[0].parameters()).dtype)
        record = MediaSession(
            reference, official, official_until=official_until,
            reference_identity=dict(checkpoint["reference_identity"]),
            reference_fingerprint=reference_fingerprint)
        for saved in checkpoint["proposals"]:
            fields = {"proposal_id", "start", "end", "valid_end", "cursor",
                      "content_fingerprint", "video", "audio"}
            if not isinstance(saved, dict) or set(saved) != fields:
                raise ValueError("invalid media proposal checkpoint")
            start, end = saved["start"], saved["end"]
            if (type(saved["proposal_id"]) is not int or type(start) is not int
                    or type(end) is not int or start != official_until
                    or end != start + self.chunk_latents
                    or not start <= saved["cursor"] <= saved["valid_end"] <= end):
                raise ValueError("invalid media proposal checkpoint frontier")
            actual = self._proposal_fingerprint(start, saved["video"], saved["audio"])
            if actual != saved["content_fingerprint"]:
                raise ValueError("media proposal checkpoint bytes changed")
            proposal = {
                "start": start, "end": end, "valid_end": saved["valid_end"],
                "cursor": start, "state": official.fork(),
                "snapshots": {start: official.fork()},
                "video": saved["video"].to(
                    self.device, dtype=next(self.models[0].parameters()).dtype),
                "audio": saved["audio"].to(
                    self.device, dtype=next(self.models[1].parameters()).dtype),
                "reference": reference, "content_fingerprint": actual,
            }
            while proposal["cursor"] < saved["cursor"]:
                stop = min(saved["cursor"], proposal["cursor"] + self.mapper.group_latents)
                self._feed(proposal["state"], proposal, proposal["cursor"], stop)
                proposal["cursor"] = stop
                proposal["snapshots"][stop] = proposal["state"].fork()
            record.proposals[saved["proposal_id"]] = proposal
        self.sessions[session] = record
        restored = self.checkpoint(session)
        if restored["checkpoint_fingerprint"] != expected:
            self.sessions.pop(session, None)
            raise RuntimeError("restored media state does not match checkpoint")
        return {"restored": True, "idempotent": False,
                "checkpoint_fingerprint": expected}

    def prepare(self, session, retain_until):
        """Promote a saved window state, replay only the unmatched partial tail."""
        record = self.sessions[session]
        if retain_until < record.official_until:
            raise ValueError("decoder cannot move before official prefix")
        if retain_until == record.official_until:
            return {"ready": True}
        proposal = next((p for p in record.proposals.values()
                         if p["start"] == record.official_until and p["end"] >= retain_until), None)
        if proposal is None:
            raise ValueError("retained decoder source no longer exists")
        starts = [index for index in proposal["snapshots"] if index <= retain_until]
        position = max(starts)
        state = proposal["snapshots"][position].fork()
        while position < retain_until:
            end = min(retain_until, position + self.mapper.group_latents)
            self._feed(state, proposal, position, end)
            position = end
        rewinds = []
        # Validate every needed snapshot before advancing official state.
        for old in record.proposals.values():
            cut = min(old["valid_end"], retain_until)
            start = old["start"]
            if start < cut < old["end"] and (cut - start) % self.mapper.group_latents:
                window_start = start + ((cut - start) // self.mapper.group_latents) * self.mapper.group_latents
                if old["cursor"] > window_start:
                    if window_start not in old["snapshots"]:
                        raise RuntimeError("missing snapshot for invalidated decoder window")
                    rewinds.append((old, window_start))
        record.official = state
        record.official_until = retain_until
        for old, window_start in rewinds:
            old["state"] = old["snapshots"][window_start].fork()
            old["cursor"] = window_start
        # Keep old retained decode available, but invalidate every old suffix.
        for old in record.proposals.values():
            old["valid_end"] = min(old["valid_end"], retain_until)
            # A window that straddles the cut depended on the rejected suffix.
            # Its uncommitted retained frames must be decoded again from the
            # last unaffected snapshot, not silently accepted by cropping RGB.
            old["snapshots"] = {p: s for p, s in old["snapshots"].items() if p <= old["valid_end"]}
        return {"ready": True, "retain_until": retain_until}

    def install(self, session, proposal_id, start, video, audio):
        record = self.sessions[session]
        if start != record.official_until:
            raise ValueError("prepare retained decoder prefix before installing new source")
        fingerprint = self._proposal_fingerprint(start, video, audio)
        if proposal_id in record.proposals:
            existing = record.proposals[proposal_id]
            if existing.get("content_fingerprint") == fingerprint:
                return {"installed": True, "idempotent": True,
                        "content_fingerprint": fingerprint}
            raise ValueError("duplicate proposal id has different latent content")
        count = video.shape[2]
        if count != self.chunk_latents:
            raise ValueError(f"media service requires a complete K{self.chunk_latents}")
        expected_audio = self.mapper.audio_boundary(start + count) - self.mapper.audio_boundary(start)
        if audio.shape[2] != expected_audio:
            raise ValueError("audio payload disagrees with native time mapping")
        state = record.official.fork()
        record.proposals[proposal_id] = {
            "start": start, "end": start + count, "valid_end": start + count,
            "cursor": start, "state": state,
            "snapshots": {start: record.official.fork()},
            "video": video.to(self.device, dtype=next(self.models[0].parameters()).dtype),
            "audio": audio.to(self.device, dtype=next(self.models[1].parameters()).dtype),
            "reference": record.reference,
            "content_fingerprint": fingerprint,
        }
        # Coordinator release follows playback, not worker decode completion:
        # a just-finished old callback may still be queued on the network.
        return {"installed": True, "idempotent": False,
                "content_fingerprint": fingerprint}

    def install_nccl(self, session, proposal_id, start, descriptors):
        """Receive proposal tensors directly into the media GPU over NCCL."""
        if self.tensor_channel is None:
            raise RuntimeError("media NCCL tensor channel is not configured")
        video, audio = self.tensor_channel.receive(descriptors)
        # The direct channel is ordered and single-flight. Always consume the
        # tensors before checking idempotence so a retried sender cannot hang.
        record = self.sessions[session]
        existing = record.proposals.get(proposal_id)
        if existing is not None:
            same = (tuple(existing["video"].shape) == tuple(video.shape)
                    and tuple(existing["audio"].shape) == tuple(audio.shape)
                    and self.torch.equal(existing["video"], video)
                    and self.torch.equal(existing["audio"], audio))
            if not same:
                raise ValueError("duplicate proposal id has different latent content")
            return {"installed": True, "idempotent": True,
                    "content_fingerprint": existing["content_fingerprint"]}
        # Avoid a full GPU->CPU copy merely to hash bytes. This identity is for
        # the volatile low-latency path; exact recovery remains on RPC mode.
        identity = json.dumps({
            "session": session, "proposal": int(proposal_id), "start": int(start),
            "descriptors": descriptors,
        }, sort_keys=True, separators=(",", ":")).encode()
        fingerprint = hashlib.sha256(identity).hexdigest()
        if start != record.official_until:
            raise ValueError("prepare retained decoder prefix before installing new source")
        count = video.shape[2]
        if count != self.chunk_latents:
            raise ValueError(f"media service requires a complete K{self.chunk_latents}")
        expected_audio = self.mapper.audio_boundary(start + count) - self.mapper.audio_boundary(start)
        if audio.shape[2] != expected_audio:
            raise ValueError("audio payload disagrees with native time mapping")
        state = record.official.fork()
        record.proposals[proposal_id] = {
            "start": start, "end": start + count, "valid_end": start + count,
            "cursor": start, "state": state,
            "snapshots": {start: record.official.fork()},
            "video": video, "audio": audio, "reference": record.reference,
            "content_fingerprint": fingerprint,
        }
        return {"installed": True, "idempotent": False,
                "content_fingerprint": fingerprint}

    def release(self, session, proposal_id):
        record = self.sessions.get(session)
        if record is not None:
            proposal = record.proposals.get(proposal_id)
            if proposal is not None and proposal["end"] > record.official_until:
                # A shortened old proposal can be released once its valid
                # retained range has become official. Never drop current KV
                # continuation source just because pixels were already played.
                if proposal["valid_end"] > record.official_until:
                    return {"released": False}
            record.proposals.pop(proposal_id, None)
        return {"released": True}

    def decode_next(self, session, proposal_id, *, requested_start=None, end_limit=None):
        record = self.sessions[session]
        proposal = record.proposals[proposal_id]
        start = proposal["cursor"] if requested_start is None else requested_start
        valid_end = proposal["valid_end"]
        if end_limit is not None:
            if type(end_limit) is not int or not proposal["start"] <= end_limit <= proposal["end"]:
                raise ValueError("invalid decoder end limit")
            valid_end = min(valid_end, end_limit)
        if type(start) is not int or not proposal["start"] <= start <= valid_end:
            raise ValueError("invalid decoder start")
        if start >= valid_end:
            return {"done": True}
        if start != proposal["cursor"]:
            if start not in proposal["snapshots"]:
                saved = max((position for position in proposal["snapshots"] if position < start),
                            default=None)
                if saved is None:
                    raise ValueError("requested decoder restart has no earlier saved state")
                state = proposal["snapshots"][saved].fork()
                position = saved
                while position < start:
                    stop = min(start, position + self.mapper.group_latents)
                    self._feed(state, proposal, position, stop)
                    position = stop
                    proposal["snapshots"][position] = state.fork()
            else:
                state = proposal["snapshots"][start].fork()
        else:
            state = proposal["state"]
        end = min(valid_end, start + self.mapper.group_latents)
        started = time.perf_counter()
        video, audio, event = self._feed(state, proposal, start, end)
        proposal["state"] = state
        proposal["cursor"] = end
        proposal["snapshots"][end] = proposal["state"].fork()
        # CPU copy fences CUDA work; duration includes the actual decoder and
        # D2H, not merely the asynchronous kernel launch interval.
        video, audio = video.detach().cpu(), audio.detach().cpu()
        return {
            "done": False, "start": start, "end": end,
            "video": video, "audio": audio,
            "sample_rate": proposal["state"].output_sample_rate,
            "wall_seconds": time.perf_counter() - started,
            "video_window_latents": event.video_window_latents,
            "audio_payload_latents": event.audio_payload_latents,
        }

    def close(self, session):
        self.sessions.pop(session, None)
        return {"closed": True}

    def verify_decoder_forks(self, session, proposal_id):
        """Real decoder gate: snapshot/partial replay vs replay from branch base.

        Uses the same pinned decoder and exact latent tensors, but reconstructs
        the reference state independently. Never promotes any verification state
        into the live session. Does not claim native generator/KV correctness.
        """
        torch = self.torch
        record = self.sessions[session]
        source = record.proposals[proposal_id]
        names = ("_video_history", "_audio_history", "_mel_history", "_previous_wave_sample")

        def equal_state(left, right):
            return left.block_index == right.block_index and all(
                (getattr(left, name) is None and getattr(right, name) is None)
                or (getattr(left, name) is not None and getattr(right, name) is not None
                    and torch.equal(getattr(left, name), getattr(right, name)))
                for name in names)

        def replay(state, position, end):
            while position < end:
                stop = min(end, position + self.mapper.group_latents)
                self._feed(state, source, position, stop)
                position = stop
            return state

        original = source["state"].fork()
        reports = []
        for count in (1, 26, 44, 45):
            boundary = source["start"] + count
            base = source["snapshots"][source["start"]].fork()
            reference = replay(base, source["start"], boundary)
            saved = max(p for p in source["snapshots"] if p <= boundary)
            candidate = replay(source["snapshots"][saved].fork(), saved, boundary)
            states_match = equal_state(reference, candidate)
            # Different continuation payload to avoid accidentally validating
            # merely replaying the discarded suffix of the original proposal.
            replacement = {
                **source, "start": boundary,
                "video": -source["video"][:, :, :3],
                "audio": -source["audio"][:, :, :(
                    self.mapper.audio_boundary(boundary+3)-self.mapper.audio_boundary(boundary))],
            }
            left_video, left_audio, _ = self._feed(reference, replacement, boundary, boundary + 3)
            right_video, right_audio, _ = self._feed(candidate, replacement, boundary, boundary + 3)
            reports.append({
                "retained": count, "snapshot_boundary": saved,
                "state_exact": states_match,
                "replacement_video_exact": torch.equal(left_video, right_video),
                "replacement_audio_exact": torch.equal(left_audio, right_audio),
                "finite": bool(torch.isfinite(right_video).all() and torch.isfinite(right_audio).all()),
            })
        unchanged = equal_state(original, source["state"])
        return {
            "gate": "native_decoder_snapshot_and_partial_replay",
            "source_state_unchanged": unchanged,
            "passed": unchanged and all(all(r[k] for k in (
                "state_exact", "replacement_video_exact", "replacement_audio_exact", "finite"
            )) for r in reports),
            "records": reports,
        }
