"""Shared optional aiortc media plane.

No JPEG round-trip: video consumes the original-resolution RGB preview cache.
RTP timestamp -> committed frame mapping is sent on the authenticated control
socket. Browser feedback must use requestVideoFrameCallback.rtpTimestamp, never
wall-clock extrapolation or the number of packets sent.
"""
import asyncio
from collections import deque
from fractions import Fraction
import json
import queue
import struct
import threading
import time

import numpy as np

FPS = 24

AIORTC_VERSION = "1.14.0"


def load_ice(path):
    """Only operator-supplied ICE servers; no implicit public STUN service."""
    config = json.loads(path.read_text()) if path else {"iceServers": []}
    if not isinstance(config, dict) or not isinstance(config.get("iceServers", []), list):
        raise ValueError("RTC ICE config must contain iceServers")
    for server in config.get("iceServers", []):
        if not isinstance(server, dict) or set(server) - {"urls", "username", "credential", "credentialType"}:
            raise ValueError("invalid RTC ICE server object")
        urls = server.get("urls", [])
        urls = [urls] if isinstance(urls, str) else urls
        if not isinstance(urls, list) or not urls or not all(
                isinstance(u, str) and u.startswith(("stun:", "turn:", "turns:")) for u in urls):
            raise ValueError("invalid operator ICE server URL")
    return config


class RTCMediaSource:
    """One clock, monotonic media PTS, held boundary video + silent underruns."""
    def __init__(self, demo, session):
        self.demo, self.session = demo, session
        self.cursor = 0
        self.pending = deque()
        self.slot = 0
        self.origin = None
        self.closed = False
        self.audio_queue = asyncio.Queue(maxsize=12)
        self.audio_dropped_frames = 0
        self.hold_frames = 0
        self.last_pull_monotonic = None
        height = getattr(demo.args, "rtc_height", demo.args.rtc_width * 4 // 7)
        self.last_rgb = np.zeros((height, demo.args.rtc_width, 3), np.uint8)
        self.last_metadata = {"frame_end": 0, "prompt_version": 0, "proposal": None, "hold": True}

    def take_frame(self):
        pull_started = time.monotonic()
        with self.demo.lock:
            s = self.demo.session
            if not s or s.id != self.session or s.stopped or s.error:
                raise EOFError("RTC session ended")
            if not self.pending:
                # Commit exactly one video frame at the backend's 24 FPS media
                # clock. Browser heartbeat is not a delivery gate.
                for unit in s.deliver(self.cursor, max_frames=1):
                    preview = self.demo.previews[unit.proposal]
                    if preview.raw_frames is None:
                        raise ValueError("RTC requires raw RGB media, not JPEG")
                    for offset in range(unit.end - unit.start):
                        source = unit.source_start + offset
                        first = source * preview.sample_rate // FPS
                        last = (source + 1) * preview.sample_rate // FPS
                        samples = np.frombuffer(preview.pcm[first * 4:last * 4], dtype="<i2").reshape(-1, 2)
                        if preview.sample_rate != 48000:
                            # Resample each source frame to exactly 2000 samples;
                            # 48kHz real H3 audio is passed through unchanged.
                            x = np.arange(len(samples))
                            target = np.arange(2000) * len(samples) / 2000
                            samples = np.stack([np.interp(target, x, samples[:, c]) for c in range(2)], axis=1)
                        pcm = np.ascontiguousarray(samples, dtype="<i2").tobytes()
                        self.pending.append((preview.raw_frames[source], pcm, {
                            "frame_end": unit.start + offset + 1,
                            "proposal": unit.proposal,
                            "prompt_version": unit.prompt_version, "hold": False,
                            "backend_commit_monotonic": time.monotonic(),
                            "rtc_pull_lock_wait_ms": (
                                time.monotonic() - pull_started
                            ) * 1000,
                        }))
                    self.cursor = unit.end
            if self.pending:
                self.last_pull_monotonic = time.monotonic()
                return self.pending.popleft()
            self.hold_frames += 1
            return self.last_rgb, bytes(2000 * 4), {**self.last_metadata, "hold": True}

    async def next_video(self):
        from av import VideoFrame
        from aiortc.mediastreams import MediaStreamError
        if self.closed:
            raise MediaStreamError
        if self.origin is None:
            self.origin = time.monotonic() + .10
        await asyncio.sleep(max(0, self.origin + self.slot / FPS - time.monotonic()))
        try:
            rgb, pcm, metadata = self.take_frame()
        except EOFError:
            raise MediaStreamError
        self.last_rgb, self.last_metadata = rgb, metadata
        audio = (self.slot * 2000, pcm)
        try:
            self.audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            # Audio startup/backpressure must never deadlock the video clock.
            # Keep the newest half-second and let Audio.recv jump its RTP PTS
            # to the retained source position when it catches up.
            self.audio_queue.get_nowait()
            self.audio_queue.put_nowait(audio)
            self.audio_dropped_frames += 1
        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = self.slot * 3750, Fraction(1, 90000)
        metadata["rtc_track_return_monotonic"] = time.monotonic()
        self.slot += 1
        return frame


class RTCManager:
    def __init__(self, demo):
        import aiortc
        if aiortc.__version__ != AIORTC_VERSION:
            raise ValueError(f"RTC timestamp adapter requires aiortc=={AIORTC_VERSION}")
        self.demo = demo
        self.config = load_ice(getattr(demo.args, "rtc_ice_config", None))
        browser_config = getattr(demo.args, "rtc_browser_ice_config", None)
        self.browser_config = load_ice(browser_config) if browser_config else self.config
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, daemon=True, name="speculative-sampling-rtc"
        )
        self.thread.start()
        self.peer = None
        self.source = None
        self.session = None
        self.state = "new"
        self.messages = queue.Queue(maxsize=256)

    def snapshot(self):
        source = self.source
        return None if source is None else {
            "video_slots": source.slot,
            "audio_queue_frames": source.audio_queue.qsize(),
            "audio_dropped_frames": source.audio_dropped_frames,
            "sender_pending_frames": len(source.pending),
            "hold_frames": source.hold_frames,
            "committed_frame": source.cursor,
            "last_pull_monotonic": source.last_pull_monotonic,
            "media_closed": source.closed,
        }

    def notify(self, value):
        # Never drop frame mappings silently: that would corrupt playhead credit.
        try:
            self.messages.put_nowait({"session": self.session, **value})
        except queue.Full:
            with self.demo.lock:
                if self.demo.session and self.demo.session.id == self.session:
                    self.demo.session.fail("RTC control mapping queue overflow")
            if self.source:
                self.source.closed = True

    async def _close(self):
        if self.source:
            self.source.closed = True
        if self.peer:
            await self.peer.close()
        self.peer, self.source, self.session = None, None, None
        self.state = "closed"

    def close_session(self):
        return asyncio.run_coroutine_threadsafe(self._close(), self.loop)

    def offer(self, session, sdp, kind):
        if kind != "offer" or not isinstance(sdp, str) or len(sdp) > 200_000:
            raise ValueError("expected a bounded RTC offer")
        if self.session == session and self.peer is not None:
            raise ValueError("RTC is already negotiated for this session; stop before reconnecting")
        future = asyncio.run_coroutine_threadsafe(self._offer(session, sdp), self.loop)
        try:
            return future.result(timeout=25)
        except Exception as exc:
            future.cancel()
            self.close_session()
            raise ValueError(f"RTC negotiation failed: {type(exc).__name__}") from exc

    async def _offer(self, session, sdp):
        from aiortc import (
            AudioStreamTrack, VideoStreamTrack, RTCConfiguration, RTCIceServer,
            RTCPeerConnection, RTCRtpSender, RTCSessionDescription,
        )
        from av import AudioFrame
        await self._close()
        self.session = session
        while not self.messages.empty():
            self.messages.get_nowait()
        servers = [RTCIceServer(**s) for s in self.config.get("iceServers", [])]
        pc = self.peer = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        source = self.source = RTCMediaSource(self.demo, session)

        class Video(VideoStreamTrack):
            async def recv(self):
                frame = await source.next_video()
                # Bound recovery time after loss, including a sharp proposal
                # splice. The built-in encoder otherwise has a very long GOP.
                if frame.pts % 90000 == 0:
                    sender._send_keyframe()
                return frame

        class Audio(AudioStreamTrack):
            def __init__(self):
                super().__init__()
                self.buffer = bytearray()
                self.samples = 0

            async def recv(self):
                from aiortc.mediastreams import MediaStreamError
                while len(self.buffer) < 960 * 4:
                    if source.closed:
                        raise MediaStreamError
                    source_start, chunk = await source.audio_queue.get()
                    if not self.buffer and source_start > self.samples:
                        self.samples = source_start
                    self.buffer.extend(chunk)
                await asyncio.sleep(max(0, source.origin + self.samples / 48000 - time.monotonic()))
                data = np.frombuffer(bytes(self.buffer[:960 * 4]), dtype="<i2").reshape(1, -1)
                del self.buffer[:960 * 4]
                frame = AudioFrame.from_ndarray(data, format="s16", layout="stereo")
                frame.sample_rate, frame.pts, frame.time_base = 48000, self.samples, Fraction(1, 48000)
                self.samples += 960
                return frame

        sender = pc.addTrack(Video())
        pc.addTrack(Audio())
        for transceiver in pc.getTransceivers():
            if transceiver.kind == "video":
                codecs = RTCRtpSender.getCapabilities("video").codecs
                # Keep RTX! Filtering to H.264 alone disables retransmission and
                # a lost fragment can freeze real (non-solid-color) video.
                transceiver.setCodecPreferences([c for c in codecs
                                                  if c.mimeType.lower() in ("video/h264", "video/rtx")])

        @pc.on("connectionstatechange")
        async def state_changed():
            if self.peer is not pc:
                return
            self.state = pc.connectionState
            self.notify({"type": "rtc_state", "state": self.state})
            if self.state in ("failed", "closed"):
                source.closed = True
                with self.demo.lock:
                    s = self.demo.session
                    if s and s.id == session and not s.stopped:
                        if self.state == "failed":
                            s.fail("RTC connection failed")
                        else:
                            # Browser closes its peer before the WS stop RPC
                            # arrives. This is a normal disconnect, not a model
                            # or timeline error.
                            s.last_seen = time.monotonic() - 11

        # Small, version-pinned adapter at the pre-SRTP boundary. It reads actual
        # wire timestamps, including aiortc's random RTP origin, and associates
        # them with the frame just encoded. No guess based on wall time / PTS.
        # BUNDLE may change a sender's transport during remote SDP processing.
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        transport = sender.transport
        send_original = transport._send_rtp
        seen, order = set(), deque()
        bitrate_initialized = False
        packet_clock = 0.0

        async def send_rtp(data):
            nonlocal bitrate_initialized, packet_clock
            if len(data) >= 12 and not 192 <= data[1] <= 223:
                timestamp, ssrc = struct.unpack_from("!II", data, 4)
                if ssrc == sender._ssrc and timestamp not in seen:
                    if not bitrate_initialized:
                        encoder = getattr(sender, "_RTCRtpSender__encoder", None)
                        if encoder is not None:
                            # Set an initial quality target once; retain subsequent
                            # receiver congestion feedback instead of overriding it.
                            encoder.target_bitrate = self.demo.args.rtc_video_bitrate
                            bitrate_initialized = True
                    seen.add(timestamp)
                    order.append(timestamp)
                    if len(order) > 512:
                        seen.remove(order.popleft())
                    self.notify({
                        "type": "rtc_frame", "rtp_timestamp": timestamp,
                        "rtp_send_monotonic": time.monotonic(),
                        **source.last_metadata,
                    })
                if ssrc == sender._ssrc:
                    # Optional extra smoothing for direct UDP paths.  TURN/TCP
                    # already applies transport backpressure; pacing it again
                    # here stalls VideoStreamTrack.recv and drops the effective
                    # source cadence well below 24 FPS.
                    pacing_bps = self.demo.args.rtc_pacing_bitrate
                    if pacing_bps:
                        now = time.monotonic()
                        packet_clock = max(now, packet_clock) + len(data) * 8 / pacing_bps
                        await asyncio.sleep(max(0, packet_clock - time.monotonic()))
            return await send_original(data)

        transport._send_rtp = send_rtp
        await pc.setLocalDescription(await pc.createAnswer())
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    def shutdown(self):
        try:
            self.close_session().result(timeout=8)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=3)
