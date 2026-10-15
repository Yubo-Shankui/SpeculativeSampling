"""Shared preview preparation: JPEG frames or RTC RGB plus aligned PCM16.

This is a preview transport, not a different model/decoder. Generation and VAE
decode remain at 1344x768. Browser canvas presentation is driven by Web Audio's
clock. Each delivered unit is a native latent interval, not a whole proposal.
"""

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .media_types import LiveUnit
from .timing import StageTimer

FPS = 24


@dataclass
class Preview:
    frames: list[str]
    pcm: bytes
    sample_rate: int
    channels: int
    width: int
    height: int
    anchor_frames: dict | None = None
    raw_frames: list | None = None

    @property
    def available(self):
        return len(self.raw_frames) if self.raw_frames is not None else len(self.frames)

    def packet(self, unit: LiveUnit):
        if self.raw_frames is not None:
            raise ValueError("RTC media cannot be delivered as JPEG packets")
        start = unit.source_start * self.sample_rate // FPS
        end = unit.source_end * self.sample_rate // FPS
        stride = 2 * self.channels
        return {
            **asdict(unit), "frames": self.frames[unit.source_start:unit.source_end],
            "audio": base64.b64encode(self.pcm[start * stride:end * stride]).decode(),
            "sample_rate": self.sample_rate, "channels": self.channels,
        }


@dataclass
class _StagedVideo:
    """Pinned host pixels guarded by the CUDA event that makes them readable."""

    ready: object
    pixels: object
    full: object | None
    gpu_keepalive: tuple


class PreviewBuilder:
    """Prepare each finalized VAE block immediately; no whole-proposal barrier."""
    def __init__(self, output, *, width=448, quality=65, anchors=False, transport="websocket",
                 proposal_frames=124):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.width, self.quality = width, quality
        self.anchors, self.transport = anchors, transport
        self.proposal_frames = proposal_frames
        self.preview = None
        self.pool = ThreadPoolExecutor(max_workers=8)
        self.video_pool = ThreadPoolExecutor(max_workers=1)
        self.video_futures = []
        self.copy_stream = None
        self.timer = StageTimer()

    def audio(self, waveform, sample_rate):
        with self.timer.span("audio_prepare"):
            return self._audio(waveform, sample_rate)

    def audio_chunk(self, start_sample, waveform, sample_rate):
        """Splice an append-only decoded PCM block into the proposal buffer."""
        with self.timer.span("audio_prepare"):
            p = self.preview
            if p is None:
                raise ValueError("initialize proposal audio before appending chunks")
            if int(sample_rate) != p.sample_rate:
                raise ValueError("streaming audio sample rate changed within a proposal")
            import numpy as np
            a = waveform.detach().float().cpu().numpy()
            if a.ndim != 2 or a.shape[0] not in (1, 2):
                raise ValueError("expected mono or stereo streaming audio")
            if a.shape[0] == 1:
                a = np.repeat(a, 2, axis=0)
            encoded = (np.clip(a.T, -1, 1) * 32767).astype("<i2").tobytes()
            byte_start = int(start_sample) * 2 * p.channels
            byte_end = min(len(p.pcm), byte_start + len(encoded))
            if byte_start < 0 or byte_start > len(p.pcm):
                raise ValueError("streaming audio chunk starts outside proposal")
            mutable = bytearray(p.pcm)
            mutable[byte_start:byte_end] = encoded[: byte_end - byte_start]
            p.pcm = bytes(mutable)

    def _audio(self, waveform, sample_rate):
        import numpy as np
        sr = int(sample_rate)
        audio = np.zeros((self.proposal_frames * sr // FPS, 2), dtype=np.float32)
        a = waveform.detach().float().cpu().numpy()
        if a.ndim != 2 or a.shape[0] not in (1, 2):
            raise ValueError("expected mono or stereo audio")
        n = min(len(audio), a.shape[1])
        audio[:n] = a[:, :n].T
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        self.preview = Preview([], pcm, sr, 2, self.width, 0,
                               {} if self.anchors else None,
                               [] if self.transport == "rtc" else None)

    def video(self, start, video):
        with self.timer.span("video_prepare"):
            return self._video(start, video)

    def _pixel_tensors(self, video):
        import torch
        import torch.nn.functional as F
        height = round(video.shape[-2] * self.width / video.shape[-1] / 2) * 2
        pixels = video if video.shape[-2:] == (height, self.width) else F.interpolate(
            video, (height, self.width), mode="bilinear", align_corners=False)
        pixels = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
        full = None
        if self.anchors and tuple(video.shape[-2:]) != (height, self.width):
            full = video.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
        return height, pixels, full

    def _install_pixels(self, start, height, pixels, full=None):
        p = self.preview
        if p is None or start != p.available:
            raise ValueError("audio must precede consecutive video chunks")
        p.height = height
        if p.raw_frames is not None:
            p.raw_frames.extend(pixels)
        else:
            def encode(frame):
                data = io.BytesIO()
                Image.fromarray(frame).save(data, "JPEG", quality=self.quality)
                return base64.b64encode(data.getvalue()).decode()
            p.frames.extend(self.pool.map(encode, pixels))
        if self.anchors:
            # RGB re-encode accepts any decoded frame. Retain frame-granular
            # anchors instead of only latent ends; the scheduler decides which
            # one survives as the next proposal's condition.
            full = pixels if full is None else full
            p.anchor_frames.update(zip(range(start, start + len(pixels)), full))
        return p

    def _video(self, start, video):
        height, pixels, full = self._pixel_tensors(video)
        pixels = pixels.cpu().numpy()
        full = None if full is None else full.cpu().numpy()
        return self._install_pixels(start, height, pixels, full)

    def _stage_video(self, video):
        """Enqueue GPU preparation and D2H without blocking the decode thread."""
        import torch

        if video.device.type != "cuda":
            return None
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=video.device)
        producer = torch.cuda.current_stream(video.device)
        produced = torch.cuda.Event()
        produced.record(producer)
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(produced)
            height, pixels, full = self._pixel_tensors(video)
            host_pixels = torch.empty_like(pixels, device="cpu", pin_memory=True)
            host_pixels.copy_(pixels, non_blocking=True)
            host_full = None
            if full is not None:
                host_full = torch.empty_like(full, device="cpu", pin_memory=True)
                host_full.copy_(full, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        return height, _StagedVideo(
            ready=ready,
            pixels=host_pixels,
            full=host_full,
            # Keep temporary CUDA allocations alive until the async copies finish.
            gpu_keepalive=(video, pixels, full),
        )

    def video_async(self, start, video, on_ready=None):
        """Prepare one post-first chunk in order while the decoder advances."""
        staged = self._stage_video(video)

        def consume():
            with self.timer.span("video_prepare"):
                if staged is None:
                    preview = self._video(start, video)
                else:
                    height, transfer = staged
                    transfer.ready.synchronize()
                    preview = self._install_pixels(
                        start,
                        height,
                        transfer.pixels.numpy(),
                        None if transfer.full is None else transfer.full.numpy(),
                    )
                if on_ready is not None:
                    on_ready(preview)
                return preview

        future = self.video_pool.submit(consume)
        self.video_futures.append(future)
        return future

    def drain_video(self):
        """Wait for ordered media preparation and surface any worker failure."""
        futures, self.video_futures = self.video_futures, []
        for future in futures:
            future.result()

    def finish(self):
        with self.timer.span("persist"):
            return self._finish()

    def _finish(self):
        p = self.preview
        if p is None or p.available != self.proposal_frames:
            raise ValueError("incomplete streaming proposal")
        self.pool.shutdown()
        record = {
            "transport": self.transport, "frames": p.frames,
            "sample_rate": p.sample_rate, "channels": p.channels,
            "width": p.width, "height": p.height, "source_frames": p.available,
            "raw_frames_persisted": False,
        }
        if self.transport == "websocket":
            # WebSocket previews are replayable artifacts. RTC video is not
            # persisted, so a megabyte-scale base64 audio tail cannot make its
            # record replayable and only delays the next production turn.
            record["audio"] = base64.b64encode(p.pcm).decode()
        else:
            record["audio_bytes"] = len(p.pcm)
        (self.output / "preview.json").write_text(json.dumps(record))
        return p

    def close(self):
        self.video_pool.shutdown(wait=True, cancel_futures=True)
        self.pool.shutdown(wait=True, cancel_futures=True)


def prepare_preview(media, output: Path, *, width=448, quality=65, anchors=False):
    import numpy as np
    import torch
    import torch.nn.functional as F

    output.mkdir(parents=True, exist_ok=True)
    video = media.video
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(f"expected decoded [T,C,H,W], got {tuple(video.shape)}")
    frame_count = int(video.shape[0])
    height = round(video.shape[-2] * width / video.shape[-1] / 2) * 2
    pixels = F.interpolate(video, (height, width), mode="bilinear", align_corners=False)
    pixels = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    def encode(frame):
        data = io.BytesIO()
        Image.fromarray(frame).save(data, "JPEG", quality=quality)
        return base64.b64encode(data.getvalue()).decode()

    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(encode, pixels))
    sr = int(media.audio_sample_rate or 48000)
    sample_count = frame_count * sr // FPS
    # Keep stereo and exact source sample boundaries, pad the short audio tail.
    audio = np.zeros((sample_count, 2), dtype=np.float32)
    if media.audio is not None:
        a = media.audio.detach().float().cpu().numpy()
        if a.ndim != 2 or a.shape[0] not in (1, 2):
            raise ValueError(f"unexpected audio shape {a.shape}")
        length = min(sample_count, a.shape[1])
        audio[:length] = a[:, :length].T
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    preview = Preview(frames, pcm, sr, 2, width, height)

    if anchors:
        full = video.clamp(0, 1).mul(255).round().to(torch.uint8)
        full = full.permute(0, 2, 3, 1).cpu().numpy()
        # RGB re-encode uses frame-level boundaries. Keep the decoded frames in
        # bounded host memory and encode only the selected anchor on demand.
        preview.anchor_frames = dict(enumerate(full))

    # An independently replayable record; no public asset or signed URL required.
    (output / "preview.json").write_text(json.dumps({
        "frames": frames, "audio": base64.b64encode(pcm).decode(),
        "sample_rate": sr, "channels": 2, "width": width, "height": height,
    }), encoding="utf-8")
    return preview


def synthetic_preview(proposal: int, width=672, *, transport="websocket"):
    """Explicit --mock mode for transport tests; never the real GPU mode."""
    import math
    import struct

    frames, raw = [], [] if transport == "rtc" else None
    for i in range(124):
        image = Image.new("RGB", (width, width * 4 // 7),
                          ((proposal * 67) % 220, (i * 3) % 220, 70))
        if raw is not None:
            import numpy as np
            raw.append(np.array(image))
        else:
            data = io.BytesIO()
            image.save(data, "JPEG", quality=65)
            frames.append(base64.b64encode(data.getvalue()).decode())
    sr = 24000
    pcm = b"".join(struct.pack("<hh", *(int(900 * math.sin(2 * math.pi * 220 * n / sr)),) * 2)
                   for n in range(124 * sr // FPS))
    return Preview(frames, pcm, sr, 2, width, width * 4 // 7, raw_frames=raw)
