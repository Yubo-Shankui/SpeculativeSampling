"""Wan VAE and chunk geometry mapped to shared LiveSession semantics."""

from __future__ import annotations

import math

from speculative_sampling.runtime.retained_kv_session import RetainedKVLiveSession
from .deployment import CAPABILITIES, LIVE_POLICY


def wan_proposal_geometry(duration_s: int, chunk_latents: int):
    if duration_s not in (5, 10) or chunk_latents not in (3, 21):
        raise ValueError("Wan supports 5/10 seconds and K=3/21")
    # Duration is an H3 UI/arena refill horizon, never the amount sampled by
    # one generator instruction. Every instruction jointly samples exactly K.
    latent_frames = chunk_latents
    unit_spans = (1,) + (4,) * (latent_frames - 1)
    frames = sum(unit_spans)
    return frames, unit_spans, (frames,), latent_frames


class WanLiveSession(RetainedKVLiveSession):
    chunk_latents = 3
    condition_handles_required = True
    mapper_factory = staticmethod(lambda: WanOutputMapper())
    live_policy = LIVE_POLICY
    capabilities = CAPABILITIES

    @staticmethod
    def rgb_boundary(latent_groups):
        return 0 if latent_groups <= 0 else 1 + 4 * (latent_groups - 1)

    @staticmethod
    def latent_boundary(rgb_frames):
        if rgb_frames < 0:
            raise ValueError("RGB boundary must be non-negative")
        return 0 if rgb_frames == 0 else 1 + (rgb_frames - 1) // 4


class WanOutputMapper:
    fps = 24
    group_latents = 1

    @staticmethod
    def rgb_boundary(position):
        return WanLiveSession.rgb_boundary(position)

    @staticmethod
    def latent_boundary(frames):
        return WanLiveSession.latent_boundary(frames)

    def seconds(self, position):
        return self.rgb_boundary(position) / self.fps

    def distance_for_seconds(self, seconds):
        return math.ceil(float(seconds) * self.fps / 4)
