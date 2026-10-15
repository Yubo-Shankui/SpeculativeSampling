"""Explicit ref2dmd AV timeline, including the one-time reference frame.

Native first K45 produces 45 generated latents *plus* the video reference.
Decoding that reference once gives 361 first frames; every later K45 adds 360.
Audio boundaries are the native 25-Hz causal latent mapping, not a fixed token
count per group. PCM delivery uses global frame boundaries to avoid drift.
"""
from dataclasses import dataclass
import math

from speculative_sampling import OutputGroup


@dataclass(frozen=True)
class LtxOutputMapper:
    fps: int = 24
    group_latents: int = 3
    version: str = "ref2av-reference-once-v2"

    def rgb_boundary(self, position):
        if position < 0:
            raise ValueError("negative timeline position")
        return 0 if position == 0 else 1 + 8 * position

    def latent_boundary(self, frames):
        if frames < 0:
            raise ValueError("negative frame position")
        return 0 if frames < 9 else (frames - 1) // 8

    def audio_boundary(self, position):
        if position < 0:
            raise ValueError("negative timeline position")
        return 0 if position == 0 else 1 + int(25 * ((8 * position + 1) / self.fps) + .25)

    def seconds(self, position):
        return self.rgb_boundary(position) / self.fps

    def pcm_boundary(self, position, sample_rate):
        """Native causal audio-VAE/vocoder output boundary before RGB crop."""
        if position == 0:
            return 0
        if sample_rate % 100:
            raise ValueError("native audio mapping requires an integral samples-per-mel")
        return (4 * self.audio_boundary(position) - 3) * (sample_rate // 100)

    def distance_for_seconds(self, seconds):
        return math.ceil(seconds * self.fps / 8)

    def mapping(self, start, count):
        result = []
        end = start + count
        for index in range(start, end):
            dependency = min(end, start + ((index - start) // self.group_latents + 1)
                             * self.group_latents)
            result.append(OutputGroup(
                index, index + 1, index, index + 1,
                self.seconds(index), self.seconds(index + 1),
                requires_audio=True, dependency_end=dependency,
            ))
        return tuple(result)
