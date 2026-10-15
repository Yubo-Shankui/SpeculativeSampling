"""Lossless PCM-to-RGB interval join across valid native decode windows.

Audio latent boundaries are not RGB frame boundaries. Keep actual PCM samples
at native absolute sample positions; never fill a missing interval with silence.
The controller receives the provenance of every window used in an AV group.
"""
from dataclasses import dataclass

from speculative_sampling import SourceDependency


@dataclass(frozen=True)
class AudioWindow:
    job_id: int
    source_start: int
    source_end: int
    sample_start: int
    pcm: object

    @property
    def sample_end(self):
        return self.sample_start + self.pcm.shape[-1]


class AudioTimeline:
    def __init__(self, sample_rate):
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("invalid PCM sample rate")
        self.sample_rate = sample_rate
        self.windows = []

    @staticmethod
    def valid(window, controller):
        proposal = controller.proposals.get(window.job_id)
        return (proposal is not None and not proposal.failed and proposal.generated
                and proposal.command.retain_until <= window.source_start
                and proposal.valid_end is not None
                and window.source_end <= proposal.valid_end)

    def append(self, window, controller):
        if window.pcm.ndim != 2 or window.pcm.shape[-1] <= 0 or window.sample_start < 0:
            raise ValueError("PCM window must be nonempty [channels,samples]")
        if not self.valid(window, controller):
            return False
        self.windows = [w for w in self.windows if self.valid(w, controller)]
        # Identical callbacks do not overwrite the original accepted payload.
        if any((w.job_id, w.source_start, w.source_end) ==
               (window.job_id, window.source_start, window.source_end) for w in self.windows):
            return False
        self.windows.append(window)
        self.windows.sort(key=lambda w: w.sample_start)
        return True

    def read(self, start, end, controller):
        """Return exact PCM and source dependencies, or None if any hole exists."""
        import torch
        if not 0 <= start < end:
            raise ValueError("invalid requested PCM interval")
        cursor, pieces, dependencies = start, [], []
        for window in self.windows:
            if not self.valid(window, controller) or window.sample_end <= cursor:
                continue
            if window.sample_start > cursor:
                return None
            stop = min(end, window.sample_end)
            pieces.append(window.pcm[:, cursor-window.sample_start:stop-window.sample_start])
            dependencies.append(SourceDependency(
                window.job_id, window.source_start, window.source_end))
            cursor = stop
            if cursor == end:
                return torch.cat(pieces, dim=-1), tuple(dict.fromkeys(dependencies))
        return None

    def prune(self, played_sample, controller):
        self.windows = [w for w in self.windows
                        if w.sample_end > played_sample and self.valid(w, controller)]
