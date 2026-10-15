"""Opt-in CUDA event timings; no extra barrier on the latent handoff path."""
from contextlib import contextmanager


class CudaStageTimer:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA event profiling requires a CUDA device")
        self.events = {}

    @contextmanager
    def stage(self, name):
        import torch
        if name in self.events:
            raise ValueError(f"duplicate GPU timing stage: {name}")
        with torch.cuda.device(self.device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
        self.events[name] = (start, end)

    def resolve(self):
        # Resolve only after the caller has handed latents off and enqueued
        # speculative prefill. Do not synchronize between denoising and handoff.
        values = {}
        for name, (start, end) in self.events.items():
            end.synchronize()
            values[name] = start.elapsed_time(end) / 1000.0
        return values
