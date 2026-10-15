"""Shared request-scoped timing without per-stage CUDA synchronize.

CUDA events measure rank-zero current-stream elapsed time (including ordered
waits), NOT the sum of GPU kernel execution time. CPU wall spans may overlap
GPU work and must never be added to GPU spans to manufacture E2E latency.
"""
from contextlib import contextmanager
import time


class StageTimer:
    def __init__(self, device=None):
        self.device = device
        self.samples = []

    @contextmanager
    def span(self, name):
        events = None
        if self.device is not None and self.device.type == "cuda":
            import torch
            with torch.cuda.device(self.device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if events:
                events[1].record()
            self.samples.append((name, elapsed, events))

    def report(self):
        """Call after the engine's existing end-of-request CUDA synchronization."""
        result = {}
        for name, elapsed, events in self.samples:
            row = result.setdefault(name, {"calls": 0, "wall_s": 0.0, "gpu_s": None})
            row["calls"] += 1
            row["wall_s"] += elapsed
            if events:
                row["gpu_s"] = (row["gpu_s"] or 0.0) + events[0].elapsed_time(events[1]) / 1000
        return result
