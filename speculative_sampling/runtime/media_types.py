"""Transport-owned media spans, independent of every model scheduler."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LiveUnit:
    start: int
    end: int
    proposal: int
    latent_index: int
    source_start: int
    source_end: int
    prompt_version: int
