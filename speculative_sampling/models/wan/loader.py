"""Load the exact official CF++ pipeline selected by a deployment config."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


class ResidentPipeline:
    """Small ownership handle used by the live worker."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def close(self):
        self.pipeline = None


def load_resident_pipeline(*, repo, config, checkpoint, device="cuda"):
    repo = Path(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from omegaconf import OmegaConf
    import torch

    cfg = OmegaConf.merge(
        OmegaConf.load(repo / "configs" / "default_config.yaml"),
        OmegaConf.load(config),
    )
    factory = getattr(importlib.import_module("pipeline"), "CausalInferencePipeline")
    pipeline = factory(cfg, device=torch.device(device))
    state = torch.load(checkpoint, map_location="cpu")
    weights = state.get("generator_ema", state.get("generator"))
    if weights is None:
        raise ValueError("CF++ checkpoint has neither generator_ema nor generator weights")
    pipeline.generator.load_state_dict(weights, strict=False)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)
    pipeline.text_encoder.to(device)
    pipeline.vae.to(device)
    return ResidentPipeline(pipeline)


__all__ = ["ResidentPipeline", "load_resident_pipeline"]
