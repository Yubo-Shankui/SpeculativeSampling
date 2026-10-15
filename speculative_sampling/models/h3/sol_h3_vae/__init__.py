"""Optional StreamRender H3 TAE encoder and causal ViT decoder backend."""

from .backend import configure_vae_backend, h3_frames_from_latents

__all__ = ["configure_vae_backend", "h3_frames_from_latents"]
