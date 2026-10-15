"""Request-scoped adapter for the StreamRender H3 TAE/causal decoder.

The stock H3 VAE remains installed on the engine.  Selecting this backend
temporarily replaces only the keyframe encode and final video decode pipeline
steps, so switching back to ``original`` has no side effects.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any


BACKENDS = {"original", "sol-h3-vae"}


def h3_frames_from_latents(latent_frames: int) -> int:
    """Return the native H3 RGB length for a valid latent sequence."""
    blocks, tail = divmod(latent_frames, 5)
    if latent_frames < 2 or tail not in {0, 2}:
        raise ValueError("H3 latent count must match 5*n or 5*n+2")
    return blocks * 17 + (5 if tail else 0)


def _asset_path(value: str, label: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "${" in expanded:
        raise ValueError(f"{label} contains an unresolved environment variable: {value}")
    path = Path(expanded)
    if path.is_dir():
        preferred = [
            candidate for candidate in (
                path / "model.safetensors",
                path / "diffusion_pytorch_model.safetensors",
            ) if candidate.is_file()
        ]
        candidates = sorted({
            *path.glob("*.safetensors"), *path.glob("*.pt"), *path.glob("*.pth")
        })
        if preferred:
            path = preferred[0]
        elif len(candidates) == 1:
            path = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"{label} directory is ambiguous; provide the checkpoint file: {path}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return state


def _install_vendor_path() -> None:
    vendor = str(Path(__file__).with_name("vendor"))
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def _callback_error(callback, *args) -> str | None:
    """Run a local media callback and return its failure without aborting decode."""
    if callback is None:
        return None
    try:
        callback(*args)
    except Exception as exc:  # media failures restart the request after ranks reconverge
        return f"{type(exc).__name__}: {exc}"
    return None


def _sync_request_error(error: str | None, *, device=None) -> str | None:
    """Propagate one rank-zero request result after the complete proposal decode."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        errors = [error]
        dist.broadcast_object_list(errors, src=0, device=device)
        error = errors[0]
    return error


class _SolH3Codec:
    def __init__(self, engine: Any, config: dict[str, Any]) -> None:
        import torch

        _install_vendor_path()
        from dev.yanzuolu.projects.minimax_h3.modeling.taeh3_native_streaming import (
            MiniMaxH3NativeStreamingTAE,
        )

        self.engine = engine
        self.rank = int(engine.rank)
        self.device = engine.device
        tae_path = _asset_path(config["tae_weights"], "TAE weights")
        self.tae = MiniMaxH3NativeStreamingTAE(
            parallel_encode=False, parallel_decode=False
        )
        self.tae.load_state_dict(_load_state_dict(tae_path), strict=True)
        self.tae.eval().requires_grad_(False).to(self.device)

        self.decoder = None
        if self.rank == 0:
            from .decoder.causal_vit_codec import CausalViTVideoDecoder

            decoder_path = _asset_path(
                config["causal_decoder_weights"], "causal decoder weights"
            )
            decoder_config = json.loads(
                Path(__file__).with_name("decoder_config.json").read_text(encoding="utf-8")
            )
            self.decoder = CausalViTVideoDecoder(
                decoder_config,
                str(decoder_path),
                chunk=2,
                cache_latents=4,
                cuda_graph=bool(config.get("cuda_graph", True)),
                fa4_root=config.get("fa4_root"),
            )
            # The stream normalizes latents before its first lazy fast-path
            # lookup, so move the parent codec buffers as well as the ViT.
            self.decoder.eval().requires_grad_(False).to(self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _encode_keyframes(self, components: Any, images: list[Any], device=None):
        import torch
        from diffusers.modular_pipelines.minimax_h3.packing import patchify_video_latents

        del device
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            latents = self.tae.encode_images(
                images, transform_input=False, use_fp16_latent=True
            )
        rows = [
            patchify_video_latents(latent.float().unsqueeze(0).cpu(), components.patch_size)
            for latent in latents
        ]
        return torch.cat(rows)

    def _decode_rank_zero(self, components: Any, block_state: Any, on_video: Any):
        import torch
        from diffusers.modular_pipelines.minimax_h3.packing import unpatchify_video_tokens

        latents = unpatchify_video_tokens(
            block_state.latents[block_state.num_condition_video_rows :],
            block_state.num_latent_frames,
            block_state.latent_height,
            block_state.latent_width,
            components.vae_latent_channels,
            components.patch_size,
        )
        mean = torch.as_tensor(
            components.vae.config.latents_mean, device=latents.device
        ).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(
            components.vae.config.latents_std, device=latents.device
        ).view(1, -1, 1, 1, 1)
        raw = (latents * std + mean).permute(0, 2, 1, 3, 4)
        stream = self.decoder.create_stream()
        chunks = []
        cursor = 0
        for index in range(0, raw.shape[1] - 1, 2):
            rgb = stream.decode(raw[:, index : index + 2])
            if rgb is None:
                raise RuntimeError("causal H3 decoder did not emit a two-latent chunk")
            chunks.append(rgb)
            on_video(cursor, rgb[0])
            cursor += int(rgb.shape[1])
        if raw.shape[1] % 2:
            stream.decode(raw[:, -1:])
            rgb = stream.flush_decoder()
            if rgb is None:
                raise RuntimeError("causal H3 decoder did not flush its final latent")
            chunks.append(rgb)
            on_video(cursor, rgb[0])
            cursor += int(rgb.shape[1])
        expected = h3_frames_from_latents(int(raw.shape[1]))
        if cursor != expected:
            raise RuntimeError(f"causal H3 decoder emitted {cursor} frames, expected {expected}")
        video = torch.cat(chunks, dim=1).permute(0, 2, 1, 3, 4).float()
        return components.video_processor.postprocess_video(
            video, output_type=block_state.output_type
        )

    @contextmanager
    def activate(self, *, on_audio=None, on_video=None):
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AudioDecodeStep,
            MiniMaxH3VideoDecodeStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3KeyframeVaeEncoderStep,
        )

        audio_original = MiniMaxH3AudioDecodeStep.__call__
        video_original = MiniMaxH3VideoDecodeStep.__call__
        encode_original = MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes
        decoded_audio_states: set[int] = set()
        codec = self

        def audio(step, components, state):
            if id(state) not in decoded_audio_states:
                components, state = audio_original(step, components, state)
                decoded_audio_states.add(id(state))
            return components, state

        def video(step, components, state):
            cancel_target = getattr(codec.engine, "transformer", None)
            if cancel_target is not None:
                cancel = getattr(cancel_target, "_h3opt_cancel_check", None)
                if cancel is not None:
                    from h3_runtime.cancellation import checkpoint

                    checkpoint(cancel_target, -1)
            components, state = audio(MiniMaxH3AudioDecodeStep(), components, state)
            waveform, sample_rate = state.get("audio"), state.get("sampling_rate")
            block_state = step.get_block_state(state)
            request_error = None
            if codec.rank == 0:
                request_error = _callback_error(
                    on_audio,
                    None if waveform is None else waveform[0],
                    sample_rate,
                )

                def publish_video(start, frames):
                    nonlocal request_error
                    if request_error is None:
                        request_error = _callback_error(on_video, start, frames)

                try:
                    block_state.videos = codec._decode_rank_zero(
                        components, block_state, publish_video
                    )
                except Exception as exc:
                    if request_error is None:
                        request_error = f"{type(exc).__name__}: {exc}"
            else:
                block_state.videos = None
            request_error = _sync_request_error(request_error, device=codec.device)
            if request_error:
                raise RuntimeError("streaming media request failed: " + request_error)
            step.set_block_state(state, block_state)
            return components, state

        MiniMaxH3AudioDecodeStep.__call__ = audio
        MiniMaxH3VideoDecodeStep.__call__ = video
        MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes = staticmethod(
            self._encode_keyframes
        )
        try:
            yield
        finally:
            MiniMaxH3AudioDecodeStep.__call__ = audio_original
            MiniMaxH3VideoDecodeStep.__call__ = video_original
            MiniMaxH3KeyframeVaeEncoderStep.encode_keyframes = staticmethod(
                encode_original
            )


class SolH3VAEEngine:
    """Engine proxy selecting the original or Sol codec per request."""

    def __init__(
        self,
        engine: Any,
        config: dict[str, Any],
        *,
        default_backend: str = "sol-h3-vae",
        available_backends: tuple[str, ...] = ("original", "sol-h3-vae"),
    ) -> None:
        self._engine = engine
        self._codec = _SolH3Codec(engine, config)
        self.default_vae_backend = default_backend
        self.available_vae_backends = available_backends
        self.vae_backend = default_backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def generate(self, *args, **kwargs):
        backend = kwargs.pop("vae_backend", self.default_vae_backend)
        if backend not in self.available_vae_backends:
            raise ValueError(f"H3 VAE backend is not loaded: {backend}")
        if backend == "original":
            return self._engine.generate(*args, **kwargs)
        on_audio = kwargs.get("on_audio")
        on_video = kwargs.get("on_video")
        kwargs["stream_vae"] = False
        with self._codec.activate(on_audio=on_audio, on_video=on_video):
            return self._engine.generate(*args, **kwargs)


class _OriginalH3VAEEngine:
    vae_backend = "original"
    default_vae_backend = "original"
    available_vae_backends = ("original",)

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def generate(self, *args, **kwargs):
        backend = kwargs.pop("vae_backend", "original")
        if backend != "original":
            raise ValueError(f"H3 VAE backend is not loaded: {backend}")
        return self._engine.generate(*args, **kwargs)


def configure_vae_backend(engine: Any, decoder: dict[str, Any]) -> Any:
    """Return the original engine or a Sol-H3 codec proxy from deployment config."""
    backend = decoder.get("backend", "original")
    if backend not in BACKENDS:
        raise ValueError(f"unsupported H3 VAE backend: {backend}")
    available = tuple(decoder.get("available_backends", [backend]))
    if not available or len(set(available)) != len(available):
        raise ValueError("decoder.available_backends must contain unique backends")
    unknown = set(available) - BACKENDS
    if unknown:
        raise ValueError(f"unsupported H3 VAE backends: {sorted(unknown)}")
    if backend not in available:
        raise ValueError("default decoder backend must be available")
    if "sol-h3-vae" not in available:
        return _OriginalH3VAEEngine(engine)
    return SolH3VAEEngine(
        engine,
        decoder["sol_h3_vae"],
        default_backend=backend,
        available_backends=available,
    )


__all__ = ["BACKENDS", "SolH3VAEEngine", "configure_vae_backend", "h3_frames_from_latents"]
