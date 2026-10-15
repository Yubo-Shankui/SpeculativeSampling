"""Generator-owned Wan CF++ KV state behind the SpeculativeSampling controller."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch


@dataclass(frozen=True)
class WanNativeProposal:
    values: object
    committed: object | None
    accepted_latents: int
    previous_pending_latents: int
    prompt_encode_seconds: float
    commit_seconds: float
    denoise_seconds: float


class WanNativeSpeculativeSamplingSession:
    """Keep official clean KV separate from replaceable future predictions.

    Denoising always samples one complete native K-frame proposal on a fork of
    official KV.  On the next controller command, only the accepted prefix is
    clean-prefilled into official KV; the remaining prediction suffix and its
    speculative cache writes are discarded.
    """

    def __init__(self, pipeline, chunk_size):
        self.pipeline = pipeline
        self.torch = __import__("torch")
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        device = next(pipeline.generator.parameters()).device
        pipeline._initialize_kv_cache(1, self.torch.bfloat16, device)
        self.official_cache = pipeline.kv_cache1
        self.official_until = 0
        self.pending = None
        self.pending_condition = None
        self.closed = False

    def _cross_cache(self):
        device = next(self.pipeline.generator.parameters()).device
        self.pipeline._initialize_crossattn_cache(1, self.torch.bfloat16, device)
        return self.pipeline.crossattn_cache

    @staticmethod
    def _clone_cache(cache):
        return [
            {
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in layer.items()
            }
            for layer in cache
        ]

    def _forward_clean(self, values, condition, start):
        torch = self.torch
        timestep = torch.ones(
            (1, values.shape[1]), device=values.device, dtype=torch.int64
        ) * self.pipeline.args.context_noise
        self.pipeline.generator(
            noisy_image_or_video=values,
            conditional_dict=condition,
            timestep=timestep,
            kv_cache=self.official_cache,
            crossattn_cache=self._cross_cache(),
            current_start=int(start) * self.pipeline.frame_seq_length,
        )

    def _sample(self, condition, start, seed):
        torch = self.torch
        device = next(self.pipeline.generator.parameters()).device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noisy = torch.randn(
            (1, self.chunk_size, 16, 60, 104),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        cache = self._clone_cache(self.official_cache)
        cross_cache = self._cross_cache()
        schedule = (
            self.pipeline.denoising_step_list_first_chunk
            if start == 0 and self.pipeline.denoising_step_list_first_chunk is not None
            else self.pipeline.denoising_step_list
        )
        denoised = None
        for index, timestep_value in enumerate(schedule):
            timestep = torch.ones(
                (1, self.chunk_size), device=device, dtype=torch.int64
            ) * timestep_value
            _, denoised = self.pipeline.generator(
                noisy_image_or_video=noisy,
                conditional_dict=condition,
                timestep=timestep,
                kv_cache=cache,
                crossattn_cache=cross_cache,
                current_start=int(start) * self.pipeline.frame_seq_length,
            )
            if index + 1 < len(schedule):
                next_timestep = schedule[index + 1]
                noise = torch.randn(
                    denoised.flatten(0, 1).shape,
                    device=device,
                    dtype=denoised.dtype,
                    generator=generator,
                )
                noisy = self.pipeline.scheduler.add_noise(
                    denoised.flatten(0, 1),
                    noise,
                    next_timestep
                    * torch.ones(
                        self.chunk_size, device=device, dtype=torch.long
                    ),
                ).unflatten(0, denoised.shape[:2])
        if denoised is None:
            raise RuntimeError("empty Wan denoising schedule")
        return denoised.detach()

    @torch.inference_mode()
    def advance(self, *, retain_until, condition=None, prompt=None, seed):
        """Advance official CF++ KV to ``retain_until`` and sample the next K.

        The encoded condition is normally prepared before controller dispatch.
        ``prompt`` remains a compatibility fallback for offline callers.
        """
        if self.closed:
            raise RuntimeError("native Wan session is closed")
        retain_until = int(retain_until)
        if not self.official_until <= retain_until:
            raise ValueError("retain_until precedes official KV")
        pending_length = 0 if self.pending is None else int(self.pending.shape[1])
        if retain_until > self.official_until + pending_length:
            raise ValueError("retain_until exceeds prediction-cache horizon")

        accepted = retain_until - self.official_until
        committed = None
        commit_started = time.perf_counter()
        if accepted:
            committed = self.pending[:, :accepted].detach()
            self._forward_clean(committed, self.pending_condition, self.official_until)
        self.official_until = retain_until
        self.pending = None
        self.pending_condition = None
        self.torch.cuda.synchronize()
        commit_seconds = time.perf_counter() - commit_started

        prompt_encode_seconds = 0.0
        if condition is None:
            if prompt is None:
                raise ValueError("encoded condition or prompt is required")
            encode_started = time.perf_counter()
            condition = self.pipeline.text_encoder(text_prompts=[prompt])
            self.torch.cuda.synchronize()
            prompt_encode_seconds = time.perf_counter() - encode_started

        denoise_started = time.perf_counter()
        proposal = self._sample(condition, retain_until, seed)
        self.torch.cuda.synchronize()
        denoise_seconds = time.perf_counter() - denoise_started
        self.pending = proposal
        self.pending_condition = condition
        return WanNativeProposal(
            values=proposal,
            committed=committed,
            accepted_latents=accepted,
            previous_pending_latents=pending_length,
            prompt_encode_seconds=prompt_encode_seconds,
            commit_seconds=commit_seconds,
            denoise_seconds=denoise_seconds,
        )

    def execute(self, **kwargs):
        """Compatibility alias for pre-v1 callers; new runtimes use advance."""
        return self.advance(**kwargs)

    def close(self):
        self.closed = True
        self.pending = None
        self.pending_condition = None
        self.official_cache = None
