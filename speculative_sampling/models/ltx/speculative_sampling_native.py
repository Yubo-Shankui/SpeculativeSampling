"""Persistent native-block vForge ReRoPE session for SpeculativeSampling.

The controller owns prediction delivery and supplies a logical safe point.
The generator keeps the checkpoint's native K layout, fixed sink, and rolling
ReRoPE blocks.  No
controller boundary is represented by changing the model's block layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
import time

import torch


@dataclass(frozen=True)
class NativePrediction:
    index: int
    selection: tuple[tuple[int, int], tuple[int, int]]
    values: dict[str, torch.Tensor]
    condition_version: int
    condition_batch: object


@dataclass(frozen=True)
class NativeProposal:
    predictions: tuple[NativePrediction, ...]
    committed: tuple[NativePrediction, ...]
    denoise_seconds: float
    commit_seconds: float
    clean_prefill_seconds: float = 0.0
    cuda_stage_seconds: dict[str, float] | None = None


def latent_ranges(batch):
    """Return one native video/audio selection for every generated q group."""
    inputs = batch.model_inputs
    layout = inputs.layout
    if layout.prefix_frames != 1:
        raise ValueError("LTX SpeculativeSampling requires one ref2av reference latent")
    ranges = []
    audio_start = 0
    for index in range(layout.frames - layout.prefix_frames):
        video_start = (layout.prefix_frames + index) * layout.frame_tokens
        video_end = video_start + layout.frame_tokens
        video_frame_end = layout.prefix_frames + index + 1
        end_seconds = ((video_frame_end - 1) * 8 + 1) / inputs.fps
        audio_end = 1 + int(25 * end_seconds + 0.25)
        ranges.append(((video_start, video_end), (audio_start, audio_end)))
        audio_start = audio_end
    return tuple(ranges)


def proposal_selection(ranges, start, count):
    selected = ranges[start : start + count]
    if len(selected) != count:
        raise ValueError("native position capacity exhausted")
    return (
        (selected[0][0][0], selected[-1][0][1]),
        (selected[0][1][0], selected[-1][1][1]),
    )


class NativeSpeculativeSamplingSession:
    """One browser session with resident model, official KV and prediction tail."""

    def __init__(
        self, model, batch, *, seed, steps, chunk_size=45,
        kv_cache_blocks=2, num_sink_blocks=1,
    ):
        from vforge.algorithms.dmd.runtime import DmdRolloutSchedule
        from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise
        from vforge.tasks.ltx.dmd.generator import LtxDmdGeneratorAdapter

        self.model = model
        self.batch = batch
        self.chunk_size = int(chunk_size)
        self.ranges = latent_ranges(batch)
        layout = batch.model_inputs.layout
        native_block_sizes = tuple(
            (end - start) // layout.frame_tokens for start, end in layout.video_ranges
        )
        if not native_block_sizes or any(size != self.chunk_size for size in native_block_sizes):
            raise ValueError(
                f"LTX SpeculativeSampling requires an unmodified K{self.chunk_size} layout; "
                f"got native blocks {native_block_sizes[:4]}"
            )
        self.adapter = LtxDmdGeneratorAdapter(
            kv_cache_blocks=kv_cache_blocks,
            num_sink_blocks=num_sink_blocks,
            rerope=True,
            # Preserve the original K3 policy across physical checkpoint
            # chunk sizes: one sink unit is 3 latents and two local units are
            # 6 latents, rather than 1/2 whole K15 or K45 proposals.
            rerope_memory_latents=3 * kv_cache_blocks,
            rerope_sink_latents=3 * num_sink_blocks,
        )
        self.schedule = DmdRolloutSchedule(tuple(float(step) / 1000.0 for step in steps))
        self.random = torch.Generator(device=batch.clean["video"].device).manual_seed(int(seed))
        self.context_level = resolve_rollout_noise(
            None, torch.zeros((), device=batch.clean["video"].device), batch
        )
        self.context_manager = self.adapter.open_rollout_context(model, batch)
        self.context = self.context_manager.__enter__()
        self.official_until = 0
        self.pending = {}
        self.closed = False
        self.pending_full_context = None

    def _sample(self, condition_batch, selection):
        from functools import partial
        from vforge.algorithms.dmd.runtime import sample_one_chunk
        from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise

        reference = condition_batch.clean["video"]

        def noise(value):
            result = torch.randn(
                value.shape, device=value.device, dtype=value.dtype, generator=self.random
            )
            observer = getattr(self, "noise_observer", None)
            if observer is not None:
                observer(result)
            return result

        video, audio = selection
        noisy = {
            "video": noise(reference[:, : video[1] - video[0]]),
            "audio": noise(condition_batch.clean["audio"][:, : audio[1] - audio[0]]),
        }
        accepted, _, _ = sample_one_chunk(
            self.model, self.adapter, condition_batch, selection, noisy, self.context,
            resolve_noise_level=partial(resolve_rollout_noise, None),
            schedule=self.schedule, phase="critic", noise_fn=noise,
            noise_dtype=torch.float32,
        )
        return {key: value.detach() for key, value in accepted.items()}

    def advance(self, *, retain_until, condition_batch, condition_version,
                on_latents=None, prepare_full=False):
        """Advance the resident iterator from the accepted logical index.

        Official ReRoPE KV, speculative proposal state, RNG, and absolute
        positions remain owned by this object across calls. The controller only
        supplies the next accepted-prefix index and an encoded condition.
        """
        if self.closed:
            raise RuntimeError("native SpeculativeSampling session is closed")
        retain_until = int(retain_until)
        if not self.official_until <= retain_until:
            raise ValueError(
                f"retain_until {retain_until} precedes official KV {self.official_until}"
            )
        if retain_until > self.official_until + len(self.pending):
            raise ValueError(
                f"retain_until {retain_until} exceeds prediction-cache horizon"
            )
        # Validate new positional metadata before committing or discarding the
        # only recoverable prediction. The worker may extend absolute metadata;
        # this never changes the native KV window or physical chunk size.
        next_ranges = latent_ranges(condition_batch)
        selection = proposal_selection(next_ranges, retain_until, self.chunk_size)
        timer = None
        if getattr(self, "profile_cuda_stages", False):
            from speculative_sampling.runtime.gpu_timing import CudaStageTimer
            timer = CudaStageTimer(condition_batch.clean["video"].device)
        commit_started = time.perf_counter()
        committed = []
        for index in range(self.official_until, retain_until):
            prediction = self.pending.get(index)
            if prediction is None:
                raise ValueError(f"prediction cache has no latent {index} to retain")
            committed.append(prediction)
        if (committed and len(committed) == self.chunk_size
                and getattr(self, "pending_full_context", None) is not None):
            self.context = self.pending_full_context
        elif committed:
            retained_condition_batch = committed[0].condition_batch
            if any(item.condition_batch is not retained_condition_batch for item in committed):
                raise RuntimeError("one accepted proposal prefix must use one condition batch")
            prepared = getattr(self, "pending_full_context", None)
            if prepared is None:
                raise RuntimeError(
                    "accepted-prefix promotion requires the proposal clean-prefill journal"
                )
            video_tokens = (
                committed[-1].selection[0][1] - committed[0].selection[0][0]
            )
            audio_tokens = (
                committed[-1].selection[1][1] - committed[0].selection[1][0]
            )
            # The preceding proposal already ran its only clean-prefill.  Its
            # fork journals raw pre-RoPE K/V before sink/local eviction, so an
            # in-K acceptance promotes the selected records without another
            # transformer forward.
            with timer.stage("prefix_prepare") if timer else nullcontext():
                self.context = self.context.promote_prefix_from(
                    prepared,
                    video_tokens=video_tokens,
                    audio_tokens=audio_tokens,
                )
        self.official_until = retain_until
        commit_seconds = time.perf_counter() - commit_started
        # The accepted prefix now lives in official generator KV. Both the
        # retained Python records and the replaceable suffix can be dropped;
        # the new K proposal below repopulates the prediction cache.
        self.pending.clear()
        self.pending_full_context = None

        self.ranges = next_ranges
        denoise_started = time.perf_counter()
        with timer.stage("denoise") if timer else nullcontext():
            values = self._sample(condition_batch, selection)
        denoise_seconds = time.perf_counter() - denoise_started
        video_start, audio_start = selection[0][0], selection[1][0]
        proposal = []
        for index in range(retain_until, retain_until + self.chunk_size):
            native_selection = self.ranges[index]
            video, audio = native_selection
            prediction = NativePrediction(
                index=index,
                selection=native_selection,
                values={
                    "video": values["video"][:, video[0] - video_start : video[1] - video_start],
                    "audio": values["audio"][:, audio[0] - audio_start : audio[1] - audio_start],
                },
                condition_version=int(condition_version),
                condition_batch=condition_batch,
            )
            proposal.append(prediction)
            self.pending[index] = prediction
        result = NativeProposal(
            tuple(proposal), tuple(committed), denoise_seconds, commit_seconds
        )
        # Every rank must traverse this phase, even though only the coordinator
        # hands tensors to media. Callback exceptions are broadcast before the
        # next collective-producing model forward.
        callback_error = None
        if on_latents is not None:
            try:
                on_latents(result)
            except Exception as exc:
                callback_error = f"{type(exc).__name__}: {exc}"
        if prepare_full:
            dist = torch.distributed
            if dist.is_initialized():
                errors = [None] * dist.get_world_size()
                dist.all_gather_object(errors, callback_error)
                callback_error = next((error for error in errors if error), None)
            if callback_error:
                raise RuntimeError(f"latent handoff failed: {callback_error}")
            prefill_started = time.perf_counter()
            full_context = self.context.fork()
            with timer.stage("clean_prefill") if timer else nullcontext():
                self.adapter.commit_block(
                    self.model, selection, values, self.context_level, condition_batch, full_context)
            self.pending_full_context = full_context
            from dataclasses import replace
            result = replace(result, clean_prefill_seconds=time.perf_counter() - prefill_started)
        elif callback_error:
            raise RuntimeError(callback_error)
        if timer is not None:
            from dataclasses import replace
            result = replace(result, cuda_stage_seconds=timer.resolve())
        return result

    def execute(self, **kwargs):
        """Compatibility alias for pre-v1 callers; new runtimes use advance."""
        return self.advance(**kwargs)

    def close(self):
        if not self.closed:
            self.context_manager.__exit__(None, None, None)
            self.closed = True

    @staticmethod
    def _hash_tensor(digest, label, value):
        digest.update(label.encode())
        if value is None:
            digest.update(b"none")
            return
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())

    @classmethod
    def _hash_pair_sequence(cls, digest, label, sequence):
        digest.update(f"{label}:{len(sequence)}".encode())
        for index, pair in enumerate(sequence):
            cls._hash_tensor(digest, f"{label}.{index}.key", pair[0])
            cls._hash_tensor(digest, f"{label}.{index}.value", pair[1])

    @classmethod
    def _hash_context(cls, digest, label, context):
        if context is None:
            digest.update(f"{label}:none".encode())
            return
        digest.update(f"{label}:layers:{len(context.layers)}".encode())
        for layer_index, layer in enumerate(context.layers):
            for cache_name in ("video_self", "audio_self", "video_cross", "audio_cross"):
                cache = getattr(layer, cache_name)
                prefix = f"{label}.{layer_index}.{cache_name}"
                metadata = {
                    "window_tokens": cache.window_tokens,
                    "sink_tokens": cache.sink_tokens,
                    "cursor": cache._cursor,
                    "rerope_memory_blocks": cache.rerope_memory_blocks,
                    "rerope_sink_blocks": cache.rerope_sink_blocks,
                    "rerope_block_policy": cache.rerope_block_policy,
                }
                digest.update(json.dumps(metadata, sort_keys=True).encode())
                cls._hash_pair_sequence(
                    digest, prefix + ".reference",
                    () if cache.reference is None else (cache.reference,))
                cls._hash_pair_sequence(
                    digest, prefix + ".sink", cache._rerrope_sink)
                cls._hash_pair_sequence(
                    digest, prefix + ".memory", cache._rerrope_memory)
                cls._hash_pair_sequence(
                    digest, prefix + ".latest_clean_write",
                    () if cache._latest_clean_write is None
                    else (cache._latest_clean_write,))
                for position_name in ("clean_pe", "reference_pe"):
                    pair = getattr(cache, position_name)
                    cls._hash_pair_sequence(
                        digest, prefix + "." + position_name,
                        () if pair is None else (pair,))
            for name in ("video_cond_pe", "audio_cond_pe"):
                pair = getattr(layer, name)
                cls._hash_pair_sequence(
                    digest, f"{label}.{layer_index}.{name}",
                    () if pair is None else (pair,))

    def state_fingerprint(self):
        """Expensive exact digest used only by recovery/parity gates."""
        digest = hashlib.sha256(json.dumps({
            "official_until": self.official_until,
            "chunk_size": self.chunk_size,
            "pending_indices": sorted(self.pending),
        }, sort_keys=True).encode())
        self._hash_context(digest, "official", self.context)
        self._hash_context(digest, "pending_full", self.pending_full_context)
        for index in sorted(self.pending):
            prediction = self.pending[index]
            digest.update(json.dumps({
                "index": prediction.index,
                "selection": prediction.selection,
                "condition_version": prediction.condition_version,
            }, sort_keys=True).encode())
            for name, value in sorted(prediction.values.items()):
                self._hash_tensor(digest, f"pending.{index}.{name}", value)
        self._hash_tensor(digest, "rng", self.random.get_state())
        return {
            "official_until": self.official_until,
            "pending_indices": sorted(self.pending),
            "sha256": digest.hexdigest(),
        }
