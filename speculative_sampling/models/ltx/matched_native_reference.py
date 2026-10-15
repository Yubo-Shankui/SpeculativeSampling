"""Independent USP1 reference for the ref2av + ReRoPE cache contract.

This module is an acceptance oracle, not a serving implementation.  It keeps
its own per-record cache and attention routing so a differential does not run
both sides through vForge's production ``KVCache``/``cached_attention`` code.
The visibility contract comes from ref2av's packed training graph; the
pre-RoPE re-anchoring contract comes from Fangtai's ``rerope_cache.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy

import torch


Pair = tuple[torch.Tensor, torch.Tensor]


class MatchedNativeKV:
    """Small record cache with fixed sink records and a rolling record tail."""

    def __init__(self, *, dtype=None, rerope_memory_blocks=0, rerope_sink_blocks=0, **_):
        if rerope_memory_blocks < 1:
            raise ValueError("matched-native oracle requires a bounded ReRoPE window")
        self.dtype = dtype
        self.memory_blocks = int(rerope_memory_blocks)
        self.sink_blocks = int(rerope_sink_blocks)
        self.reference: Pair | None = None
        self.sink: list[Pair] = []
        self.memory: list[Pair] = []
        self.clean_pe: Pair | None = None
        self.reference_pe: Pair | None = None

    @property
    def clean(self) -> tuple[Pair, ...]:
        return tuple((*self.sink, *self.memory))

    def _store(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(dtype=self.dtype).clone()

    def write(self, key, value, mode) -> None:
        from vforge.models.ltx.kv_cache import CacheMode

        pair = (self._store(key), self._store(value))
        if mode == CacheMode.PREFILL_REF:
            if self.reference is not None or self.clean:
                raise RuntimeError("reference must be the first oracle cache write")
            self.reference = pair
        elif mode == CacheMode.PREFILL_CLEAN:
            if self.reference is None:
                raise RuntimeError("reference must be written before clean history")
            if len(self.sink) < self.sink_blocks:
                self.sink.append(pair)
            else:
                self.memory = [*self.memory, pair][-self.memory_blocks :]
        else:
            raise ValueError("denoise cannot mutate the oracle cache")

    def shard_reference_heads(self, *, size, **_) -> None:
        if size != 1:
            raise NotImplementedError("matched-native oracle is intentionally USP1-only")

    def fork(self):
        result = copy.copy(self)
        result.sink = list(self.sink)
        result.memory = list(self.memory)
        return result

    def snapshot(self):
        # Cached tensors are detached immutable values. A list copy prevents a
        # later speculative append from changing the denoise view's records.
        return self.fork()


@dataclass
class MatchedNativeLayer:
    video_self: MatchedNativeKV
    audio_self: MatchedNativeKV
    video_cross: MatchedNativeKV
    audio_cross: MatchedNativeKV
    video_cond_pe: Pair | None = None
    audio_cond_pe: Pair | None = None

    def snapshot(self):
        return MatchedNativeLayer(
            self.video_self.snapshot(),
            self.audio_self.snapshot(),
            self.video_cross.snapshot(),
            self.audio_cross.snapshot(),
            self.video_cond_pe,
            self.audio_cond_pe,
        )


class MatchedNativeModelCache:
    def __init__(
        self,
        num_layers,
        *,
        dtype=None,
        rerope_memory_blocks=0,
        rerope_sink_blocks=0,
        rerope_cross_attention=False,
        **_,
    ):
        if not rerope_cross_attention:
            raise ValueError("ref2av matched-native oracle requires cross-cache ReRoPE")
        options = dict(
            dtype=dtype,
            rerope_memory_blocks=rerope_memory_blocks,
            rerope_sink_blocks=rerope_sink_blocks,
        )
        self.layers = [
            MatchedNativeLayer(*(MatchedNativeKV(**options) for _ in range(4)))
            for _ in range(num_layers)
        ]

    def set_cond_pe(self, video_pe, audio_pe) -> None:
        for layer in self.layers:
            layer.video_cond_pe, layer.audio_cond_pe = video_pe, audio_pe

    def set_window_pe(self, video_pe, audio_pe) -> None:
        for layer in self.layers:
            layer.video_self.clean_pe, layer.audio_self.clean_pe = video_pe, audio_pe

    def set_cross_window_pe(self, video_pe, audio_pe, video_ref_pe, audio_ref_pe) -> None:
        for layer in self.layers:
            layer.video_cross.clean_pe = video_pe
            layer.audio_cross.clean_pe = audio_pe
            layer.video_cross.reference_pe = video_ref_pe
            layer.audio_cross.reference_pe = audio_ref_pe

    def fork(self):
        result = copy.copy(self)
        result.layers = [
            MatchedNativeLayer(
                layer.video_self.fork(),
                layer.audio_self.fork(),
                layer.video_cross.fork(),
                layer.audio_cross.fork(),
                layer.video_cond_pe,
                layer.audio_cond_pe,
            )
            for layer in self.layers
        ]
        return result


def _rotate(raw, pe, rope_type):
    from vforge.models.ltx.rope import apply_rotary_emb

    if raw is None or not raw.shape[1]:
        return raw
    if pe is None:
        raise ValueError("matched-native cached raw key has no current-slot PE")
    return apply_rotary_emb(raw, tuple(part.to(raw.dtype) for part in pe), rope_type)


def _history(cache: MatchedNativeKV, attn) -> Pair | None:
    records = cache.clean
    if not records:
        return None
    raw = torch.cat([record[0] for record in records], dim=1)
    value = torch.cat([record[1] for record in records], dim=1)
    return _rotate(raw, cache.clean_pe, attn.rope_type), value


def _cat_pairs(pairs):
    valid = [pair for pair in pairs if pair is not None and pair[0].shape[1]]
    if not valid:
        raise ValueError("matched-native attention has no visible keys")
    return tuple(torch.cat([pair[index] for pair in valid], dim=1) for index in (0, 1))


def matched_native_self_attention(
    attn,
    x,
    context,
    pe,
    k_pe,
    cache,
    mode,
    cond_pe=None,
    perturbation_mask=None,
    all_perturbed=False,
    sequence_length=None,
    clean_attends_reference=False,
):
    """Ref2av cached self-attention without production cache helpers."""
    from vforge.models.ltx.kv_cache import CacheMode
    from vforge.models.ltx.rope import apply_rotary_emb

    if sequence_length is not None:
        raise NotImplementedError("matched-native oracle is intentionally USP1-only")
    if (all_perturbed or perturbation_mask is not None) and mode != CacheMode.DENOISE:
        raise ValueError("attention perturbations are denoise-only")
    context = x if context is None else context
    value = attn.to_v(context)
    if all_perturbed:
        output = value
    else:
        query = attn.q_norm(attn.to_q(x))
        raw_key = attn.k_norm(attn.to_k(context))
        key_pe = pe if k_pe is None else k_pe
        if pe is not None:
            query = apply_rotary_emb(query, pe, attn.rope_type)
        key = apply_rotary_emb(raw_key, key_pe, attn.rope_type) if key_pe is not None else raw_key
        reference = None
        if cache.reference is not None and cache.reference[0].shape[1]:
            reference = (_rotate(cache.reference[0], cond_pe, attn.rope_type), cache.reference[1])
        if mode == CacheMode.PREFILL_REF:
            cache.write(raw_key, value, mode)
            pairs = [(key, value)]
        elif mode == CacheMode.PREFILL_CLEAN:
            pairs = [reference if clean_attends_reference else None, _history(cache, attn), (key, value)]
            cache.write(raw_key, value, mode)
        elif mode == CacheMode.DENOISE:
            pairs = [reference, _history(cache, attn), (key, value)]
        else:
            raise ValueError(f"unsupported cache mode: {mode}")
        keys, values = _cat_pairs(pairs)
        output = attn._run_attention(
            attn.attention_function, query, keys, values, mask=None, sparsity=None
        )
        if perturbation_mask is not None:
            output = output * perturbation_mask + value * (1 - perturbation_mask)
    if attn.to_gate_logits is not None:
        output = attn.gated_attention_function(x, output, attn)
    return attn.to_out(output)


def matched_native_cross_attention(
    attn,
    x,
    context,
    pe,
    k_pe,
    cache,
    mode,
    *,
    query_length=None,
    context_length=None,
):
    """Full-causal ref2av A/V cache with read-time ReRoPE."""
    from vforge.models.ltx.kv_cache import CacheMode
    from vforge.models.ltx.rope import apply_rotary_emb

    if query_length is not None or context_length is not None:
        raise NotImplementedError("matched-native oracle is intentionally USP1-only")
    value = attn.to_v(context)
    raw_key = attn.k_norm(attn.to_k(context))
    key = apply_rotary_emb(raw_key, k_pe, attn.rope_type) if k_pe is not None else raw_key
    key = key.to(value.dtype)
    if mode == CacheMode.PREFILL_REF:
        cache.write(raw_key, value, mode)
        return None
    if x is None:
        raise ValueError("cross-modal query is required")
    reference = None
    if cache.reference is not None and cache.reference[0].shape[1]:
        reference = (
            _rotate(cache.reference[0], cache.reference_pe, attn.rope_type),
            cache.reference[1],
        )
    keys, values = _cat_pairs([reference, _history(cache, attn), (key, value)])
    if mode == CacheMode.PREFILL_CLEAN:
        cache.write(raw_key, value, mode)
    elif mode != CacheMode.DENOISE:
        raise ValueError(f"unsupported cross cache mode: {mode}")
    query = attn.q_norm(attn.to_q(x))
    if pe is not None:
        query = apply_rotary_emb(query, pe, attn.rope_type)
    output = attn._run_attention(
        attn.attention_function, query, keys, values, mask=None, sparsity=None
    )
    if attn.to_gate_logits is not None:
        output = attn.gated_attention_function(x, output, attn)
    return attn.to_out(output)


@contextmanager
def matched_native_runtime():
    """Temporarily route one USP1 rollout through the independent oracle."""
    import vforge.models.ltx.causal as causal
    import vforge.tasks.ltx.dmd.generator as generator

    originals = (
        causal.cached_attention,
        causal.cached_cross_attention,
        generator.ModelKVCache,
    )
    causal.cached_attention = matched_native_self_attention
    causal.cached_cross_attention = matched_native_cross_attention
    generator.ModelKVCache = MatchedNativeModelCache
    try:
        yield
    finally:
        causal.cached_attention, causal.cached_cross_attention, generator.ModelKVCache = originals
