"""Experimental execution-only probes; never enabled by production defaults.

Attention heads are mathematically independent. Fixing the per-call head count
and contiguous layout can remove topology-dependent backend kernel choices.
This still requires measured numerical equivalence before production adoption.
"""
from functools import wraps


def fixed_head_calls(kernel, head_count):
    if type(head_count) is not int or head_count < 1:
        raise ValueError("fixed head count must be a positive integer")

    @wraps(kernel)
    def call(q, k, v, **kwargs):
        import torch
        if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
                or q.shape[2] != k.shape[2] or k.shape[2] != v.shape[2]
                or q.shape[2] % head_count):
            raise ValueError("fixed-head probe requires divisible equal-head BTHD Q/K/V")
        if kwargs.get("attn_mask") is not None or kwargs.get("sparsity") is not None:
            raise ValueError("fixed-head probe is restricted to dense native A/V attention")
        return torch.cat([
            kernel(q[:, :, i:i+head_count].contiguous(),
                   k[:, :, i:i+head_count].contiguous(),
                   v[:, :, i:i+head_count].contiguous(), **kwargs)
            for i in range(0, q.shape[2], head_count)
        ], dim=2)
    return call


def install_fixed_cross_heads(model, head_count):
    """Return a restoration function. No weights, positions or caches change."""
    previous = []
    for block in model.transformer_blocks:
        for name in ("audio_to_video_attn", "video_to_audio_attn"):
            attention = getattr(block, name)
            previous.append((attention, attention.attention_function))
            attention.attention_function = fixed_head_calls(
                attention.attention_function, head_count)

    def restore():
        for attention, kernel in previous:
            attention.attention_function = kernel
    return restore


def install_text_math_attention(model):
    """Diagnostic only: native SDPA math path for text attention, masks intact.

    Query-shard lengths differ across USP. This isolates masked text-kernel
    selection without changing text visibility, weights, K/V or projections.
    It is not enabled in a deployed adapter; compare against native numerics
    at the same degree as well as across degrees before considering adoption.
    """
    from functools import partial
    from torch.nn.attention import SDPBackend
    from vforge.ops import sdpa_attention
    from vforge.models.ltx.ops import attention_in_value_dtype
    kernel = partial(attention_in_value_dtype, partial(
        sdpa_attention, priority=(SDPBackend.MATH,), contiguous=False))
    previous = []
    for block in model.transformer_blocks:
        for name in ("attn2", "audio_attn2"):
            attention = getattr(block, name)
            previous.append((attention, attention.attention_function,
                             attention.masked_attention_function))
            attention.attention_function = kernel
            attention.masked_attention_function = kernel

    def restore():
        for attention, unmasked, masked in previous:
            attention.attention_function = unmasked
            attention.masked_attention_function = masked
    return restore


def fixed_row_projection(forward, rows):
    """Diagnostic constant-M GEMM calls, retaining the native projection."""
    if type(rows) is not int or rows < 1:
        raise ValueError("projection row count must be positive")

    @wraps(forward)
    def call(value):
        import torch
        if value.ndim < 2:
            raise ValueError("projection input must have token and feature dimensions")
        flat = value.reshape(-1, value.shape[-1])
        if not flat.shape[0]:
            return forward(value)
        parts = []
        for start in range(0, flat.shape[0], rows):
            block = flat[start:start+rows].contiguous()
            count = block.shape[0]
            if count < rows:
                block = torch.cat((block, block.new_zeros(rows-count, block.shape[-1])), 0)
            parts.append(forward(block)[:count])
        result = torch.cat(parts, dim=0)
        return result.reshape(*value.shape[:-1], result.shape[-1])
    return call


def install_fixed_gate_rows(model, rows):
    """All native attention gate projections, no replacement weights or math."""
    previous = []
    for block in model.transformer_blocks:
        for attention in block.modules():
            projection = getattr(attention, "to_gate_logits", None)
            if projection is not None:
                previous.append((projection, projection.forward))
                projection.forward = fixed_row_projection(projection.forward, rows)

    def restore():
        for projection, forward in previous:
            projection.forward = forward
    return restore


def canonical_gate_projection(forward, value, *, global_length, canonical_degree,
                              rank, world, gather):
    """Reference-shape probe: emulate an unchanged canonical USP gate GEMM.

    Expensive full-sequence gather is diagnostic only. Unlike fixed-row padding,
    this retains the canonical native three-dimensional per-rank input shape.
    The canonical topology calls the original forward directly.
    """
    import torch
    if global_length is None or world == canonical_degree:
        return forward(value)
    if value.ndim != 3 or canonical_degree < 1:
        raise ValueError("canonical gate requires BTD activations and a positive degree")
    full = gather(value)[:, :global_length]
    rows = (global_length + canonical_degree - 1) // canonical_degree
    missing = rows * canonical_degree - global_length
    if missing:
        full = torch.cat((full, full.new_zeros(full.shape[0], missing, full.shape[-1])), 1)
    output = torch.cat([
        forward(full[:, start:start+rows].contiguous())
        for start in range(0, rows * canonical_degree, rows)
    ], dim=1)[:, :global_length]
    local = value.shape[1]
    start, end = rank * local, min((rank + 1) * local, global_length)
    result = output[:, start:end]
    if result.shape[1] < local:
        result = torch.cat((result, result.new_zeros(
            result.shape[0], local-result.shape[1], result.shape[-1])), dim=1)
    return result


def install_canonical_gate_shards(model, canonical_degree):
    """Reproduce native canonical-USP gate projection shape, no weight changes."""
    import torch
    import torch.distributed as dist
    from vforge.distributed.ulysses import get_usp_group
    if type(canonical_degree) is not int or canonical_degree < 1:
        raise ValueError("canonical gate degree must be positive")
    previous, handles = [], []
    for block in model.transformer_blocks:
        lengths = {}

        def before(module, args, kwargs, lengths=lengths):
            for name, values in zip(("video", "audio"), args[:2], strict=True):
                lengths[name] = None if values is None else values.global_sequence_length

        handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
        for name, modality in (
            ("attn1", "video"), ("attn2", "video"),
            ("audio_attn1", "audio"), ("audio_attn2", "audio"),
            ("audio_to_video_attn", "video"), ("video_to_audio_attn", "audio"),
        ):
            attention = getattr(block, name, None)
            projection = getattr(attention, "to_gate_logits", None)
            if projection is None:
                continue
            original = projection.forward
            previous.append((projection, original))

            def projected(value, original=original, modality=modality, lengths=lengths):
                group = get_usp_group()
                world = dist.get_world_size(group) if dist.is_initialized() else 1
                rank = dist.get_rank(group) if dist.is_initialized() else 0

                def gather(local):
                    if world == 1:
                        return local
                    parts = [torch.empty_like(local) for _ in range(world)]
                    dist.all_gather(parts, local.contiguous(), group=group)
                    return torch.cat(parts, dim=1)

                return canonical_gate_projection(
                    original, value, global_length=lengths.get(modality),
                    canonical_degree=canonical_degree, rank=rank, world=world, gather=gather)

            projection.forward = projected

    def restore():
        for handle in handles:
            handle.remove()
        for projection, forward in previous:
            projection.forward = forward
    return restore
