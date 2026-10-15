"""Cached sequence parallelism for the resident Wan CF++ generator.

The model weights are replicated. Video tokens are sharded at the first
transformer block, self-attention uses Ulysses head/sequence exchange (and a
two-way ring at world size eight), and the sharded tokens remain local through
cross-attention and the FFN. The model head gathers the video sequence before
unpatchifying.
"""

from __future__ import annotations

import types

import torch
import torch.distributed as dist


def _exchange(tensor, scatter_dim, gather_dim, group, size):
    if size == 1:
        return tensor
    if tensor.shape[scatter_dim] % size:
        raise ValueError("Ulysses scatter dimension is not divisible by its degree")
    send = torch.stack([part.contiguous() for part in tensor.chunk(size, dim=scatter_dim)])
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=group)
    return torch.cat(list(receive.unbind(0)), dim=gather_dim).contiguous()


def _flash_with_lse(query, key, value):
    result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        0.0,
        False,
        False,
        scale=query.shape[-1] ** -0.5,
    )
    return result[0].transpose(1, 2).contiguous(), result[1]


def _local_rope(value, grid, frequencies, start_frame, rank, world):
    """Apply the upstream Wan RoPE to this rank's contiguous spatial shard."""
    frames, height, width = map(int, grid[0].tolist())
    spatial = height * width
    if spatial % world:
        raise ValueError("spatial token count is not divisible by USP degree")
    local_spatial = spatial // world
    complex_channels = value.shape[-1] // 2
    temporal, vertical, horizontal = frequencies.split(
        [complex_channels - 2 * (complex_channels // 3),
         complex_channels // 3, complex_channels // 3],
        dim=1,
    )
    positions = torch.arange(
        rank * local_spatial,
        (rank + 1) * local_spatial,
        device=value.device,
    )
    phase = torch.cat(
        [
            temporal[start_frame:start_frame + frames, None, :].expand(
                frames, local_spatial, -1
            ),
            vertical[positions // width][None].expand(frames, local_spatial, -1),
            horizontal[positions % width][None].expand(frames, local_spatial, -1),
        ],
        dim=-1,
    ).reshape(frames * local_spatial, 1, complex_channels)
    complex_value = torch.view_as_complex(
        value[0].double().reshape(frames * local_spatial, value.shape[2], complex_channels, 2)
    )
    return torch.view_as_real(complex_value * phase).flatten(2).unsqueeze(0).to(value.dtype)


class WanCachedUSP:
    """Install cached USP on a loaded CF++ pipeline."""

    supported_degrees = (2, 4, 8)

    def __init__(self):
        if not dist.is_initialized():
            raise RuntimeError("WanCachedUSP requires an initialized process group")
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        if self.world not in self.supported_degrees:
            raise ValueError(f"unsupported Wan USP degree {self.world}")
        self.ulysses = min(self.world, 4)
        self.ring = self.world // self.ulysses
        self.ulysses_group = None
        self.ring_group = None
        for ring_index in range(self.ring):
            ranks = list(
                range(ring_index * self.ulysses, (ring_index + 1) * self.ulysses)
            )
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.ulysses_group = group
        for head_index in range(self.ulysses):
            ranks = [ring_index * self.ulysses + head_index for ring_index in range(self.ring)]
            group = dist.new_group(ranks)
            if self.rank in ranks:
                self.ring_group = group
        self.ring_rank = self.rank // self.ulysses
        self.head_rank = self.rank % self.ulysses
        self._handles = []

    def _write_cache(self, cache, key, value, current_start):
        full_new_tokens = key.shape[1] * self.ring
        current_end = int(current_start) + full_new_tokens
        global_end = int(cache["global_end_index"].item())
        local_end = int(cache["local_end_index"].item())
        delta = current_end - global_end
        if delta % self.ring:
            raise RuntimeError("global/local Wan cache positions lost ring alignment")
        local_end += delta // self.ring
        capacity = cache["k"].shape[1]
        if local_end > capacity:
            evicted = local_end - capacity
            retained = int(cache["local_end_index"].item()) - evicted
            if retained < 0:
                raise RuntimeError("Wan cache advance exceeds the rolling window")
            if retained:
                cache["k"][:, :retained].copy_(
                    cache["k"][:, evicted:evicted + retained].clone()
                )
                cache["v"][:, :retained].copy_(
                    cache["v"][:, evicted:evicted + retained].clone()
                )
            local_end -= evicted
        local_start = local_end - key.shape[1]
        if local_start < 0 or local_end > capacity:
            raise RuntimeError("Wan cache write falls outside the rolling window")
        cache["k"][:, local_start:local_end].copy_(key)
        cache["v"][:, local_start:local_end].copy_(value)
        cache["global_end_index"].fill_(current_end)
        cache["local_end_index"].fill_(local_end)
        return local_end

    def _attention(
        self,
        module,
        hidden,
        seq_lens,
        grid_sizes,
        frequencies,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        del seq_lens, block_mask, cache_start
        if kv_cache is None:
            raise RuntimeError("resident Wan USP requires the causal KV cache")
        batch, sequence = hidden.shape[:2]
        heads, head_dim = module.num_heads, module.head_dim
        query = module.norm_q(module.q(hidden)).view(batch, sequence, heads, head_dim)
        key = module.norm_k(module.k(hidden)).view(batch, sequence, heads, head_dim)
        value = module.v(hidden).view(batch, sequence, heads, head_dim)
        spatial = int(grid_sizes[0, 1] * grid_sizes[0, 2])
        if int(current_start) % spatial:
            raise ValueError("Wan current_start must be aligned to a latent frame")
        start_frame = int(current_start) // spatial
        query = _local_rope(
            query, grid_sizes, frequencies, start_frame, self.rank, self.world
        )
        key = _local_rope(
            key, grid_sizes, frequencies, start_frame, self.rank, self.world
        )
        query, key, value = _exchange(
            torch.stack([query, key, value]),
            3,
            2,
            self.ulysses_group,
            self.ulysses,
        ).unbind(0)
        local_end = self._write_cache(kv_cache, key, value, current_start)
        cached_key = kv_cache["k"][:, :local_end]
        cached_value = kv_cache["v"][:, :local_end]
        if self.ring == 1:
            output, _ = _flash_with_lse(query, cached_key, cached_value)
        else:
            if self.ring != 2:
                raise RuntimeError("only the validated two-way Wan ring is supported")
            peer = (1 - self.ring_rank) * self.ulysses + self.head_rank
            packed = torch.stack([cached_key, cached_value])
            received = torch.empty_like(packed)
            work = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, packed, peer, self.ring_group),
                    dist.P2POp(dist.irecv, received, peer, self.ring_group),
                ]
            )
            own, own_lse = _flash_with_lse(query, packed[0], packed[1])
            for operation in work:
                operation.wait()
            other, other_lse = _flash_with_lse(query, received[0], received[1])
            total_lse = torch.logaddexp(own_lse, other_lse)
            own_weight = torch.exp(own_lse - total_lse).transpose(1, 2).unsqueeze(-1)
            other_weight = torch.exp(other_lse - total_lse).transpose(1, 2).unsqueeze(-1)
            output = (own.float() * own_weight + other.float() * other_weight).to(query.dtype)
        output = _exchange(output, 1, 2, self.ulysses_group, self.ulysses)
        return module.o(output.flatten(2))

    def install(self, pipeline):
        model = pipeline.generator.model
        if model.num_heads % self.ulysses:
            raise ValueError("Wan attention heads are not divisible by Ulysses degree")
        if int(getattr(pipeline, "local_attn_size", -1)) < 1:
            raise ValueError("Wan cached USP requires a finite rolling KV window")

        def shard_first_block(_module, args, kwargs):
            hidden = args[0]
            frames = kwargs["e"].shape[1]
            hidden = hidden.unflatten(1, (frames, hidden.shape[1] // frames))
            hidden = hidden.chunk(self.world, dim=2)[self.rank].contiguous().flatten(1, 2)
            return (hidden, *args[1:]), kwargs

        def gather_head(_module, _args, output):
            pieces = [torch.empty_like(output) for _ in range(self.world)]
            dist.all_gather(pieces, output.contiguous())
            return torch.cat(pieces, dim=2)

        self._handles.append(
            model.blocks[0].register_forward_pre_hook(shard_first_block, with_kwargs=True)
        )
        self._handles.append(model.head.register_forward_hook(gather_head))
        for block in model.blocks:
            block.self_attn.forward = types.MethodType(
                lambda module, *args, **kwargs: self._attention(module, *args, **kwargs),
                block.self_attn,
            )

        def initialize_kv_cache(owner, batch_size, dtype, device):
            spatial = int(owner.frame_seq_length)
            capacity = int(owner.local_attn_size) * spatial // self.ring
            heads = int(model.num_heads) // self.ulysses
            head_dim = int(model.blocks[0].self_attn.head_dim)
            owner.kv_cache1 = [
                {
                    "k": torch.zeros(
                        [batch_size, capacity, heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, capacity, heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
                for _ in range(owner.num_transformer_blocks)
            ]

        pipeline._initialize_kv_cache = types.MethodType(initialize_kv_cache, pipeline)
        return self


__all__ = ["WanCachedUSP"]
