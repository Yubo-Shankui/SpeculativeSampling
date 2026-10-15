"""First-denoise activation samples to locate cross-USP numerical divergence.

Diagnostic only. Collectives gather identical global token locations; hooks
never modify model outputs. Do not include diagnostic runs in speed profiles.
"""
from __future__ import annotations


def capture_first_denoise(session, output, *, full_cross_inputs=False):
    import torch
    import torch.distributed as dist
    from pathlib import Path
    from vforge.models.ltx.kv_cache import CacheMode

    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    active = {"enabled": False, "lengths": {}}
    samples = {}
    full_tensors = {}
    handles = []
    restorations = []
    original = session.adapter.denoise_block

    def collect(name, tensor, global_length):
        if tensor is None or tensor.ndim != 3 or not tensor.shape[1]:
            return
        indices = torch.linspace(0, global_length - 1, min(32, global_length),
                                 device=tensor.device).long().unique()
        local_length = tensor.shape[1]
        offset = rank * local_length
        selected = tensor.new_zeros((tensor.shape[0], len(indices), tensor.shape[-1]))
        mask = (indices >= offset) & (indices < offset + local_length)
        selected[:, mask] = tensor[:, indices[mask]-offset]
        if world > 1:
            dist.all_reduce(selected, op=dist.ReduceOp.SUM)
        if rank == 0:
            samples[name] = selected.detach().cpu()
        if full_cross_inputs and name.startswith("00.audio."):
            # Audio has only ~377 tokens. Keep every row through first-layer
            # self/text/projection stages; sparse near-rounding differences
            # were invisible to the original 32-token sampling.
            if world > 1:
                parts = [torch.empty_like(tensor) for _ in range(world)]
                dist.all_gather(parts, tensor.contiguous())
                complete = torch.cat(parts, dim=1)[:, :global_length]
            else:
                complete = tensor[:, :global_length]
            if rank == 0:
                full_tensors[name] = complete.detach().cpu()

    for index, block in enumerate(session.model.transformer_blocks):
        def before(module, args, kwargs, index=index):
            if not active["enabled"] or kwargs.get("cache_mode") != CacheMode.DENOISE:
                return
            video, audio = args[:2]
            for name, values in (("video", video), ("audio", audio)):
                if values is not None:
                    length = values.global_sequence_length or values.x.shape[1]
                    active["lengths"][name] = length
                    collect(f"{index:02d}.{name}.input", values.x, length)

        def after(module, args, kwargs, outputs, index=index):
            if not active["enabled"] or kwargs.get("cache_mode") != CacheMode.DENOISE:
                return
            for name, values in zip(("video", "audio"), outputs, strict=True):
                if values is not None:
                    collect(f"{index:02d}.{name}.output", values.x, active["lengths"][name])

        handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(block.register_forward_hook(after, with_kwargs=True))
        # Detailed first-block probes bracket the first attention computation.
        if index == 0:
            for attn_name in ("audio_to_video_attn", "video_to_audio_attn"):
                attn = getattr(block, attn_name)
                kernel = attn.attention_function

                def kernel_probe(q, k, v, *, _kernel=kernel, _name=attn_name,
                                 _heads=attn.heads, **kwargs):
                    if not active["enabled"]:
                        return _kernel(q, k, v, **kwargs)

                    def save_heads(name, value):
                        # Kernel inputs already have global sequence and local
                        # heads. Reassemble heads, not sequence, for comparison.
                        indices = torch.linspace(
                            0, value.shape[1]-1, min(32, value.shape[1]),
                            device=value.device).long().unique()
                        selected = value[:, indices].contiguous()
                        if world > 1:
                            parts = [torch.empty_like(selected) for _ in range(world)]
                            dist.all_gather(parts, selected)
                            selected = torch.cat(parts, dim=2)[:, :, :_heads]
                        if rank == 0:
                            samples[f"00.kernel.{_name}.{name}"] = selected.cpu()
                        if full_cross_inputs:
                            # Samples can miss a single divergent context token
                            # which affects every query. Gather one head at a
                            # time to bound diagnostic GPU scratch allocation.
                            cpu_heads = []
                            for head in range(value.shape[2]):
                                local = value[:, :, head:head+1].contiguous()
                                if world > 1:
                                    parts = [torch.empty_like(local) for _ in range(world)]
                                    dist.all_gather(parts, local)
                                else:
                                    parts = [local]
                                if rank == 0:
                                    cpu_heads.append([part.cpu() for part in parts])
                            if rank == 0:
                                full_tensors[f"00.kernel.{_name}.{name}"] = torch.cat([
                                    cpu_heads[h][r] for r in range(world)
                                    for h in range(value.shape[2])
                                ], dim=2)[:, :, :_heads].contiguous()
                    for name, value in (("q", q), ("k", k), ("v", v)):
                        save_heads(name, value)
                    result = _kernel(q, k, v, **kwargs)
                    save_heads("output", result)
                    return result

                restorations.append((attn, kernel))
                attn.attention_function = kernel_probe
                for label in ("to_q", "to_k", "to_v", "q_norm", "k_norm"):
                    module = getattr(attn, label)
                    modality = ("audio" if (attn_name == "audio_to_video_attn"
                                            and label in ("to_k", "to_v", "k_norm"))
                                or (attn_name == "video_to_audio_attn"
                                    and label in ("to_q", "q_norm")) else "video")

                    def cross_before(module, args, name=attn_name, label=label, modality=modality):
                        if active["enabled"]:
                            collect(f"00.{modality}.cross.{name}.{label}.input",
                                    args[0], active["lengths"][modality])

                    def cross_after(module, args, result, name=attn_name, label=label, modality=modality):
                        if active["enabled"]:
                            collect(f"00.{modality}.cross.{name}.{label}.output",
                                    result, active["lengths"][modality])

                    handles.append(module.register_forward_pre_hook(cross_before))
                    handles.append(module.register_forward_hook(cross_after))
            for modality, prefix in (("video", "attn1"), ("audio", "audio_attn1")):
                attn = getattr(block, prefix, None)
                if attn is None:
                    continue
                for label in ("to_q", "to_k", "to_v", "q_norm", "k_norm", "to_out"):
                    module = getattr(attn, label)

                    def detail(module, args, result, modality=modality, label=label):
                        if active["enabled"] and modality in active["lengths"]:
                            collect(f"00.{modality}.self.{label}", result,
                                    active["lengths"][modality])

                    handles.append(module.register_forward_hook(detail))
            text_attn = block.audio_attn2
            for attribute in ("attention_function", "masked_attention_function"):
                text_kernel = getattr(text_attn, attribute)

                def text_kernel_probe(q, k, v, *, _kernel=text_kernel, **kwargs):
                    if active["enabled"]:
                        collect("00.audio.text.kernel.q", q.flatten(2), active["lengths"]["audio"])
                        # K/V and text mask are replicated along heads, not
                        # exchanged by USP; keep rank-0's full context tensors.
                        if rank == 0 and full_cross_inputs:
                            full_tensors["00.text.context.k"] = k.detach().cpu()
                            full_tensors["00.text.context.v"] = v.detach().cpu()
                        mask = kwargs.get("attn_mask")
                        if mask is not None and mask.ndim == 4 and mask.shape[-2] == q.shape[1]:
                            collect("00.audio.text.kernel.mask",
                                    mask.transpose(1, 2).flatten(2), active["lengths"]["audio"])
                    output = _kernel(q, k, v, **kwargs)
                    if active["enabled"]:
                        collect("00.audio.text.kernel.output",
                                output.flatten(2), active["lengths"]["audio"])
                    return output

                # Different attribute than the cross-attention restorations.
                setattr(text_attn, attribute, text_kernel_probe)
                handles.append(_RestoreAttribute(text_attn, attribute, text_kernel))
            for label in ("to_q", "q_norm", "to_out", "to_gate_logits"):
                if getattr(block.audio_attn2, label, None) is None:
                    continue
                module = getattr(block.audio_attn2, label)

                def text_before(module, args, label=label):
                    if active["enabled"]:
                        collect(f"00.audio.text.{label}.input", args[0],
                                active["lengths"]["audio"])

                def text_after(module, args, result, label=label):
                    if active["enabled"]:
                        collect(f"00.audio.text.{label}.output", result,
                                active["lengths"]["audio"])

                handles.append(module.register_forward_pre_hook(text_before))
                handles.append(module.register_forward_hook(text_after))
            for module_name, modality in (
                ("attn2", "video"), ("audio_attn2", "audio"),
                ("ff", "video"), ("audio_ff", "audio"),
                ("video_to_audio_attn.to_out", "audio"),
                ("audio_to_video_attn.to_out", "video"),
                ("audio_ff.net.0.proj", "audio"),
                ("audio_ff.net.0", "audio"), ("audio_ff.net.2", "audio"),
            ):
                module = block
                for component in module_name.split("."):
                    module = getattr(module, component, None)
                    if module is None:
                        break
                if module is None:
                    continue

                def extra_before(module, args, modality=modality, module_name=module_name):
                    if active["enabled"] and args and hasattr(args[0], "ndim"):
                        collect(f"00.{modality}.{module_name}.input", args[0],
                                active["lengths"][modality])

                def extra_after(module, args, result, modality=modality, module_name=module_name):
                    if active["enabled"]:
                        collect(f"00.{modality}.{module_name}.output", result,
                                active["lengths"][modality])

                handles.append(module.register_forward_pre_hook(extra_before))
                handles.append(module.register_forward_hook(extra_after))

    called = False

    def observed(*args, **kwargs):
        nonlocal called
        if called:
            return original(*args, **kwargs)
        called = True
        active["enabled"] = True
        try:
            result = original(*args, **kwargs)
        finally:
            active["enabled"] = False
            for handle in handles:
                handle.remove()
            for attn, kernel in restorations:
                attn.attention_function = kernel
        if rank == 0:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(samples, path)
            if full_cross_inputs:
                torch.save(full_tensors, path.with_name("first_denoise_full_cross.pt"))
        samples.clear()
        full_tensors.clear()
        return result

    session.adapter.denoise_block = observed


class _RestoreAttribute:
    """Match hook handle cleanup without adding a model/runtime dependency."""
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def remove(self):
        setattr(self.module, self.name, self.value)
