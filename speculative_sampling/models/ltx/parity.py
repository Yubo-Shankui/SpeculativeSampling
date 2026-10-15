"""Real-weight native vForge rollout vs session adapter differential gate.

This gate proves only the named vForge sampler/session equivalence. It does not
prove that vForge's ReRoPE port matches Fangtai's distinct implementation.
"""
from functools import partial


def export_parallel_fixture(model, batch, output, *, steps, seed=42, chunks=5,
                            diagnose=False, full_cross_inputs=False):
    """Persist full latent tensors + actual RNG draws for cross-USP comparison.

    Correctness-only: synchronous CPU hashing is excluded from latency profiles.
    Each candidate independently consumes its actual native random tensors.
    """
    import json
    from pathlib import Path
    import torch
    import torch.distributed as dist
    from .speculative_sampling_native import NativeSpeculativeSamplingSession
    from .condition_audit import tensor_fingerprint
    rank = dist.get_rank() if dist.is_initialized() else 0
    output = Path(output)
    session = NativeSpeculativeSamplingSession(model, batch, seed=seed, steps=steps)
    if diagnose:
        from .parallel_diagnostics import capture_first_denoise
        capture_first_denoise(session, output / "first_denoise_activations.pt",
                             full_cross_inputs=full_cross_inputs)
    metadata = {
        "seed": seed, "steps": list(steps), "chunks": chunks,
        "usp_size": dist.get_world_size() if dist.is_initialized() else 1,
        "inputs": {
            name: tensor_fingerprint(getattr(batch.model_inputs, name))
            for name in ("reference_video", "reference_audio", "video_context", "audio_context",
                         "context_positions", "video_positions", "audio_positions")
            if getattr(batch.model_inputs, name, None) is not None
        },
        "noise": [], "outputs": [],
    }
    session.noise_observer = lambda tensor: metadata["noise"].append({
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": tensor_fingerprint(tensor)})
    try:
        with torch.inference_mode():
            for index in range(chunks):
                proposal = session.execute(retain_until=index * 45, condition_batch=batch,
                                           condition_version=1, prepare_full=True)
                values = {key: torch.cat([p.values[key] for p in proposal.predictions], 1).cpu()
                          for key in ("video", "audio")}
                name = f"chunk_{index:02d}.pt"
                if rank == 0:
                    output.mkdir(parents=True, exist_ok=True)
                    torch.save(values, output / name)
                metadata["outputs"].append({
                    "file": name,
                    "fingerprints": {key: tensor_fingerprint(value) for key, value in values.items()},
                })
    finally:
        session.close()
    if dist.is_initialized():
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, metadata)
    else:
        ranks = [metadata]
    if rank == 0:
        (output / "fixture.json").write_text(json.dumps({
            "ranks_agree": all(record == ranks[0] for record in ranks),
            "fixture": metadata, "rank_metadata": ranks,
        }, indent=2))


def compare_native_session(model, batch, *, steps, seed=42, chunks=5, chunk_size=45,
                           kv_cache_blocks=2, num_sink_blocks=1):
    import torch
    from vforge.algorithms.dmd.runtime import DmdRolloutSchedule, rollout_generator
    from vforge.tasks.ltx.dmd.generator import LtxDmdGeneratorAdapter
    from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise
    from .speculative_sampling_native import NativeSpeculativeSamplingSession

    if len(batch.model_inputs.layout.video_ranges) != chunks:
        raise ValueError("parity batch must contain exactly the compared native chunks")
    random = torch.Generator(device=batch.clean["video"].device).manual_seed(seed)

    def noise(value):
        return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=random)

    class RecordingAdapter(LtxDmdGeneratorAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.commits = []

        def commit_block(self, model, selection, values, level, batch, context):
            super().commit_block(model, selection, values, level, batch, context)
            self.commits.append({
                "selection": selection,
                "layers": [
                    {name: [pair[0].shape[1] for pair in getattr(layer, name).clean]
                     for name in ("video_self", "audio_self", "video_cross", "audio_cross")}
                    for layer in context.layers
                ],
            })

    baseline_adapter = RecordingAdapter(
        kv_cache_blocks=kv_cache_blocks, num_sink_blocks=num_sink_blocks, rerope=True)
    with torch.inference_mode():
        baseline = rollout_generator(
            model, baseline_adapter, batch, batch.clean,
            resolve_noise_level=partial(resolve_rollout_noise, None),
            schedule=DmdRolloutSchedule(tuple(step / 1000 for step in steps)),
            phase="critic", noise_fn=noise, initial_block_noise_fn=noise,
            noise_dtype=torch.float32, clean_context=True,
        )
        reference_tokens = batch.model_inputs.layout.prefix_frames * batch.model_inputs.layout.frame_tokens
        baseline["video"] = baseline["video"][:, reference_tokens:]
        expected_rng = random.get_state()
        session = NativeSpeculativeSamplingSession(
            model, batch, seed=seed, steps=steps, chunk_size=chunk_size,
            kv_cache_blocks=kv_cache_blocks, num_sink_blocks=num_sink_blocks)
        reports = []
        offsets = {"video": 0, "audio": 0}
        try:
            for index in range(chunks):
                proposal = session.execute(
                    retain_until=index * chunk_size, condition_batch=batch,
                    condition_version=1, prepare_full=True)
                errors = {}
                for key in offsets:
                    actual = torch.cat([p.values[key] for p in proposal.predictions], dim=1)
                    expected = baseline[key][:, offsets[key]:offsets[key] + actual.shape[1]]
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                    maximum = (actual.float() - expected.float()).abs().max().item()
                    errors[key] = {
                        "max_absolute_error": maximum, "finite": finite,
                        "exact": torch.equal(actual, expected),
                        "shape": list(actual.shape),
                    }
                    offsets[key] += actual.shape[1]
                context = session.pending_full_context
                layout = [
                    {name: [pair[0].shape[1] for pair in getattr(layer, name).clean]
                     for name in ("video_self", "audio_self", "video_cross", "audio_cross")}
                    for layer in context.layers
                ]
                reports.append({
                    "chunk": index, "modalities": errors,
                    "cache_layout_matches": layout == baseline_adapter.commits[index]["layers"],
                })
            rng_matches = torch.equal(expected_rng, session.random.get_state())
        finally:
            session.close()
    return {
        "gate": "vforge_full_rollout_vs_speculative_sampling_full_accept",
        "upstream_rerope_parity_proven": False,
        "chunks": chunks, "rng_state_matches": rng_matches,
        "passed": rng_matches and all(
            record["cache_layout_matches"] and all(
                values["finite"] and values["exact"] for values in record["modalities"].values()
            ) for record in reports
        ),
        "records": reports,
    }


def compare_matched_native(model, batch, *, steps, seed=42, chunks=5,
                           kv_cache_blocks=2, num_sink_blocks=1):
    """Differential production r38 against an independent USP1 cache oracle.

    Unlike :func:`compare_native_session`, the reference side does not use
    vForge's ``KVCache`` or cached-attention functions.  It combines the
    ref2av packed-training visibility contract with Fangtai-style pre-RoPE
    record re-anchoring.  This gate intentionally rejects USP>1; topology
    equivalence remains a separate gate.
    """
    from contextlib import nullcontext
    from functools import partial

    import torch

    from vforge.algorithms.dmd.runtime import DmdRolloutSchedule, rollout_generator
    from vforge.distributed.parallel_state import get_parallel_state
    from vforge.tasks.ltx.dmd.generator import LtxDmdGeneratorAdapter
    from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise

    from .matched_native_reference import matched_native_runtime

    if get_parallel_state().config.usp_size != 1:
        raise ValueError("matched-native semantic differential must run at USP1")
    if len(batch.model_inputs.layout.video_ranges) != chunks:
        raise ValueError("parity batch must contain exactly the compared native chunks")

    schedule = DmdRolloutSchedule(tuple(step / 1000 for step in steps))
    reference_tokens = (
        batch.model_inputs.layout.prefix_frames * batch.model_inputs.layout.frame_tokens
    )

    class RecordingAdapter(LtxDmdGeneratorAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.commits = []

        def commit_block(self, model, selection, values, level, active_batch, context):
            super().commit_block(model, selection, values, level, active_batch, context)
            self.commits.append([
                {
                    name: [pair[0].shape[1] for pair in getattr(layer, name).clean]
                    for name in ("video_self", "audio_self", "video_cross", "audio_cross")
                }
                for layer in context.layers
            ])

    def run(reference):
        random = torch.Generator(device=batch.clean["video"].device).manual_seed(seed)

        def noise(value):
            return torch.randn(
                value.shape, device=value.device, dtype=value.dtype, generator=random
            )

        adapter = RecordingAdapter(
            kv_cache_blocks=kv_cache_blocks,
            num_sink_blocks=num_sink_blocks,
            rerope=True,
        )
        runtime = matched_native_runtime() if reference else nullcontext()
        with runtime, torch.inference_mode():
            values = rollout_generator(
                model,
                adapter,
                batch,
                batch.clean,
                resolve_noise_level=partial(resolve_rollout_noise, None),
                schedule=schedule,
                phase="critic",
                noise_fn=noise,
                initial_block_noise_fn=noise,
                noise_dtype=torch.float32,
                clean_context=True,
            )
        values["video"] = values["video"][:, reference_tokens:]
        return values, adapter.commits, random.get_state()

    production = run(False)
    reference = run(True)
    records = []
    for name in ("video", "audio"):
        actual = production[0][name]
        expected = reference[0][name]
        delta = actual.float() - expected.float()
        records.append({
            "modality": name,
            "shape": list(actual.shape),
            "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            "exact": bool(torch.equal(actual, expected)),
            "max_absolute_error": delta.abs().max().item(),
            "normalized_rmse": (
                delta.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-12)
            ).item(),
        })
    cache_layout_exact = production[1] == reference[1]
    rng_exact = torch.equal(production[2], reference[2])
    passed = cache_layout_exact and rng_exact and all(
        record["finite"] and record["exact"] for record in records
    )
    return {
        "gate": "independent_ref2av_rerope_matched_native",
        "matched_native_semantics_proven": passed,
        "reference_uses_production_kv_cache": False,
        "reference_uses_production_cached_attention": False,
        "reference_contracts": [
            "ref2av_full_causal_self_and_cross_visibility",
            "fangtai_pre_rope_read_time_reanchoring",
            "native_record_sink_and_recent_window",
            "ref_follow_false",
        ],
        "chunks": chunks,
        "cache_layout_exact": cache_layout_exact,
        "rng_state_exact": rng_exact,
        "passed": passed,
        "records": records,
    }


def compare_matched_native_partial(model, batch, *, steps, seed=42,
                                   retained=(1, 26, 44, 45), chunk_size=45,
                                   kv_cache_blocks=2, num_sink_blocks=1):
    """Validate the explicit shortened-record extension against the oracle.

    Fangtai's public rollout commits complete blocks. SpeculativeSampling's in-K cut is
    therefore named as a local extension: one accepted contiguous prefix is
    one shortened history record. This differential checks that declaration;
    it does not relabel the operation as an upstream-provided API.
    """
    from contextlib import nullcontext

    import torch

    from vforge.distributed.parallel_state import get_parallel_state

    from .speculative_sampling_native import NativeSpeculativeSamplingSession
    from .matched_native_reference import matched_native_runtime

    if get_parallel_state().config.usp_size != 1:
        raise ValueError("matched-native partial differential must run at USP1")

    def layout(context):
        return [
            {
                name: [pair[0].shape[1] for pair in getattr(layer, name).clean]
                for name in ("video_self", "audio_self", "video_cross", "audio_cross")
            }
            for layer in context.layers
        ]

    def run(boundary, reference):
        runtime = matched_native_runtime() if reference else nullcontext()
        with runtime, torch.inference_mode():
            session = NativeSpeculativeSamplingSession(
                model,
                batch,
                seed=seed,
                steps=steps,
                chunk_size=chunk_size,
                kv_cache_blocks=kv_cache_blocks,
                num_sink_blocks=num_sink_blocks,
            )
            try:
                first = session.execute(
                    retain_until=0,
                    condition_batch=batch,
                    condition_version=1,
                    prepare_full=True,
                )
                second = session.execute(
                    retain_until=boundary,
                    condition_batch=batch,
                    condition_version=1,
                    prepare_full=True,
                )
                outputs = {
                    name: torch.cat(
                        [
                            torch.cat([item.values[name] for item in proposal.predictions], dim=1)
                            for proposal in (first, second)
                        ],
                        dim=1,
                    ).cpu()
                    for name in ("video", "audio")
                }
                return outputs, layout(session.context), session.random.get_state()
            finally:
                session.close()

    records = []
    with torch.inference_mode():
        for boundary in retained:
            production = run(boundary, False)
            reference = run(boundary, True)
            modalities = {}
            for name in ("video", "audio"):
                actual, expected = production[0][name], reference[0][name]
                delta = actual.float() - expected.float()
                modalities[name] = {
                    "shape": list(actual.shape),
                    "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                    "exact": bool(torch.equal(actual, expected)),
                    "max_absolute_error": delta.abs().max().item(),
                }
            records.append({
                "retained": boundary,
                "declared_operation": "one_shortened_native_history_record",
                "cache_layout_exact": production[1] == reference[1],
                "rng_state_exact": torch.equal(production[2], reference[2]),
                "modalities": modalities,
            })
            del production, reference
    passed = all(
        record["cache_layout_exact"]
        and record["rng_state_exact"]
        and all(value["finite"] and value["exact"] for value in record["modalities"].values())
        for record in records
    )
    return {
        "gate": "independent_speculative_sampling_partial_rerope_extension",
        "upstream_defines_partial_commit": False,
        "local_extension_explicit": True,
        "reference_uses_production_kv_cache": False,
        "reference_uses_production_cached_attention": False,
        "retained_boundaries": list(retained),
        "passed": passed,
        "records": records,
    }


def compare_speculative_prefill(model, batches, *, steps, seed=42, retained=(1, 26, 44, 45)):
    """Check repeatability of single-prefill journal promotion on real tensors.

    The legacy lazy-prefix replay deliberately no longer exists: every proposal
    runs one clean-prefill and an accepted partial boundary selects its journaled
    raw K/V without another model forward. Conditions are checked at denoise entry.
    """
    import torch
    from .speculative_sampling_native import NativeSpeculativeSamplingSession

    def run(boundary):
        session = NativeSpeculativeSamplingSession(model, batches[0], seed=seed, steps=steps)
        outputs, inputs, cache = [], [], []
        original = session.adapter.denoise_block

        def observed(model, noisy, selection, level, batch, context):
            inputs.append(next(i for i, candidate in enumerate(batches) if candidate is batch))
            return original(model, noisy, selection, level, batch, context)

        session.adapter.denoise_block = observed
        try:
            for i, position in enumerate((0, boundary, boundary + 26)):
                proposal = session.execute(
                    retain_until=position, condition_batch=batches[i],
                    condition_version=i + 1, prepare_full=True)
                outputs.append({
                    key: torch.cat([p.values[key] for p in proposal.predictions], dim=1).cpu()
                    for key in ("video", "audio")
                })
                cache.append([
                    {name: [(k.cpu(), v.cpu()) for k, v in getattr(layer, name).clean]
                     for name in ("video_self", "audio_self", "video_cross", "audio_cross")}
                    for layer in session.context.layers
                ])
            rng = session.random.get_state()
        finally:
            session.close()
        return outputs, inputs, cache, rng

    reports = []
    with torch.inference_mode():
        for boundary in retained:
            first = run(boundary)
            second = run(boundary)
            latents_match = all(
                torch.equal(left[key], right[key]) and torch.isfinite(right[key]).all().item()
                for left, right in zip(first[0], second[0], strict=True)
                for key in ("video", "audio")
            )
            cache_match = True
            for left_state, right_state in zip(first[2], second[2], strict=True):
                for left, right in zip(left_state, right_state, strict=True):
                    for name in left:
                        if len(left[name]) != len(right[name]):
                            cache_match = False
                            continue
                        cache_match &= all(
                            torch.equal(a, b) for lp, rp in zip(left[name], right[name])
                            for a, b in zip(lp, rp)
                        )
            steps_per_chunk = len(steps) - 1
            expected_conditions = [i for i in range(3) for _ in range(steps_per_chunk)]
            reports.append({
                "retained": boundary, "latents_exact": latents_match,
                "official_cache_exact": cache_match,
                "actual_denoise_conditions": second[1],
                "condition_routing_passed": first[1] == second[1] == expected_conditions,
                "rng_exact": torch.equal(first[3], second[3]),
            })
            del first, second
    return {
        "gate": "single_prefill_journal_repeatability",
        "independent_upstream_partial_semantics_proven": False,
        "passed": all(r["latents_exact"] and r["official_cache_exact"]
                      and r["condition_routing_passed"] and r["rng_exact"] for r in reports),
        "records": reports,
    }
