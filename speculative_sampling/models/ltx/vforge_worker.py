"""Single-B300 resident vForge/LTX-DMD worker for the shared live demo."""

from __future__ import annotations

import importlib.util
import sys
import time
import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


class VForgeLtxWorker:
    model_id = "ltx23_dmd"
    mock = False

    def __init__(self, args):
        self.args = args
        self.split_services = bool(getattr(args, "text_endpoint", None))
        if self.split_services != bool(getattr(args, "media_endpoint", None)):
            raise ValueError("text and media endpoints must be configured together")
        self.text_client = self.media_client = None
        self.media_tensor_channel = None
        self.condition_recoveries = []
        self.media_recoveries = []
        if self.split_services:
            from speculative_sampling.runtime.rpc import TensorClient
            from speculative_sampling.runtime.recovery_journal import MediaCheckpointJournal
            token = Path(args.service_token_file).read_text().strip()
            self.text_client = TensorClient(args.text_endpoint, token)
            self.media_client = TensorClient(args.media_endpoint, token)
            self.media_journal = MediaCheckpointJournal(args.output)
            if args.preload_prompts:
                raise ValueError("split text service manages its own condition cache")
        if not args.vforge_repo or not args.config:
            raise ValueError("the vForge worker requires --vforge-repo and --config")
        sys.path.insert(0, str(Path(args.vforge_repo).resolve()))

        import torch
        from omegaconf import OmegaConf
        from vforge.distributed.parallelize import module_has_meta_tensors
        from vforge.distributed.runtime import initialize_runtime
        from vforge.evaluation.runner import _load_initial_weights
        from vforge.hydra_config import StandaloneConfig
        from vforge.models.factory import resolve_model
        from vforge.models.ltx.decoding import LtxDecoder
        from vforge.tasks.common import resolve_config
        from vforge.tasks.ltx.dmd.eval import build_eval_components
        from vforge.tasks.translate import build_ops_config, build_parallelize_config
        from vforge.training.engine import EnginePolicy, create_engine

        self.torch = torch
        cfg = resolve_config(OmegaConf.merge(
            OmegaConf.structured(StandaloneConfig), OmegaConf.load(args.config)
        ))
        from speculative_sampling.runtime.execution import apply_native_execution
        cfg = apply_native_execution(cfg, getattr(args, "validated_native_execution", None))
        cfg = replace(
            cfg,
            parallel=replace(
                cfg.parallel,
                usp_size=args.usp_size,
                dp_size=1,
                dp_replicate_size=1,
                dp_shard_size=1,
                # Serving keeps the inference-only shards resident. Training
                # and evaluation configs may enable reshard/checkpointing, but
                # neither belongs in a latency-sensitive forward-only worker.
                reshard_after_forward=False,
                gradient_checkpointing=False,
            ),
            model=replace(cfg.model, extras={
                **cfg.model.extras,
                # The K45 checkpoint was trained with K45 as its physical
                # generation/cache block. SpeculativeSampling must not rewrite this to
                # a q=1 layout merely to expose finer delivery boundaries.
                "first_block_latent_frames": args.chunk_latents,
                "block_latent_frames": args.chunk_latents,
                "seconds": None,
            }),
            evaluation=replace(cfg.evaluation, num_samples=1, sample_selection="first"),
        )
        parallelize = build_parallelize_config(cfg)
        self.runtime = initialize_runtime(
            parallelize.parallel,
            build_ops_config(cfg.attention, deterministic=cfg.runtime.deterministic),
        )
        components = build_eval_components(cfg)
        self.prompt_contexts = {}
        self.prompt_encoder = None
        if args.preload_prompts:
            from vforge.models.ltx.text_encoder import encode_prompts

            document = json.loads(Path(args.preload_prompts).read_text())
            prompts = []
            for trace in document.get("traces", [document]):
                for request in trace["requests"]:
                    prompt = str(request["prompt"])
                    if prompt not in prompts:
                        prompts.append(prompt)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            contexts = None
            if rank == 0:
                encoded = encode_prompts(
                    cfg.model.pretrained_path,
                    cfg.model.extras["text_encoder_path"],
                    prompts,
                    device=self.runtime.device,
                    max_length=cfg.model.extras["max_text_length"],
                )
                contexts = {
                    prompt: {key: value.cpu() for key, value in context.items()}
                    for prompt, context in zip(prompts, encoded, strict=True)
                }
            if torch.distributed.is_initialized():
                payload = [contexts]
                torch.distributed.broadcast_object_list(
                    payload, src=0, device=self.runtime.device
                )
                contexts = payload[0]
            self.prompt_contexts = contexts or {}
        model = resolve_model(components.model)
        if not module_has_meta_tensors(model):
            model = model.to(self.runtime.device)
        engine = create_engine(
            cfg.parallel.engine,
            EnginePolicy(parallelize=parallelize, run_backward=False),
        )
        model = engine.prepare_model(model)
        model = engine.materialize_model(model, self.runtime.device)
        _load_initial_weights(
            model, components.initial_weights, weights_strict=cfg.runtime.weights_strict
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            model = model.to(dtype=torch.bfloat16)
        model = model.requires_grad_(False).eval()
        self.cfg = cfg
        self.components = components
        self.model = model
        self.sample = next(iter(components.dataloader))
        self.reference_sample = None
        self.reference_key = None
        self.inference = components.build_inference(model, lambda **kw: model(**kw))
        self.decoder = LtxDecoder(cfg.model.pretrained_path, self.runtime.device)
        decoder_source = (
            Path(args.interactive_streaming_repo)
            / "packages/ltx-dmd/src/ltx_dmd/streaming_decoder.py"
        )
        if not decoder_source.is_file():
            raise FileNotFoundError(f"missing upstream streaming decoder: {decoder_source}")
        spec = importlib.util.spec_from_file_location("ltx_upstream_streaming_decoder", decoder_source)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        upstream_streaming_decoder = module.ZeroLookaheadAVStreamingDecoder

        class ForkableStreamingDecoder(upstream_streaming_decoder):
            """Upstream decoder with speculative state fork/rollback support.

            Model weights remain resident and shared.  Only its small bounded
            latent/mel histories are forked, so decoding a prediction tail can
            never advance the official decoder frontier.
            """

            def feed_variable(self, video, audio):
                if self.block_index == 0:
                    self.expected_first_video_latents = int(video.shape[2])
                    self.expected_first_audio_latents = int(audio.shape[2])
                else:
                    self.expected_video_latents = int(video.shape[2])
                    self.expected_audio_latents = int(audio.shape[2])
                return self.feed(video, audio)

            def fork(self):
                forked = type(self)(
                    self.video_decoder, self.audio_decoder, self.vocoder,
                    video_history_latents=self.video_history_latents,
                    audio_history_latents=self.audio_history_latents,
                    vocoder_history_mel=self.vocoder_history_mel,
                    value_ramp_ms=self.value_ramp_ms,
                    expected_first_video_latents=self.expected_first_video_latents,
                    expected_video_latents=self.expected_video_latents,
                    expected_first_audio_latents=self.expected_first_audio_latents,
                    expected_audio_latents=self.expected_audio_latents,
                    verbose=self.verbose,
                )
                # feed() replaces these bounded tensors rather than mutating
                # them, so sharing immutable snapshots is sufficient.
                forked._video_history = self._video_history
                forked._audio_history = self._audio_history
                forked._mel_history = self._mel_history
                forked._previous_wave_sample = self._previous_wave_sample
                forked.block_index = self.block_index
                forked.events = list(self.events)
                return forked

        self.streaming_decoder_class = ForkableStreamingDecoder
        self.speculative_sampling_sessions = {}
        self.speculative_sampling_decoders = {}
        self.speculative_sampling_prompt_timelines = {}

        if self.split_services:
            self.server = None
            return
        scripts = Path(args.vforge_repo).resolve().parent / "frame_wise" / "scripts"
        if not (scripts / "ltx_online_serving.py").exists():
            scripts = Path(args.online_serving_dir).resolve()
        sys.path.insert(0, str(scripts))
        from ltx_online_serving import LtxOnlineServer

        self.server = LtxOnlineServer(
            model.generator,
            kv_cache_blocks=(args.kv_cache_blocks if args.rollout == "upstream-rerope"
                             else int(cfg.model.extras.get("kv_cache_blocks", 0))),
            num_sink_blocks=(args.num_sink_blocks if args.rollout == "upstream-rerope"
                             else int(cfg.model.extras.get("num_sink_blocks", 0))),
            rerope=args.rollout == "upstream-rerope",
        )

    def warmup(self):
        # Keep Gemma resident only on rank 0. Other ranks hold only their DiT
        # USP shard; prompt contexts are broadcast after each rank-0 encode.
        dist = self.torch.distributed
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.split_services:
            error, warm_context = None, None
            if rank == 0:
                deadline = time.monotonic() + min(
                    120., float(getattr(self.args, "worker_recovery_timeout", 600)))
                while True:
                    try:
                        self.media_client.call("health")
                        warm_context = self.text_client.call(
                            "encode", prompt="warmup")["context"]
                        break
                    except Exception as exc:
                        if time.monotonic() >= deadline:
                            error = str(exc)
                            break
                        # Dedicated workers load concurrently and the media
                        # HTTP listener starts immediately after its NCCL
                        # rendezvous. Treat connection refusal as startup, not
                        # a fatal one-shot dependency probe.
                        time.sleep(.25)
            payload = [warm_context, error]
            if dist.is_initialized():
                dist.broadcast_object_list(payload, src=0, device=self.runtime.device)
            if payload[1]:
                raise RuntimeError(payload[1])
            self.prompt_contexts["warmup"] = payload[0]
            return
        warm_context = None
        if rank == 0:
            from vforge.models.ltx.text_encoder import PromptEncoder

            self.prompt_encoder = PromptEncoder(
                self.cfg.model.pretrained_path,
                self.cfg.model.extras["text_encoder_path"],
                device=self.runtime.device,
                # Henry conditions were produced with Gemma K=1024, then
                # stripped and independently connected at J=256 per segment.
                max_length=1024,
            )
            # Compile/warm Gemma while the service still reports not-ready;
            # otherwise the first browser prompt pays ~31s despite later
            # unseen prompts taking ~0.1s on the same rank-0 B300.
            warm_context = self.prompt_encoder.encode("warmup")
            warm_context = {key: value.cpu() for key, value in warm_context.items()}
        if dist.is_initialized():
            payload = [warm_context]
            dist.broadcast_object_list(payload, src=0, device=self.runtime.device)
            warm_context = payload[0]
        self.prompt_contexts["warmup"] = warm_context

        # prepare_inputs participates in USP collectives, so every rank must
        # build the warmup batch even though only rank 0 owns the decoders.
        warm_batch = None
        if self.args.compile_decoders:
            # _batch adds the single reference latent; request exactly two
            # generated native chunks without a trailing partial block.
            warm_batch = self._batch("warmup", 2 * self.args.chunk_latents)

        if rank == 0:
            # VAE/vocoder are also rank-0-only and resident before the service
            # advertises ready; the first interactive request must not pay a
            # hidden CPU->GPU transfer or decoder graph compilation penalty.
            self.decoder._load()
            self.decoder._models = tuple(
                model.to(self.runtime.device).eval().requires_grad_(False)
                for model in self.decoder._models
            )
            if self.args.compile_decoders:
                self.decoder._models = tuple(
                    self.torch.compile(
                        model,
                        mode="default",
                        fullgraph=False,
                        options={"triton.cudagraphs": False},
                    )
                    for model in self.decoder._models
                )
                self._warm_streaming_decoders(warm_batch)
        # Avoid consuming a proposal here so the first browser request keeps
        # its requested seed. The outer barrier publishes ready only after the
        # rank-0 text encoder is fully resident.
        self.torch.cuda.synchronize(self.runtime.device)

    def _warm_streaming_decoders(self, batch):
        """Compile both first-block and steady K45 streaming decode shapes."""
        torch = self.torch
        layout = batch.model_inputs.layout
        if len(layout.audio_ranges) < 2:
            raise RuntimeError("decoder warmup requires two native K45 blocks")
        video_decoder, audio_decoder, vocoder = self.decoder._models
        first_audio = layout.audio_ranges[0][1] - layout.audio_ranges[0][0]
        steady_audio = layout.audio_ranges[1][1] - layout.audio_ranges[1][0]
        stream = self.streaming_decoder_class(
            video_decoder, audio_decoder, vocoder,
            expected_first_video_latents=self.args.chunk_latents,
            expected_video_latents=self.args.chunk_latents,
            expected_first_audio_latents=first_audio,
            expected_audio_latents=steady_audio,
        )
        dtype = next(video_decoder.parameters()).dtype
        video = torch.zeros(
            1, 128, self.args.chunk_latents, layout.height, layout.width,
            device=self.runtime.device, dtype=dtype,
        )
        audio_first = torch.zeros(
            1, 8, first_audio, 16, device=self.runtime.device,
            dtype=next(audio_decoder.parameters()).dtype,
        )
        audio_steady = torch.zeros(
            1, 8, steady_audio, 16, device=self.runtime.device,
            dtype=next(audio_decoder.parameters()).dtype,
        )
        first = stream.feed(video, audio_first)
        steady = stream.feed(video, audio_steady)
        del first, steady, stream, video, audio_first, audio_steady, batch
        torch.cuda.synchronize(self.runtime.device)

    def _encode_reference(self, path, *, width, height):
        # Reuse the upstream image preprocessing and native tiled VAE encoder.
        # Do not substitute the dataset's first video latent for an upload.
        root = Path(self.args.interactive_streaming_repo).resolve()
        for source in (root / "src", *(root / "packages" / name / "src" for name in (
            "ltx-core", "ltx-pipelines", "ltx-tf", "ltx-trainer", "ltx-dmd",
        ))):
            if str(source) not in sys.path:
                sys.path.append(str(source))
        from ltx_interactive.cli import encode_reference_image

        latent, _image = encode_reference_image(
            path, self.cfg.model.pretrained_path, width, height, self.runtime.device
        )
        return latent

    def _reference_sample(self, task):
        """Bind the actual image once per session, identically on all USP ranks."""
        path = str(Path(task["reference"]).expanduser().resolve())
        key = (task["session"], path)
        if self.reference_key == key:
            return self.reference_sample
        if self.reference_key is not None and self.reference_key[0] == key[0]:
            raise ValueError("reference cannot change inside an active LTX session")

        height, width = self.sample["video_latents"]["latents"].shape[-2:]
        dist = self.torch.distributed
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Broadcast errors too: a failed rank-0 image load must not leave peers
        # waiting for a latent that will never be sent.
        payload = [None, None]
        if rank == 0:
            try:
                if getattr(self, "split_services", False):
                    latent = self._media_reference(
                        task, path, width * 32, height * 32)
                else:
                    latent = self._encode_reference(Path(path), width=width * 32, height=height * 32)
                expected = (1, 128, 1, height, width)
                if tuple(latent.shape) != expected:
                    raise ValueError(f"reference latent shape {tuple(latent.shape)} != {expected}")
                payload[0] = latent.detach().cpu()
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            dist.broadcast_object_list(payload, src=0, device=self.runtime.device)
        if payload[1] is not None:
            raise RuntimeError(f"LTX reference encoding failed: {payload[1]}")

        # Preserve only the template's non-image metadata/conditioning. The
        # single-frame encoded upload supplies both geometry and reference.
        sample = {
            **self.sample,
            "video_latents": {**self.sample["video_latents"], "latents": payload[0]},
        }
        self.reference_key, self.reference_sample = key, sample
        return sample

    def _batch(self, prompt, latent_frames, *, sample=None):
        context = self.prompt_contexts.get(prompt)
        if context is None:
            dist = self.torch.distributed
            rank = dist.get_rank() if dist.is_initialized() else 0
            error = None
            if rank == 0:
                try:
                    if getattr(self, "split_services", False):
                        context = self.text_client.call("encode", prompt=prompt)["context"]
                    else:
                        if self.prompt_encoder is None:
                            raise RuntimeError("rank-0 resident prompt encoder is not initialized")
                        context = self.prompt_encoder.encode_timed_segment(
                            prompt,
                            segment_length=int(self.cfg.model.extras.get(
                                "timed_prompt_segment_length", 256
                            )),
                        )
                        context = {key: value.cpu() for key, value in context.items()}
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            payload = [context, error]
            if dist.is_initialized():
                dist.broadcast_object_list(payload, src=0, device=self.runtime.device)
            if payload[1]:
                raise RuntimeError(f"text condition failed: {payload[1]}")
            context = payload[0]
            self.prompt_contexts[prompt] = context
        prefix = int(self.cfg.model.extras.get("algorithm_variant") == "ref2av_hardmask_0914")
        batch = self.inference.prepare_inputs(
            self.sample if sample is None else sample, latent_frames=prefix + latent_frames
        )
        inputs = replace(
            batch.model_inputs,
            video_context=context["video_context"].to(
                device=self.runtime.device, dtype=self.torch.bfloat16
            ),
            audio_context=context["audio_context"].to(
                device=self.runtime.device, dtype=self.torch.bfloat16
            ),
            context_positions=context.get("context_positions"),
            prompt_attention_mask=context.get("prompt_attention_mask"),
        )
        return replace(batch, model_inputs=inputs)

    def _resolve_condition_handle(self, task, handle):
        """Resolve or explicitly reconstruct one text handle after restart."""
        from speculative_sampling.runtime.rpc import RemoteWorkerRestarted

        if (handle.get("owner") != "ltx_text"
                or handle.get("condition_version") != task["prompt_version"]):
            raise ValueError("condition handle owner/version mismatch")
        for name in ("epoch", "key", "fingerprint"):
            if not isinstance(handle.get(name), str) or not handle[name]:
                raise ValueError(f"condition handle has invalid {name}")
        old_epoch = handle["epoch"]
        recovery_started = None

        def rehydrate(reason):
            nonlocal recovery_started
            if recovery_started is None:
                recovery_started = time.monotonic()
            try:
                encoded = self.text_client.call("encode_handle", prompt=task["prompt"])
            except RemoteWorkerRestarted as restart:
                # The coordinator may already have encoded this task on the
                # replacement text worker while the distributed DiT process
                # still has the previous epoch pinned.  Adopt only the epoch
                # named by the journaled handle, then reconstruct and verify
                # the content-addressed condition below.
                if restart.observed_epoch != handle["epoch"]:
                    raise RuntimeError(
                        "text worker restarted again after condition was journaled") from restart
                self.text_client.accept_restart(restart)
                encoded = self.text_client.call("encode_handle", prompt=task["prompt"])
            if (encoded.get("key") != handle["key"]
                    or encoded.get("fingerprint") != handle["fingerprint"]):
                raise RuntimeError("re-encoded condition differs from journaled handle")
            context = self.text_client.call("resolve", key=encoded["key"])
            self.condition_recoveries.append({
                "session": task["session"],
                "condition_version": int(task["prompt_version"]),
                "key": handle["key"],
                "fingerprint": handle["fingerprint"],
                "previous_epoch": old_epoch,
                "recovered_epoch": self.text_client.epoch,
                "reason": reason,
                "recovery_seconds": time.monotonic() - recovery_started,
            })
            return context

        if self.text_client.epoch not in (None, old_epoch):
            return rehydrate("client_already_observed_new_epoch")
        try:
            context = self.text_client.call("resolve", key=handle["key"])
        except RemoteWorkerRestarted as restart:
            recovery_started = time.monotonic()
            self.text_client.accept_restart(restart)
            return rehydrate("remote_restart_observed_during_resolve")
        except RuntimeError:
            # A newly constructed DiT client can learn the current epoch from
            # an expired-handle response. Rebuild only when that response also
            # proves that the journaled epoch is no longer current.
            if self.text_client.epoch != old_epoch:
                recovery_started = time.monotonic()
                return rehydrate("handle_expired_on_current_epoch")
            raise
        if self.text_client.epoch != old_epoch:
            return rehydrate("initial_client_discovered_new_epoch")
        return context

    def _restore_media_epoch(self, task, restart):
        """Adopt a media epoch and restore the last coordinator-accepted state."""
        self.media_client.accept_restart(restart)
        restored = None
        try:
            checkpoint = self.media_journal.load(task["session"])
        except FileNotFoundError:
            checkpoint = None
        if checkpoint is not None:
            restored = self.media_client.call(
                "restore_checkpoint", checkpoint=checkpoint)
            if restored.get("restored") is not True:
                raise RuntimeError("media worker did not confirm checkpoint restore")
        event = {
            "session": task["session"],
            "previous_epoch": restart.previous_epoch,
            "recovered_epoch": restart.observed_epoch,
            "checkpoint": restored,
        }
        self.media_recoveries.append(event)
        return event

    def _media_transaction(self, task, video, audio):
        """Retry one content-addressed install/checkpoint transaction."""
        from speculative_sampling.runtime.rpc import RemoteCallError, RemoteWorkerRestarted

        deadline = time.monotonic() + float(
            getattr(self.args, "worker_recovery_timeout", 600))
        recovery_started = None
        while True:
            try:
                self.media_client.call(
                    "prepare", session=task["session"],
                    retain_until=int(task["retain_until"]))
                tensor_channel = getattr(self, "media_tensor_channel", None)
                if tensor_channel is not None:
                    import threading
                    descriptors = tensor_channel.describe(video, audio)
                    remote = {}

                    def receive_remote():
                        try:
                            remote["install"] = self.media_client.call(
                                "install_nccl", session=task["session"],
                                proposal_id=task["proposal"],
                                start=int(task["retain_until"]),
                                descriptors=descriptors)
                        except BaseException as exc:
                            remote["error"] = exc

                    # The receiver must enter NCCL recv before the sender's
                    # first lazy communicator initialization can complete.
                    receiver = threading.Thread(
                        target=receive_remote, name="ltx-media-nccl-recv")
                    receiver.start()
                    _descriptors, pending = tensor_channel.send(video, audio)
                    tensor_channel.wait(pending)
                    receiver.join(timeout=float(
                        getattr(self.args, "worker_recovery_timeout", 600)))
                    if receiver.is_alive():
                        raise TimeoutError("media NCCL receive RPC did not finish")
                    if "error" in remote:
                        raise remote["error"]
                    install = remote["install"]
                    # Keep the coordinator journal schema intact while making
                    # clear that this development path has no durable media
                    # checkpoint and must fail the session on worker loss.
                    checkpoint = {
                        "checkpoint_fingerprint": hashlib.sha256(
                            ("volatile:" + install["content_fingerprint"]).encode()
                        ).hexdigest(),
                        "volatile": True,
                    }
                else:
                    install = self.media_client.call(
                        "install", session=task["session"], proposal_id=task["proposal"],
                        start=int(task["retain_until"]), video=video, audio=audio)
                    checkpoint = self.media_client.call(
                        "checkpoint", session=task["session"])
                if recovery_started is not None:
                    self.media_recoveries.append({
                        "session": task["session"],
                        "transaction_recovered": True,
                        "recovery_seconds": time.monotonic() - recovery_started,
                    })
                return install, checkpoint
            except RemoteWorkerRestarted as restart:
                if recovery_started is None:
                    recovery_started = time.monotonic()
                self._restore_media_epoch(task, restart)
            except RemoteCallError:
                raise
            except Exception as exc:
                if recovery_started is None:
                    recovery_started = time.monotonic()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "media worker did not recover during latent install") from exc
                time.sleep(.5)

    def _media_reference(self, task, path, width, height):
        """Acquire one immutable reference across an availability transition."""
        from speculative_sampling.runtime.rpc import RemoteCallError, RemoteWorkerRestarted

        deadline = time.monotonic() + float(
            getattr(self.args, "worker_recovery_timeout", 600))
        recovery_started = None
        while True:
            try:
                if self.reference_key is not None:
                    self.media_client.call("close", session=self.reference_key[0])
                result = self.media_client.call(
                    "encode_reference", session=task["session"], path=path,
                    width=width, height=height)
                if recovery_started is not None:
                    self.media_recoveries.append({
                        "session": task["session"],
                        "reference_recovered": True,
                        "recovery_seconds": time.monotonic() - recovery_started,
                    })
                return result
            except RemoteWorkerRestarted as restart:
                if recovery_started is None:
                    recovery_started = time.monotonic()
                self._restore_media_epoch(task, restart)
            except RemoteCallError:
                raise
            except Exception as exc:
                if recovery_started is None:
                    recovery_started = time.monotonic()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "media worker did not recover during reference acquisition") from exc
                time.sleep(.5)

    def _speculative_sampling_batch(self, task, latent_frames):
        """Build the native full-length MTSS condition at this safe point."""
        handle = task.get("condition_handle")
        if handle is not None:
            dist = self.torch.distributed
            rank = dist.get_rank() if dist.is_initialized() else 0
            payload = [None, None]
            if rank == 0:
                try:
                    payload[0] = self._resolve_condition_handle(task, handle)
                except Exception as exc:
                    payload[1] = f"{type(exc).__name__}: {exc}"
            if dist.is_initialized():
                dist.broadcast_object_list(payload, src=0, device=self.runtime.device)
            if payload[1]:
                raise RuntimeError(f"condition resolution failed: {payload[1]}")
            self.prompt_contexts[task["prompt"]] = payload[0]
        batch = self._batch(task["prompt"], latent_frames, sample=self._reference_sample(task))
        inputs = batch.model_inputs
        timeline = self.speculative_sampling_prompt_timelines.setdefault(task["session"], [])
        version = int(task["prompt_version"])
        if not timeline or timeline[-1]["version"] != version:
            video_start = (inputs.layout.prefix_frames + int(task["retain_until"])) * inputs.layout.frame_tokens
            if video_start >= inputs.video_positions.shape[2]:
                raise ValueError("safe point exceeds MTSS position capacity")
            activation = (
                0.0
                if int(task["retain_until"]) == 0
                else float(inputs.video_positions[0, 0, video_start].float().mean().item())
            )
            encoded = self.prompt_contexts[task["prompt"]]
            timeline.append({
                "version": version,
                "prompt": task["prompt"],
                "activation": activation,
                "video_context": encoded["video_context"],
                "audio_context": encoded["audio_context"],
            })

        video_parts = []
        audio_parts = []
        position_parts = []
        for index, segment in enumerate(timeline):
            video = segment["video_context"]
            audio = segment["audio_context"]
            if video.shape[1] != audio.shape[1]:
                raise ValueError("MTSS video/audio connector lengths must match")
            start = segment["activation"]
            end = timeline[index + 1]["activation"] if index + 1 < len(timeline) else 1_000_000.0
            video_parts.append(video)
            audio_parts.append(audio)
            position_parts.append(self.torch.tensor(
                (start, end), dtype=self.torch.float32
            ).view(1, 1, 2).expand(1, video.shape[1], 2))

        video_context = self.torch.cat(video_parts, dim=1).to(
            device=self.runtime.device, dtype=self.torch.bfloat16
        )
        audio_context = self.torch.cat(audio_parts, dim=1).to(
            device=self.runtime.device, dtype=self.torch.bfloat16
        )
        context_positions = self.torch.cat(position_parts, dim=1).to(self.runtime.device)
        attention_mask = self.torch.ones(
            context_positions.shape[:2], dtype=self.torch.int64, device=self.runtime.device
        )
        return replace(batch, model_inputs=replace(
            inputs,
            video_context=video_context,
            audio_context=audio_context,
            context_positions=context_positions,
            prompt_attention_mask=attention_mask,
        ))

    def generate(self, task, on_block, prompt_source=None):
        if self.args.controller == "speculative-sampling":
            return self._generate_speculative_sampling(task, on_block)
        from speculative_sampling.runtime.media import PreviewBuilder
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.types import AudioLatentShape, VideoLatentShape

        started = time.perf_counter()
        duration = int(task["generation"]["duration"])
        if self.args.rollout == "upstream-rerope" and self.args.chunk_latents == 45:
            latent_frames = self.args.chunk_latents * max(
                1, self.args.rerope_session_seconds // 15
            )
        else:
            latent_frames = {5: 15, 10: 30, 15: 45}[duration]
        batch = self._batch(task["prompt"], latent_frames, sample=self._reference_sample(task))
        encoded_at = time.perf_counter()
        session = self.server.open_session(
            batch,
            seed=int(task["seed"]),
            steps=self.cfg.model.extras["denoising_steps"],
            mode=task["serving_policy"],
            prompt_to_batch=self._replace_prompt,
        )
        rank = self.torch.distributed.get_rank() if self.torch.distributed.is_initialized() else 0
        builder = stream_decoder = preview = None
        video_cursor = audio_cursor = 0
        denoise_s = commit_s = 0.0
        prompt_version = int(task.get("prompt_version", 1))
        try:
            while True:
                update = None
                if rank == 0 and prompt_source is not None:
                    snapshot = prompt_source()
                    if snapshot and snapshot["prompts"]:
                        latest = max(int(value) for value in snapshot["prompts"])
                        if latest > prompt_version:
                            update = (latest, snapshot["prompts"][latest])
                if self.torch.distributed.is_initialized():
                    payload = [update]
                    self.torch.distributed.broadcast_object_list(payload, src=0)
                    update = payload[0]
                if update is not None:
                    prompt_version, prompt = update
                    session.submit_prompt(prompt)
                ready = session.step()
                if ready is None:
                    break
                denoise_s += ready.denoise_seconds
                commit_s += ready.commit_seconds
                if rank != 0:
                    continue
                if self.decoder._models is None:
                    self.decoder._load()
                video_decoder, audio_decoder, vocoder = self.decoder._models
                for model in self.decoder._models:
                    model.to(self.runtime.device)
                video_frames = ready.values["video"].shape[1] // (batch.model_inputs.layout.height * batch.model_inputs.layout.width)
                audio_range = batch.model_inputs.layout.audio_ranges[ready.index]
                audio_frames = int(audio_range[1] - audio_range[0])
                if stream_decoder is None:
                    stream_decoder = self.streaming_decoder_class(
                        video_decoder, audio_decoder, vocoder,
                        expected_first_video_latents=video_frames,
                        expected_video_latents=video_frames,
                        expected_first_audio_latents=audio_frames,
                        expected_audio_latents=(
                            int(batch.model_inputs.layout.audio_ranges[1][1]
                                - batch.model_inputs.layout.audio_ranges[1][0])
                            if len(batch.model_inputs.layout.audio_ranges) > 1
                            else audio_frames
                        ),
                    )
                    builder = PreviewBuilder(
                        task["output"],
                        width=self.args.rtc_width if task["transport"] == "rtc" else self.args.preview_width,
                        transport=task["transport"], proposal_frames=task["proposal_frames"],
                    )
                    silence = self.torch.zeros(
                        2, task["proposal_frames"] * stream_decoder.output_sample_rate // 24,
                        dtype=self.torch.float32,
                    )
                    builder.audio(silence, stream_decoder.output_sample_rate)
                video_latent = VideoLatentPatchifier(patch_size=1).unpatchify(
                    ready.values["video"],
                    output_shape=VideoLatentShape(
                        batch=1, channels=128, frames=video_frames,
                        height=batch.model_inputs.layout.height,
                        width=batch.model_inputs.layout.width,
                    ),
                )
                audio_latent = AudioPatchifier(patch_size=1).unpatchify(
                    ready.values["audio"],
                    output_shape=AudioLatentShape(
                        batch=1, channels=8, frames=audio_frames, mel_bins=16,
                    ),
                )
                video_latent = video_latent.to(dtype=next(video_decoder.parameters()).dtype)
                audio_latent = audio_latent.to(dtype=next(audio_decoder.parameters()).dtype)
                emitted_video, emitted_audio, _ = stream_decoder.feed(video_latent, audio_latent)
                video = ((emitted_video + 1) / 2).clamp(0, 1)[0].permute(1, 0, 2, 3)
                builder.audio_chunk(audio_cursor, emitted_audio[0], stream_decoder.output_sample_rate)
                audio_cursor += emitted_audio.shape[-1]
                preview = builder.video(video_cursor, video)
                video_cursor += video.shape[0]
                if on_block:
                    on_block(video_cursor, preview)
        finally:
            session.close()
        self.torch.cuda.synchronize(self.runtime.device)
        generated_at = time.perf_counter()
        if rank != 0:
            return None, {}, generated_at - encoded_at
        decoded_at = generated_at
        try:
            if builder is None or video_cursor != task["proposal_frames"]:
                raise RuntimeError(
                    f"streaming decoder produced {video_cursor}/{task['proposal_frames']} frames"
                )
            preview = builder.finish()
        except Exception:
            builder.close()
            raise
        finished = time.perf_counter()
        stages = {
            "prompt_encode": {"wall_s": encoded_at - started},
            "dit": {"wall_s": generated_at - encoded_at},
            "vae_preview": {"wall_s": finished - generated_at},
            "denoise": {"wall_s": denoise_s},
            "commit": {"wall_s": commit_s},
            "decode": {"wall_s": decoded_at - generated_at},
        }
        return preview, stages, generated_at - encoded_at

    def _generate_speculative_sampling(self, task, on_block):
        """Execute one controller command against a persistent native session."""
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.types import AudioLatentShape, VideoLatentShape
        from speculative_sampling.models.ltx.speculative_sampling_native import NativeSpeculativeSamplingSession

        started = time.perf_counter()
        # Position metadata covers several native K commands, while the
        # resident ReRoPE cache itself still owns exactly one fixed sink and
        # two rolling blocks. This is position capacity, not cache capacity.
        requested_capacity = max(
            10 * self.args.chunk_latents,
            3 * self.args.rerope_session_seconds,
            int(task["retain_until"]) + self.args.chunk_latents,
        )
        capacity = (
            (requested_capacity + self.args.chunk_latents - 1)
            // self.args.chunk_latents
            * self.args.chunk_latents
        )
        # ``_batch`` adds the single ref2av reference latent itself.  Passing
        # ``capacity + 1`` here creates a trailing one-latent physical block,
        # which is not the checkpoint's native K layout.
        recovery_start = len(self.condition_recoveries)
        media_recovery_start = len(self.media_recoveries)
        batch = self._speculative_sampling_batch(task, capacity)
        condition_audit = None
        if getattr(self.args, "audit_condition", False):
            from speculative_sampling.models.ltx.condition_audit import audit_condition
            condition_audit = audit_condition(
                batch, task, self.speculative_sampling_prompt_timelines[task["session"]],
                self.cfg.model.extras, include_fingerprints=True)
        encoded_at = time.perf_counter()
        native = self.speculative_sampling_sessions.get(task["session"])
        if native is None:
            for old in self.speculative_sampling_sessions.values():
                old.close()
            self.speculative_sampling_sessions.clear()
            self.speculative_sampling_decoders.clear()
            self.speculative_sampling_prompt_timelines = {
                task["session"]: self.speculative_sampling_prompt_timelines[task["session"]]
            }
            native = NativeSpeculativeSamplingSession(
                self.model.generator, batch, seed=int(task["seed"]),
                steps=self.cfg.model.extras["denoising_steps"],
                chunk_size=self.args.chunk_latents,
                kv_cache_blocks=self.args.kv_cache_blocks,
                num_sink_blocks=self.args.num_sink_blocks,
            )
            self.speculative_sampling_sessions[task["session"]] = native
        native.profile_cuda_stages = bool(getattr(self.args, "profile_cuda_stages", False))
        on_latents = None
        replay_content_fingerprint = None
        if self.split_services:
            dist = self.torch.distributed
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                def on_latents(proposal):
                    nonlocal replay_content_fingerprint
                    layout = batch.model_inputs.layout
                    video_tokens = self.torch.cat([p.values["video"] for p in proposal.predictions], 1)
                    audio_tokens = self.torch.cat([p.values["audio"] for p in proposal.predictions], 1)
                    video = video_tokens.reshape(
                        1, self.args.chunk_latents, layout.height, layout.width, 128
                    ).permute(0, 4, 1, 2, 3).contiguous()
                    audio = audio_tokens.reshape(1, -1, 8, 16).permute(0, 2, 1, 3).contiguous()
                    if getattr(self, "media_tensor_channel", None) is None:
                        video = video.cpu()
                        audio = audio.cpu()
                    if task.get("_recovery_replay"):
                        digest = hashlib.sha256()
                        start = int(task["retain_until"])
                        for name, value in ((f"video:{start}", video),
                                            (f"audio:{start}", audio)):
                            value = value.detach().cpu().contiguous()
                            digest.update(name.encode())
                            digest.update(str((value.dtype, tuple(value.shape))).encode())
                            digest.update(value.view(self.torch.uint8).numpy().tobytes())
                        replay_content_fingerprint = digest.hexdigest()
                        return
                    install, media_checkpoint = self._media_transaction(
                        task, video, audio)
                    latent_event = {
                        "event": "latents_ready", "task": task,
                        "media_install": install,
                        "media_checkpoint": media_checkpoint,
                    }
                    # This callback must run before the sole clean-prefill.
                    # External DiT service deployments forward it immediately
                    # to the coordinator; in-process deployments apply it
                    # directly.  MediaPump owns all decode RPCs.
                    if on_block:
                        on_block(latent_event, None)
        proposal = native.advance(
            retain_until=int(task["retain_until"]), condition_batch=batch,
            condition_version=int(task["prompt_version"]),
            on_latents=on_latents, prepare_full=self.split_services,
        )
        dist = self.torch.distributed
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.torch.cuda.synchronize(self.runtime.device)
        generated_at = time.perf_counter()
        cuda_stages = None
        if native.profile_cuda_stages:
            rank_stages = [proposal.cuda_stage_seconds]
            if dist.is_initialized():
                rank_stages = [None] * dist.get_world_size()
                dist.all_gather_object(rank_stages, proposal.cuda_stage_seconds)
            cuda_stages = {
                "clock_domain": "cuda_event_elapsed_seconds",
                "stream": "current_compute_stream", "ranks": rank_stages,
                "max_rank_seconds": {
                    name: max(record.get(name, 0.) for record in rank_stages)
                    for name in ("prefix_prepare", "denoise", "clean_prefill")
                },
                "note": "Stream elapsed time includes host launch gaps; not summed kernel duration.",
            }
        if rank != 0:
            return None, {}, generated_at - encoded_at
        if self.split_services:
            stages = {
                "condition_prefix_handoff": {"wall_s": encoded_at - started},
                "dit": {"wall_s": proposal.denoise_seconds, "device": f"usp{self.args.usp_size}"},
                "prefix_prepare": {"wall_s": proposal.commit_seconds},
                "clean_prefill": {"wall_s": proposal.clean_prefill_seconds},
            }
            if condition_audit is not None:
                stages["actual_condition"] = condition_audit
            if len(self.condition_recoveries) > recovery_start:
                stages["condition_recovery"] = {
                    "events": self.condition_recoveries[recovery_start:]
                }
            if len(self.media_recoveries) > media_recovery_start:
                events = self.media_recoveries[media_recovery_start:]
                stages["media_recovery"] = {
                    "events": events
                }
                stages["dependency_recovery_seconds"] = sum(
                    float(event.get("recovery_seconds", 0.)) for event in events)
            if len(self.condition_recoveries) > recovery_start:
                stages["dependency_recovery_seconds"] = (
                    stages.get("dependency_recovery_seconds", 0.) + sum(
                        float(event.get("recovery_seconds", 0.))
                        for event in self.condition_recoveries[recovery_start:]))
            if task.get("_recovery_replay"):
                if replay_content_fingerprint is None:
                    raise RuntimeError("recovery replay did not fingerprint latent output")
                stages["recovery_replay"] = {
                    "media_content_fingerprint": replay_content_fingerprint,
                    "media_side_effects": False,
                }
            if cuda_stages is not None:
                stages["cuda_events"] = cuda_stages
            return None, stages, generated_at - encoded_at

        from speculative_sampling.runtime.media import PreviewBuilder
        if self.decoder._models is None:
            self.decoder._load()
        video_decoder, audio_decoder, vocoder = self.decoder._models
        for model in self.decoder._models:
            model.to(self.runtime.device)

        def unpack(items):
            video_values = self.torch.cat([item.values["video"] for item in items], dim=1)
            audio_values = self.torch.cat([item.values["audio"] for item in items], dim=1)
            video_frames = len(items)
            audio_frames = audio_values.shape[1]
            video_latent = VideoLatentPatchifier(patch_size=1).unpatchify(
                video_values,
                output_shape=VideoLatentShape(
                    batch=1, channels=128, frames=video_frames,
                    height=batch.model_inputs.layout.height,
                    width=batch.model_inputs.layout.width,
                ),
            ).to(dtype=next(video_decoder.parameters()).dtype)
            audio_latent = AudioPatchifier(patch_size=1).unpatchify(
                audio_values,
                output_shape=AudioLatentShape(
                    batch=1, channels=8, frames=audio_frames, mel_bins=16,
                ),
            ).to(dtype=next(audio_decoder.parameters()).dtype)
            return video_latent, audio_latent

        video_latent, audio_latent = unpack(proposal.predictions)
        official_decoder = self.speculative_sampling_decoders.get(task["session"])
        if official_decoder is None:
            official_decoder = self.streaming_decoder_class(
                video_decoder, audio_decoder, vocoder,
                expected_first_video_latents=self.args.chunk_latents,
                expected_video_latents=self.args.chunk_latents,
                expected_first_audio_latents=audio_latent.shape[2],
                expected_audio_latents=audio_latent.shape[2],
            )
            self.speculative_sampling_decoders[task["session"]] = official_decoder

        # Advance the official decoder only across the generator predictions
        # that SpeculativeSampling accepted at this exact safe point.  The new K45 tail
        # is decoded on a fork and may be discarded without contaminating the
        # next continuation.
        commit_decode_started = time.perf_counter()
        if proposal.committed:
            accepted_video, accepted_audio = unpack(proposal.committed)
            official_decoder.feed_variable(accepted_video, accepted_audio)
        commit_decode_seconds = time.perf_counter() - commit_decode_started
        stream_decoder = official_decoder.fork()
        emitted_video, emitted_audio, _event = stream_decoder.feed_variable(
            video_latent, audio_latent
        )
        video = ((emitted_video + 1) / 2).clamp(0, 1)[0].permute(1, 0, 2, 3)
        builder = PreviewBuilder(
            task["output"],
            width=self.args.rtc_width if task["transport"] == "rtc" else self.args.preview_width,
            transport=task["transport"], proposal_frames=task["proposal_frames"],
        )
        try:
            builder.audio(emitted_audio[0], stream_decoder.output_sample_rate)
            preview = builder.video(0, video)
            if video.shape[0] != task["proposal_frames"]:
                raise RuntimeError(
                    f"streaming decoder produced {video.shape[0]}/{task['proposal_frames']} frames"
                )
            if on_block:
                on_block(video.shape[0], preview)
            preview = builder.finish()
        except Exception:
            builder.close()
            raise
        finished = time.perf_counter()
        stages = {
            "prompt_encode": {"wall_s": encoded_at - started, "device": "rank0_gpu"},
            "dit": {"wall_s": proposal.denoise_seconds, "device": f"usp{self.args.usp_size}"},
            "commit": {"wall_s": proposal.commit_seconds, "retain_until": task["retain_until"]},
            "commit_decode": {"wall_s": commit_decode_seconds, "device": "rank0_gpu"},
            "vae_preview": {"wall_s": finished - generated_at, "device": "rank0_gpu"},
        }
        return preview, stages, generated_at - encoded_at

    def _replace_prompt(self, old_batch, prompt):
        """Replace only text conditioning; preserve layout/reference/cache."""
        prefix = int(self.cfg.model.extras.get("algorithm_variant") == "ref2av_hardmask_0914")
        updated = self._batch(prompt, old_batch.model_inputs.layout.frames - prefix)
        old = old_batch.model_inputs
        new = updated.model_inputs
        return replace(old_batch, model_inputs=replace(
            old,
            video_context=new.video_context,
            audio_context=new.audio_context,
            context_positions=new.context_positions,
            prompt_attention_mask=new.prompt_attention_mask,
        ))

    def close(self):
        from vforge.distributed.runtime import destroy_runtime

        if self.prompt_encoder is not None:
            self.prompt_encoder.close()
            self.prompt_encoder = None
        for session in self.speculative_sampling_sessions.values():
            session.close()
        self.speculative_sampling_sessions.clear()
        self.speculative_sampling_decoders.clear()
        self.speculative_sampling_prompt_timelines.clear()
        self.reference_sample = None
        self.reference_key = None
        destroy_runtime()


def create_worker(args):
    worker = VForgeLtxWorker(args)
    worker.args = args
    return worker
