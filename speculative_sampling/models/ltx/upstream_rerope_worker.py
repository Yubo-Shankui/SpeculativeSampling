"""Resident adapter for the upstream LTXInteractiveStreaming ReRoPE runtime."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


class UpstreamReropeWorker:
    model_id = "ltx23_dmd_mtss_rerope"
    mock = False

    def __init__(self, args):
        if not args.interactive_streaming_repo or not args.upstream_config or not args.upstream_checkpoint:
            raise ValueError("upstream ReRoPE requires repo, config, and checkpoint")
        root = Path(args.interactive_streaming_repo).resolve()
        for path in (
            root / "src",
            root / "packages/ltx-core/src",
            root / "packages/ltx-pipelines/src",
            root / "packages/ltx-tf/src",
            root / "packages/ltx-trainer/src",
            root / "packages/ltx-dmd/src",
        ):
            sys.path.insert(0, str(path))

        import torch
        import yaml
        from ltx_interactive import cli
        from ltx_trainer.model_loader import embedding_weight_paths

        self.args = args
        self.torch = torch
        self.cli = cli
        self.config_path = Path(args.upstream_config)
        with self.config_path.open() as handle:
            config = cli.DMDConfig(**yaml.safe_load(os.path.expandvars(handle.read())))
        config.kv_cache_blocks = 2
        config.num_sink_blocks = 1
        config.ref_follow = True
        self.config = config
        self.dit_device = torch.device(args.dit_device)
        self.prompt_device = torch.device(args.prompt_device)
        self.decode_device = torch.device(args.decode_device)

        # Load once and keep resident. run_session's global loader hooks return
        # these exact objects for every browser session.
        self.text_encoder = cli.load_text_encoder(
            config.text_encoder_path, device=self.prompt_device, dtype=cli.DTYPE
        )
        self.embeddings_processor = cli.load_embeddings_processor(
            checkpoint_path=embedding_weight_paths(config.model_path, config.text_encoder_path),
            gemma_model_path=config.text_encoder_path,
            device=self.prompt_device,
            dtype=cli.DTYPE,
        ).eval().requires_grad_(False)
        self.decoders = cli.load_decoders(config.model_path)
        for model in self.decoders:
            model.to(self.decode_device).eval().requires_grad_(False)
        self.student = cli.load_student(
            config.model_path, str(args.upstream_checkpoint), config.rote_mode
        ).eval().requires_grad_(False).to(self.dit_device)

        cli.load_text_encoder = lambda *unused_args, **unused_kwargs: self.text_encoder
        cli.load_embeddings_processor = (
            lambda *unused_args, **unused_kwargs: self.embeddings_processor
        )
        cli.load_decoders = lambda *unused_args, **unused_kwargs: self.decoders
        cli.load_student = lambda *unused_args, **unused_kwargs: self.student

    def warmup(self):
        for device in (self.dit_device, self.prompt_device, self.decode_device):
            self.torch.cuda.synchronize(device)

    def generate(self, task, on_block, prompt_source=None):
        from speculative_sampling.runtime.media import PreviewBuilder

        started = time.perf_counter()
        output_dir = Path(task["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = output_dir / "prompts.live.json"
        prompt_file.write_text(json.dumps({
            "prompts": [{"arrival_seconds": 0.0, "text": task["prompt"]}]
        }, ensure_ascii=False, indent=2))

        events = [self.cli.PromptEvent(index=0, arrival_seconds=0.0, text=task["prompt"])]
        events_lock = threading.Lock()
        stop_watcher = threading.Event()

        def watch_prompts():
            seen = 1
            last_arrival = 0.0
            while not stop_watcher.wait(0.05):
                snapshot = prompt_source() if prompt_source else None
                if not snapshot:
                    continue
                prompts = snapshot["prompts"]
                for version in sorted(prompts):
                    if int(version) <= seen:
                        continue
                    arrival = max(last_arrival + 0.001, float(snapshot["played_frame"]) / 24.0)
                    with events_lock:
                        events.append(self.cli.PromptEvent(
                            index=len(events), arrival_seconds=arrival,
                            text=str(prompts[version]),
                        ))
                    last_arrival = arrival
                    seen = int(version)

        watcher = threading.Thread(target=watch_prompts, daemon=True, name="ltx-live-prompts")
        watcher.start()
        original_loader = self.cli.load_prompt_events
        original_controller = self.cli.StreamingPromptController
        self.cli.load_prompt_events = lambda unused_path: events

        source_cursor = 0
        audio_cursor = 0
        builder = None

        def consume(chunk):
            nonlocal source_cursor, audio_cursor, builder
            if builder is None:
                builder = PreviewBuilder(
                    output_dir,
                    width=self.args.rtc_width if task["transport"] == "rtc" else self.args.preview_width,
                    transport=task["transport"],
                    proposal_frames=task["proposal_frames"],
                )
                silence = self.torch.zeros(
                    2, task["proposal_frames"] * chunk.audio_sample_rate // 24,
                    dtype=self.torch.float32,
                )
                builder.audio(silence, chunk.audio_sample_rate)
            builder.audio_chunk(audio_cursor, chunk.audio_samples, chunk.audio_sample_rate)
            audio_cursor += chunk.audio_samples.shape[-1]
            video = chunk.video_frames.permute(1, 0, 2, 3).contiguous()
            preview = builder.video(source_cursor, video)
            source_cursor += video.shape[0]
            if on_block:
                on_block(source_cursor, preview)

        run_args = SimpleNamespace(
            config=str(self.config_path),
            checkpoint=str(self.args.upstream_checkpoint),
            image=task["reference"],
            prompt_list=str(prompt_file),
            stop_seconds=float(self.args.rerope_session_seconds),
            max_prompt_seconds=float(self.args.rerope_session_seconds),
            positive_prompt_max_length=1024,
            condition_mode="streaming",
            prompt_switch_policy="next_chunk",
            output=str(output_dir / "upstream_stream.mp4"),
            width=832,
            height=448,
            fps=24.0,
            seed=int(task["seed"]),
            device=str(self.dit_device),
            dit_device=str(self.dit_device),
            prompt_device=str(self.prompt_device),
            decode_device=str(self.decode_device),
            kv_cache_blocks=2,
            num_sink_blocks=1,
            denoising_step_list=None,
            video_history_latents=2,
            audio_history_latents=32,
            vocoder_history_mel=40,
            audio_value_ramp_ms=1.0,
            precomputed_latent=None,
            precomputed_conditions=None,
            prompt_cycle_seconds=None,
            skip_offline=True,
            skip_stream_save=True,
            keep_models=True,
        )
        try:
            metadata = self.cli.run_session(run_args, external_chunk_consumer=consume)
        finally:
            stop_watcher.set()
            watcher.join(timeout=1)
            self.cli.load_prompt_events = original_loader
            self.cli.StreamingPromptController = original_controller
        if builder is None:
            raise RuntimeError("upstream ReRoPE emitted no media blocks")
        if source_cursor != task["proposal_frames"]:
            raise RuntimeError(
                f"upstream ReRoPE emitted {source_cursor}/{task['proposal_frames']} frames"
            )
        preview = builder.finish()
        elapsed = time.perf_counter() - started
        stages = {
            "upstream_rerope": {"wall_s": elapsed},
            "rollout": {"wall_s": metadata["rollout_seconds"]},
        }
        return preview, stages, elapsed

    def close(self):
        pass


def create_worker(args):
    return UpstreamReropeWorker(args)
