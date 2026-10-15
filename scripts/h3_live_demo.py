#!/usr/bin/env python3
"""MiniMax H3 model worker for the shared SpeculativeSampling live runtime.

Run with torchrun on the resident 8xB300 node. HTTP, WebSocket, RTC, tracing
and browser delivery are owned by :mod:`speculative_sampling.runtime`; this file owns
only H3 model loading, generation and boundary-image continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from http.server import ThreadingHTTPServer

from PIL import Image

from speculative_sampling.models.h3 import boundary_image_shape
from speculative_sampling.models.h3.deployment import CAPABILITIES as H3_CAPABILITIES
from speculative_sampling.runtime import web_app as shared_web
from speculative_sampling.runtime.deployment import load_deployment
from speculative_sampling.runtime.media import PreviewBuilder, prepare_preview, synthetic_preview


_SESSION_RUNTIME = os.environ.get("H3_SESSION_RUNTIME", "speculative_sampling")
if _SESSION_RUNTIME == "speculative_sampling":
    from speculative_sampling.models.h3 import H3SpeculativeSamplingLiveSession as LiveSession
elif _SESSION_RUNTIME == "legacy":
    # Explicit comparison escape hatch. New deployments use shared SpeculativeSampling.
    from h3_runtime.live import LiveSession
    LiveSession.capabilities = H3_CAPABILITIES
else:
    raise ValueError("H3_SESSION_RUNTIME must be 'speculative_sampling' or 'legacy'")


def source_revision(path):
    """Return traceable source metadata for Git and release archives."""
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("SPECULATIVE_SAMPLING_SOURCE_REVISION", "release-archive"), None


def session_capabilities():
    return shared_web.session_capabilities(LiveSession, H3_CAPABILITIES)


class Demo(shared_web.Demo):
    """Bind H3's model session to the model-independent live runtime."""

    session_class = LiveSession
    allowed_durations = (5, 10)
    proposal_frames_by_duration = {5: 124, 10: 243}
    rtc_durations = (5,)
    rtc_geometries = ((1344, 768), (832, 480))

    def __init__(self, args):
        if getattr(args, "live_policy", None) is None:
            from speculative_sampling.models.h3.deployment import LIVE_POLICY
            args.live_policy = LIVE_POLICY
        super().__init__(args)
        # Keep the public request duration at the established five-second arm,
        # while allowing a benchmark deployment to select the shortest native
        # 22-frame proposal.  The generated frame count, not this label, drives
        # model execution and SpeculativeSampling geometry.
        proposal_frames = int(getattr(args, "proposal_frames", 124))
        if proposal_frames != 124:
            self.allowed_durations = (5,)
            self.proposal_frames_by_duration = {5: proposal_frames}

    def start(self, body):
        # This deployment always continues from a decoded full-resolution RGB
        # boundary. Latent/relay modes belong only to the frozen legacy app.
        body = dict(body)
        body["mode"] = "image"
        return super().start(body)


make_handler = shared_web.make_handler
serve_stream = shared_web.serve_stream
control_round_trip = shared_web.control_round_trip
AsyncJsonlTrace = shared_web.AsyncJsonlTrace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--game-config", type=Path,
        help="optional JavaScript game definition served to the action UI",
    )
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preview-width", type=int, default=448)
    parser.add_argument("--generation-width", type=int, default=1344)
    parser.add_argument("--generation-height", type=int, default=768)
    parser.add_argument("--attention-backend", choices=("dense", "sol", "sol_bsa"),
                        default="sol_bsa")
    parser.add_argument("--vae-decode", choices=("stream", "whole"), default="stream",
                        help="stream: seven-latent windows, finalized 17-frame blocks; whole: baseline")
    rtc_group = parser.add_mutually_exclusive_group()
    rtc_group.add_argument(
        "--enable-rtc", dest="enable_rtc", action="store_true",
        help="enable WebRTC delivery (default)",
    )
    rtc_group.add_argument(
        "--disable-rtc", dest="enable_rtc", action="store_false",
        help="disable WebRTC and use the WebSocket fallback",
    )
    parser.set_defaults(enable_rtc=True)
    parser.add_argument("--rtc-width", type=int, default=1344)
    parser.add_argument("--rtc-height", type=int, default=768)
    parser.add_argument("--rtc-video-bitrate", type=int, default=3000000,
                        help="initial H.264 bitrate; aiortc congestion feedback remains enabled")
    parser.add_argument("--rtc-pacing-bitrate", type=int, default=0,
                        help="extra RTP packet pacer in bit/s; keep 0 for TCP/TURN paths")
    parser.add_argument("--rtc-ice-config", type=Path,
                        help="operator JSON with iceServers; default local/LAN host candidates only")
    parser.add_argument("--rtc-browser-ice-config", type=Path,
                        help="optional browser-specific ICE config (e.g. TURN reached through SSH)")
    parser.add_argument("--initial-budget", type=float, default=3.5)
    parser.add_argument(
        "--proposal-frames", type=int, choices=(22, 124, 243), default=124,
        help="native H3 proposal length; 22 is the short-proposal study arm",
    )
    parser.add_argument(
        "--deployment-config", type=Path,
        default=Path(__file__).resolve().parents[1] / "speculative_sampling" / "models" / "h3"
                / "configs" / "boundary_image_5s_b300_sol_h3_vae.json",
        help="versioned measured scheduling policy",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-latency", type=float, default=1.2)
    args = parser.parse_args()
    if args.game_config is not None and (
        not args.game_config.is_file() or args.game_config.suffix != ".js"
    ):
        parser.error("--game-config must point to an existing .js file")
    deployment_spec = load_deployment(args.deployment_config)
    args.live_policy = deployment_spec.policy
    args.vae_backend_config = deployment_spec.document["decoder"]
    args.vae_backend = args.vae_backend_config.get("backend", "original")
    args.vae_backends = tuple(
        args.vae_backend_config.get("available_backends", [args.vae_backend])
    )
    if (args.generation_width, args.generation_height) not in (
            (1344, 768), (832, 480), (832, 448), (1280, 736)):
        parser.error("unsupported default generation geometry")
    if (not 32 <= args.rtc_width <= 1344 or not 32 <= args.rtc_height <= 768
            or args.rtc_width % 2 or args.rtc_height % 2):
        parser.error("RTC dimensions must be even and within 1344x768")
    if not 500000 <= args.rtc_video_bitrate <= 3000000:
        parser.error("--rtc-video-bitrate must be in aiortc H.264 range [500000, 3000000]")
    if args.rtc_pacing_bitrate != 0 and args.rtc_pacing_bitrate < args.rtc_video_bitrate:
        parser.error("--rtc-pacing-bitrate must be 0 or at least --rtc-video-bitrate")
    rank = int(os.environ.get("RANK", 0))
    demo = Demo(args) if rank == 0 else None
    server = None
    stream_server = None
    if demo:
        commit, worktree_dirty = source_revision(Path(__file__).parent)
        model_manifest = Path(args.model) / "checkpoint_content.json"
        versions = {}
        for name in ("torch", "diffusers", "websockets", "Pillow", "aiortc", "av"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        (demo.root / "run_metadata.json").write_text(json.dumps({
            "code_commit": commit, "model": args.model, "mock": args.mock,
            "worktree_dirty": worktree_dirty,
            "source_sha256": {
                str(p.resolve().relative_to(Path(__file__).resolve().parents[1])):
                    hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path(shared_web.__file__),
                          Path(shared_web.__file__).with_name("web") / "interactive.html",
                          Path(shared_web.__file__).with_name("web") / "latency.js",
                          Path(shared_web.__file__).with_name("web") / "startup_buffer.js",
                          args.deployment_config,
                          *sorted(Path(__file__).with_name("h3_runtime").glob("*.py"))]
            },
            "preview_width": args.preview_width, "http_port": args.port,
            "generation_width": args.generation_width,
            "generation_height": args.generation_height,
            "attention_backend": args.attention_backend,
            "vae_decode": args.vae_decode,
            "vae_backend": args.vae_backend,
            "vae_backends": list(args.vae_backends),
            "rtc_enabled": args.enable_rtc, "rtc_width": args.rtc_width,
            "rtc_height": args.rtc_height,
            "rtc_video_bitrate": args.rtc_video_bitrate,
            "rtc_pacing_bitrate": args.rtc_pacing_bitrate,
            "sampling": {
                "mode": session_capabilities().sampling.mode,
                "requires_refill": session_capabilities().sampling.requires_refill,
            },
            "live_scheduling": {
                "auto_extend_timeline": args.live_policy.auto_extend_timeline,
                "extension_mode": args.live_policy.extension_mode,
                "condition_ready_to_first_commit_seconds":
                    args.live_policy.condition_ready_to_first_commit_seconds,
                "continuation_to_first_commit_seconds":
                    args.live_policy.continuation_to_first_commit_seconds,
                "ema_alpha": args.live_policy.ema_alpha,
                "ema_margin_seconds": args.live_policy.ema_margin_seconds,
                "delivery_credit_frames": args.live_policy.delivery_credit_frames,
                "rtc_jitter_target_seconds": args.live_policy.rtc_jitter_target_seconds,
                "playhead_clock_mode": args.live_policy.playhead_clock_mode,
                "playhead_feedback_timeout_seconds":
                    args.live_policy.playhead_feedback_timeout_seconds,
                "rewrite_horizon_fallback":
                    args.live_policy.rewrite_horizon_fallback,
                "rewrite_readiness_window":
                    args.live_policy.rewrite_readiness_window,
                "rewrite_guard_frames":
                    args.live_policy.rewrite_guard_frames,
            },
            "websocket_port": args.port + 1, "python": sys.version,
            "versions": versions,
            "checkpoint_manifest_sha256": hashlib.sha256(model_manifest.read_bytes()).hexdigest()
            if model_manifest.is_file() else None,
        }, indent=2))
        server = ThreadingHTTPServer((args.bind, args.port), make_handler(demo))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True, name="h3-http").start()
        stream_server = serve_stream(demo)
        print(f"HTTP listening on {args.bind}:{args.port}; model warming up", flush=True)
    engine = None
    latent_bank = {}
    bank_session = None
    try:
        if not args.mock:
            import torch
            import torch.distributed as dist
            from h3_runtime.engine import MiniMaxH3Inference
            from h3_runtime.cancellation import GenerationSuperseded

            engine = MiniMaxH3Inference(args.model, adapter_path=None, task="i2v",
                                        attention_backend=args.attention_backend)
            from speculative_sampling.models.h3.sol_h3_vae import configure_vae_backend

            engine = configure_vae_backend(engine, args.vae_backend_config)
            ref = args.reference if args.reference else Image.new(
                "RGB", (args.generation_width, args.generation_height))
            for vae_backend in engine.available_vae_backends:
                warm = engine.generate(
                    "A cat sitting calmly in a room.", duration=5,
                    num_frames=args.proposal_frames,
                    width=args.generation_width, height=args.generation_height, image=ref,
                    seed=42, retain_latents=True,
                    stream_vae=args.vae_decode == "stream",
                    vae_backend=vae_backend,
                )
                if rank == 0:
                    warm_dir = args.output / "_warmup" / vae_backend
                    prepare_preview(warm, warm_dir, width=args.preview_width)
                    print(
                        f"{vae_backend} warmup complete; normalized latent shape ",
                        tuple(engine.last_video_latents.shape), flush=True,
                    )
                    del warm
        if demo:
            demo.ready = True
        while True:
            task = None
            if rank == 0:
                task = demo.next_task()
            if not args.mock and dist.is_initialized():
                request = [task]
                dist.broadcast_object_list(request, src=0, device=engine.device)
                task = request[0]
            if task is None:
                # All ranks must finish the same idle collective. Previously
                # rank zero waited for user input while peers entered a NCCL
                # broadcast, killing an idle service after the 600s timeout.
                if demo:
                    demo.idle_ticks += 1
                time.sleep(0.2)
                continue
            begin = time.monotonic()
            stages, engine_elapsed = {}, None
            if args.mock:
                time.sleep(args.mock_latency)
                preview = synthetic_preview(task["proposal"],
                                            args.rtc_width if task["transport"] == "rtc" else args.preview_width,
                                            transport=task["transport"])
                if args.vae_decode == "stream" or task["vae_backend"] == "sol-h3-vae":
                    for end in [17, 34, 51, 68, 85, 102, 119, 124]:
                        demo.publish(task, preview, end)
                        time.sleep(.015)
            else:
                if bank_session != task["session"]:
                    latent_bank.clear()
                    bank_session = task["session"]
                latent_bank = {k: v for k, v in latent_bank.items() if k in task["keep_proposals"]}
                kwargs = {}
                anchor = task["anchor"]
                if anchor is None:
                    kwargs["image"] = task["reference"]
                elif task["mode"] == "latent":
                    z = latent_bank[anchor["proposal"]]
                    kwargs["condition_latent"] = z[:, :, anchor["latent_index"]:anchor["latent_index"] + 1]
                elif task["mode"] == "relay":
                    context = task["relay_context"]
                    z = latent_bank[context["proposal"]]
                    kwargs["relay_latents"] = z[:, :, context["latent_start"]:context["latent_stop"]]
                else:
                    anchor_tensor = torch.empty(
                        boundary_image_shape(task["generation"]),
                        dtype=torch.uint8,
                        device=engine.device,
                    )
                    anchor_error = None
                    if rank == 0:
                        try:
                            with demo.lock:
                                rgb = demo.previews[anchor["proposal"]].anchor_frames[
                                    anchor["source_frame"]
                                ].copy()
                            if tuple(rgb.shape) != tuple(anchor_tensor.shape):
                                raise ValueError(
                                    f"anchor geometry {tuple(rgb.shape)} != expected "
                                    f"{tuple(anchor_tensor.shape)}"
                                )
                            anchor_tensor.copy_(torch.from_numpy(rgb).to(engine.device))
                        except Exception as exc:
                            anchor_error = f"{type(exc).__name__}: {exc}"
                    if dist.is_initialized():
                        errors = [anchor_error]
                        dist.broadcast_object_list(errors, src=0, device=engine.device)
                        anchor_error = errors[0]
                    if anchor_error:
                        raise RuntimeError("failed to prepare boundary image: " + anchor_error)
                    if dist.is_initialized():
                        dist.broadcast(anchor_tensor, src=0)
                    kwargs["image"] = Image.fromarray(anchor_tensor.cpu().numpy())
                builder = PreviewBuilder(
                    task["output"], width=args.rtc_width if task["transport"] == "rtc" else args.preview_width,
                    anchors=task["mode"] == "image", transport=task["transport"],
                    proposal_frames=task["proposal_frames"],
                ) if rank == 0 else None

                def on_video(start, video):
                    expected = (task["generation"]["height"], task["generation"]["width"])
                    if tuple(video.shape[-2:]) != expected:
                        raise ValueError(f"decoded geometry {tuple(video.shape[-2:])} != requested {expected}")
                    if start == 0:
                        # Preserve the proven first-pixel path. Later chunks can
                        # overlap D2H/media preparation with causal decode.
                        preview = builder.video(start, video)
                        demo.publish(task, preview)
                    else:
                        builder.video_async(
                            start, video,
                            on_ready=lambda preview: demo.publish(task, preview),
                        )

                try:
                    def cancel_check(_step):
                        if rank != 0 or demo is None or task["action"] not in ("continuation", "refill"):
                            return False
                        with demo.lock:
                            return bool(
                                demo.session
                                and demo.session.id == task["session"]
                                and demo.session.should_cancel_inflight(task)
                            )

                    generation = dict(task["generation"])
                    generation["num_frames"] = task["proposal_frames"]
                    media = engine.generate(
                        task["prompt"], **generation, seed=task["seed"], retain_latents=True,
                        profile_stages=True,
                        stream_vae=args.vae_decode == "stream",
                        vae_backend=task["vae_backend"],
                        on_audio=builder.audio if builder else None,
                        on_video=on_video if builder else None,
                        cancel_check=(
                            cancel_check if task["action"] in ("continuation", "refill") else None
                        ), **kwargs)
                except GenerationSuperseded as exc:
                    if builder:
                        builder.close()
                    if demo:
                        demo.cancel(task, str(exc))
                    print(
                        f"cancelled session={task['session']} proposal={task['proposal']} "
                        f"prompt_version={task['prompt_version']}", flush=True,
                    )
                    continue
                except BaseException:
                    if builder:
                        builder.close()
                    raise
                latent_bank[task["proposal"]] = engine.last_video_latents
                if rank == 0:
                    with demo.lock:
                        if demo.session and demo.session.id == task["session"]:
                            demo.session.event("generation_geometry", proposal=task["proposal"],
                                generation=task["generation"],
                                latent_shape=list(engine.last_video_latents.shape))
                preview = None
                prep_error = None
                if rank == 0:
                    try:
                        builder.drain_video()
                        if task["vae_backend"] == "original" and args.vae_decode == "whole":
                            builder.audio(media.audio, media.audio_sample_rate)
                            builder.video(0, media.video)
                        preview = builder.finish()
                        stages = {**engine.last_stage_timings, **builder.timer.report()}
                        engine_elapsed = media.elapsed_s
                        del media
                    except Exception:
                        prep_error = traceback.format_exc()
                    finally:
                        builder.close()
                if dist.is_initialized():
                    errors = [prep_error]
                    dist.broadcast_object_list(errors, src=0, device=engine.device)
                    prep_error = errors[0]
                if prep_error:
                    raise RuntimeError(prep_error)
            if demo:
                elapsed = time.monotonic() - begin
                demo.finish(task, preview, elapsed, stages, engine_elapsed)
                print(f"ready session={task['session']} proposal={task['proposal']} "
                      f"mode={task['mode']} boundary={task['boundary']} "
                      f"prompt_version={task['prompt_version']} end_to_end_s={elapsed:.3f}", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        if demo:
            with demo.lock:
                demo.global_error = error.splitlines()[-1]
                demo.model_busy = False
                if demo.session:
                    demo.session.fail(demo.global_error)
        # Do not silently serve a dead distributed worker as healthy.
        raise
    finally:
        if server:
            server.shutdown()
        if stream_server:
            stream_server.shutdown()
        if demo and demo.rtc:
            demo.rtc.shutdown()
        if demo and demo.trace:
            demo.trace.close()


if __name__ == "__main__":
    main()
