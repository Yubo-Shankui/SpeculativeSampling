#!/usr/bin/env python3
"""Model-independent single-session HTTP, WebSocket and RTC application.

Only rank zero serves HTTP. Model requests are broadcast in order to ALL ranks;
HTTP handlers never invoke CUDA/distributed operations. The token protects APIs;
bind an internal interface and expose only through SSH localhost forwarding.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import hashlib
import importlib.metadata
import io
import json
import math
import os
import queue
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from .deployment import live_scheduling_status
from .media import PreviewBuilder, prepare_preview, synthetic_preview


class AsyncJsonlTrace:
    """Keep result persistence off the RTC media-clock thread."""

    _STOP = object()

    def __init__(self, path: Path):
        self.path = path
        self.queue = queue.Queue(maxsize=65536)
        self.dropped = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="speculative-sampling-trace-writer"
        )
        self.thread.start()

    def emit(self, event):
        if self.closed:
            return
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            # Never block the 24 FPS media clock on result storage. The large
            # queue covers many minutes; record any exceptional loss on close.
            self.dropped += 1

    def _run(self):
        with self.path.open("a", encoding="utf-8", buffering=1024 * 1024) as output:
            last_flush = time.monotonic()
            while True:
                try:
                    item = self.queue.get(timeout=0.25)
                except queue.Empty:
                    item = None
                if item is self._STOP:
                    break
                if item is not None:
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                if time.monotonic() - last_flush >= 0.5:
                    output.flush()
                    last_flush = time.monotonic()
            if self.dropped:
                output.write(json.dumps({
                    "event": "trace_overflow", "dropped": self.dropped,
                    "wall_time": time.time(), "monotonic": time.monotonic(),
                }) + "\n")
            output.flush()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(self._STOP)
        self.thread.join(timeout=10)


def session_capabilities(session_class, fallback=None):
    return getattr(session_class, "capabilities", fallback)


def control_round_trip(now: float, echoed: float, browser_delay: float) -> float | None:
    """Remove time deliberately held in the browser from an echoed RTT.

    Background-tab timer throttling and main-thread work can delay a heartbeat
    for seconds.  That delay is neither network time nor model safety margin.
    """
    if not all(math.isfinite(value) for value in (now, echoed, browser_delay)):
        return None
    if not 0 <= browser_delay <= 60:
        return None
    value = now - echoed - browser_delay
    return value if 0 <= value <= 10 else None


class Demo:
    session_class = None
    allowed_durations = (5, 10)
    proposal_frames_by_duration = {5: 124, 10: 243}
    rtc_durations = (5,)
    rtc_geometries = ((1344, 768), (832, 480))
    def __init__(self, args):
        self.args = args
        self.live_policy = getattr(args, "live_policy", None)
        if self.live_policy is None:
            raise ValueError("live runtime requires an explicit deployment policy")
        self.session_class = getattr(args, "session_class", None) or type(self).session_class
        if self.session_class is None:
            raise ValueError("live runtime requires a model session class")
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token = args.token_file.read_text().strip()
        if len(self.token) < 24:
            raise ValueError("token file must contain at least 24 random characters")
        self.lock = threading.RLock()
        self.session = None
        self.previews = {}
        self.ready = False
        self.model_busy = False
        self.global_error = None
        self.idle_ticks = 0
        self.trace = None
        self.rtc = None
        if getattr(args, "enable_rtc", False):
            from .rtc import RTCManager
            self.rtc = RTCManager(self)
        self.transport = "rtc" if self.rtc is not None else "websocket"

    def status(self):
        capabilities = session_capabilities(self.session_class)
        default_vae_backend = getattr(self.args, "vae_backend", "original")
        vae_backends = list(getattr(
            self.args, "vae_backends", (default_vae_backend,)
        ))
        return {
            "ready": self.ready, "mock": self.args.mock,
            "session_runtime": (
                f"{self.session_class.__module__}.{self.session_class.__name__}"
            ),
            "model_busy": self.model_busy, "error": self.global_error,
            "idle_ticks": self.idle_ticks,
            "preview_width": self.args.preview_width,
            "vae_decode": getattr(self.args, "vae_decode", "whole"),
            "vae_backends": vae_backends,
            "default_vae_backend": default_vae_backend,
            "session_vae_backend": getattr(self.session, "vae_backend", None),
            "transport": self.transport,
            "conditioning": dict(capabilities.conditioning),
            "live_scheduling": live_scheduling_status(self.live_policy),
            "rtc": {"available": self.rtc is not None,
                    "state": self.rtc.state if self.rtc else "disabled",
                    "width": getattr(self.args, "rtc_width", 1344),
                    "height": getattr(self.args, "rtc_height", 768),
                    "initial_video_bitrate": getattr(self.args, "rtc_video_bitrate", 3000000),
                    "packet_pacing_bitrate": getattr(self.args, "rtc_pacing_bitrate", 8000000),
                    "jitter_target_seconds": self.live_policy.rtc_jitter_target_seconds,
                    "media_source": self.rtc.snapshot() if self.rtc else None,
                    "ice_config": self.rtc.browser_config if self.rtc else None},
            "session_state": self.session.snapshot() if self.session else None,
        }

    def start(self, body):
        received_at = time.monotonic()
        if not self.ready or self.model_busy:
            raise ValueError("worker is warming up or finishing a prior request")
        if self.session and not self.session.stopped and not self.session.error:
            raise ValueError("one session at a time; stop the current session first")
        prompt = self.session_class._prompt(body.get("prompt"))
        mode = body.get("mode", "latent")
        if mode not in ("latent", "image", "relay"):
            raise ValueError("unknown conditioning mode")
        transport = body.get(
            "transport", "rtc" if self.rtc is not None else "websocket"
        )
        if transport not in ("websocket", "rtc"):
            raise ValueError("unknown media transport")
        if transport == "rtc" and self.rtc is None:
            raise ValueError(
                "RTC is unavailable; install the RTC dependencies and do not pass --disable-rtc"
            )
        default_vae_backend = getattr(self.args, "vae_backend", "original")
        available_vae_backends = tuple(getattr(
            self.args, "vae_backends", (default_vae_backend,)
        ))
        vae_backend = body.get("vae_backend", default_vae_backend)
        if vae_backend not in available_vae_backends:
            raise ValueError("requested VAE backend is not loaded")
        capabilities = session_capabilities(self.session_class)
        accepts_image = bool(capabilities.conditioning.get("image", False))
        raw = body.get("image")
        if raw and not accepts_image:
            raise ValueError("selected model variant does not accept a reference image")
        image = None
        if accepts_image:
            if raw:
                try:
                    encoded = raw.split(",", 1)[-1]
                    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    if image.width * image.height > 24_000_000:
                        raise ValueError("image too large")
                    image.thumbnail((2048, 2048))
                except Exception as exc:
                    raise ValueError("invalid reference image") from exc
            elif self.args.reference:
                image = Image.open(self.args.reference).convert("RGB")
            else:
                raise ValueError("upload an initial reference image")
        seed = int(body.get("seed", 42))
        if not 0 <= seed < 2**63 - 1:
            raise ValueError("seed out of range")
        generation = {
            "width": int(body.get("generation_width", self.args.generation_width)),
            "height": int(body.get("generation_height", self.args.generation_height)),
            "duration": int(body.get("duration", 5)),
        }
        if (generation["width"], generation["height"]) not in (
            (1344, 768), (832, 480), (832, 448), (1280, 736),
        ):
            raise ValueError("unsupported generation geometry")
        if generation["duration"] not in self.allowed_durations:
            raise ValueError(f"duration must be one of {self.allowed_durations}")
        if self.args.mock and generation["duration"] != 5:
            raise ValueError("mock media currently supports only 5 seconds")
        if transport == "rtc" and (generation["duration"] not in self.rtc_durations
                                  or (generation["width"], generation["height"]) not in self.rtc_geometries):
            raise ValueError("benchmark variable geometry is validated with WebSocket only")
        # Session ID is created before its path is bound; initial log is buffered.
        events = []
        live_policy = self.live_policy
        session_kwargs = {
            "seed": seed,
            "mode": mode,
            "initial_budget": self.args.initial_budget,
            "delivery_ahead_frames": live_policy.delivery_credit_frames,
            "emit": events.append,
            "proposal_frames": self.proposal_frames_by_duration[generation["duration"]],
            "serving_policy": body.get("serving_policy", "co"),
        }
        if capabilities.sampling.mode == "bidirectional":
            configured_clock = live_policy.playhead_clock_mode
            effective_clock = configured_clock
            if configured_clock == "transport_paced":
                # RTC commits exactly one frame on the backend 24 FPS clock,
                # so its commit frontier is the precise scheduling clock. A
                # WebSocket can greedily drain future media and must instead
                # retain the browser-feedback clock.
                effective_clock = (
                    "committed" if transport == "rtc" else "extrapolated"
                )
            session_kwargs.update(
                auto_extend_timeline=live_policy.auto_extend_timeline,
                extension_mode=live_policy.extension_mode,
                condition_ready_to_first_commit_seconds=(
                    live_policy.condition_ready_to_first_commit_seconds
                ),
                continuation_to_first_commit_seconds=(
                    live_policy.continuation_to_first_commit_seconds
                ),
                ema_alpha=live_policy.ema_alpha,
                ema_margin_seconds=live_policy.ema_margin_seconds,
                playhead_clock_mode=effective_clock,
                playhead_feedback_timeout_seconds=(
                    live_policy.playhead_feedback_timeout_seconds
                ),
                rewrite_horizon_fallback=live_policy.rewrite_horizon_fallback,
                rewrite_readiness_window=live_policy.rewrite_readiness_window,
                rewrite_guard_frames=live_policy.rewrite_guard_frames,
            )
        session = self.session_class(prompt, "", **session_kwargs)
        initial_next = body.get("next_prompts")
        if initial_next is None:
            initial_next = [body.get("next_prompt", prompt)]
        if hasattr(session, "next_prompts"):
            session.update_prompt(prompt, next_prompts=initial_next)
        session.generation = generation
        session.vae_backend = vae_backend
        session.prompt_received_at[1] = received_at
        directory = self.root / session.id
        directory.mkdir()
        if image is not None:
            reference = directory / "reference.png"
            image.save(reference)
            session.reference = str(reference)
        if self.trace:
            self.trace.close()
        self.trace = AsyncJsonlTrace(directory / "trace.jsonl")
        session.emit = self.trace.emit
        for event in events:
            self.trace.emit(event)
        self.session = session
        self.transport = transport
        self.previews = {}
        (directory / "request.json").write_text(json.dumps({
            "prompt": prompt, "mode": mode, "seed": seed,
            "model": self.args.model, "mock": self.args.mock,
            "preview_width": self.args.preview_width,
            "transport": transport, "rtc_width": getattr(self.args, "rtc_width", 1344),
            "vae_decode": getattr(self.args, "vae_decode", "whole"),
            "vae_backend": vae_backend,
            "generation": generation, "serving_policy": session.serving_policy,
            "proposal_frames": session.proposal_frames,
            "sampling": {
                "mode": capabilities.sampling.mode,
                "requires_refill": capabilities.sampling.requires_refill,
            },
            "live_scheduling": {
                "auto_extend_timeline": live_policy.auto_extend_timeline,
                "extension_mode": live_policy.extension_mode,
                "condition_ready_to_first_commit_seconds":
                    live_policy.condition_ready_to_first_commit_seconds,
                "continuation_to_first_commit_seconds":
                    live_policy.continuation_to_first_commit_seconds,
                "ema_alpha": live_policy.ema_alpha,
                "ema_margin_seconds": live_policy.ema_margin_seconds,
                "delivery_credit_frames": live_policy.delivery_credit_frames,
                "rtc_jitter_target_seconds": live_policy.rtc_jitter_target_seconds,
                "playhead_clock_mode": live_policy.playhead_clock_mode,
                "effective_playhead_clock_mode": getattr(
                    session, "playhead_clock_mode", live_policy.playhead_clock_mode,
                ),
                "playhead_feedback_timeout_seconds":
                    live_policy.playhead_feedback_timeout_seconds,
                "rewrite_horizon_fallback": live_policy.rewrite_horizon_fallback,
                "rewrite_readiness_window": live_policy.rewrite_readiness_window,
                "rewrite_guard_frames": live_policy.rewrite_guard_frames,
            },
        }, ensure_ascii=False, indent=2))
        return session.snapshot()

    def checked_session(self, body):
        if not self.session or body.get("session") != self.session.id:
            raise ValueError("session is no longer active")
        return self.session

    def command(self, path, body):
        if path == "/api/clock":
            return {"server_monotonic": time.monotonic()}
        if path == "/api/rtc/offer":
            with self.lock:
                session = self.checked_session(body)
                if session.stopped or self.transport != "rtc" or self.rtc is None:
                    raise ValueError("no active RTC session")
                sid = session.id
            # ICE gathering must not hold the scheduler/control lock.
            return self.rtc.offer(sid, body.get("sdp"), body.get("type"))
        with self.lock:
            if path == "/api/start":
                return self.start(body)
            session = self.checked_session(body)
            if path == "/api/prompt":
                prompt = body.get("prompt")
                if hasattr(session, "next_prompts"):
                    next_prompts = body.get("next_prompts")
                    if next_prompts is None:
                        next_prompt = body.get("next_prompt")
                        if next_prompt is None:
                            next_prompt = body.get("followup_prompt", prompt)
                        next_prompts = [next_prompt]
                    version = session.update_prompt(prompt, next_prompts=next_prompts)
                else:
                    version = session.update_prompt(prompt)
                    followup = body.get("followup_prompt")
                    if followup is not None:
                        session.set_followup_prompt(version, followup)
                return {"version": version, "arrival_monotonic": session.prompt_received_at[version]}
            if path == "/api/scheduling/lead":
                update = getattr(session, "update_scheduling_lead", None)
                if update is None:
                    raise ValueError("live scheduling updates are not supported by this model")
                return update(
                    body.get("lead_seconds"),
                    condition_seconds=body.get("condition_lead_seconds"),
                    continuation_seconds=body.get("continuation_lead_seconds"),
                )
            if path == "/api/heartbeat":
                session.heartbeat(int(body.get("played_frame", 0)))
                return self.status()
            if path == "/api/stop":
                session.stop()
                if self.rtc:
                    self.rtc.close_session()
                return {"stopped": True}
            raise ValueError("unknown command")

    def delivery(self, query):
        with self.lock:
            session = self.checked_session({"session": query.get("session", [""])[0]})
            if self.transport == "rtc":
                raise ValueError("RTC session media is delivered only by the RTC peer")
            units = session.deliver(int(query.get("cursor", ["0"])[0]))
            # Fully build the response while protected; pruning cannot race it.
            return {"units": [self.previews[u.proposal].packet(u) for u in units],
                    "state": session.snapshot()}

    def next_task(self):
        with self.lock:
            if not self.session:
                return None
            if self.transport == "rtc" and (not self.rtc or self.rtc.session != self.session.id
                                             or self.rtc.state != "connected"):
                return None
            task = self.session.plan()
            if task:
                self.model_busy = True
                task["output"] = str(self.root / task["session"] / f"proposal_{task['proposal']:05d}")
                task["transport"] = self.transport
                task["generation"] = self.session.generation
                task["vae_backend"] = getattr(
                    self.session, "vae_backend", getattr(self.args, "vae_backend", "original")
                )
                # Output path is also part of the in-flight identity.
                self.session.busy["output"] = task["output"]
                self.session.busy["transport"] = self.transport
                self.session.busy["generation"] = self.session.generation
                self.session.busy["vae_backend"] = task["vae_backend"]
            return task

    def publish(self, task, preview, source_end=None):
        with self.lock:
            if self.session and self.session.id == task["session"] and not self.session.stopped:
                self.previews[task["proposal"]] = preview
                self.session.publish(task, preview.available if source_end is None else source_end)

    def finish(self, task, preview, elapsed, stages=None, engine_elapsed=None):
        with self.lock:
            self.model_busy = False
            if self.session and self.session.id == task["session"]:
                if self.session.complete(task, elapsed):
                    self.session.record_metrics(task, stages or {}, engine_elapsed, mock=self.args.mock)
                    self.previews[task["proposal"]] = preview
                    keep = {u.proposal for u in self.session.units
                            if u.end > max(0, self.session.played - 24)}
                    self.previews = {k: v for k, v in self.previews.items() if k in keep}

    def cancel(self, task, reason="superseded"):
        with self.lock:
            self.model_busy = False
            if self.session and self.session.id == task["session"]:
                return self.session.cancel(task, reason)
            return False


def make_handler(demo):
    html_path = Path(__file__).with_name("web") / "interactive.html"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            # Avoid leaking bearer token or uploaded data to logs.
            if self.command != "GET" or "/api/units" not in self.path:
                print("http", self.command, urlparse(self.path).path, fmt % args, flush=True)

        def send(self, value, status=200, content_type="application/json"):
            data = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            expected = "Bearer " + demo.token
            if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
                self.send({"error": "unauthorized"}, 401)
                return False
            return True

        def do_GET(self):
            try:
                path = urlparse(self.path)
                if path.path in ("/", "/margit-duel", "/margit-duel.html"):
                    return self.send(html_path.read_bytes(), content_type="text/html; charset=utf-8")
                if path.path == "/latency.js":
                    return self.send(html_path.with_name("latency.js").read_bytes(),
                                     content_type="text/javascript; charset=utf-8")
                if path.path == "/startup_buffer.js":
                    return self.send(html_path.with_name("startup_buffer.js").read_bytes(),
                                     content_type="text/javascript; charset=utf-8")
                if path.path == "/game-config.js":
                    game_config = getattr(demo.args, "game_config", None)
                    if game_config is None:
                        content = b"window.SPECULATIVE_SAMPLING_GAME = null;\n"
                    else:
                        game_config = Path(game_config)
                        if not game_config.is_file() or game_config.suffix != ".js":
                            raise ValueError("game config must be an existing .js file")
                        content = game_config.read_bytes()
                    return self.send(content, content_type="text/javascript; charset=utf-8")
                if not self.authorized():
                    return
                if path.path == "/api/status":
                    with demo.lock:
                        status = demo.status()
                    return self.send(status)
                if path.path == "/api/units":
                    return self.send(demo.delivery(parse_qs(path.query)))
                if path.path == "/api/reference" and demo.args.reference:
                    return self.send(demo.args.reference.read_bytes(), content_type="image/png")
                return self.send({"error": "not found"}, 404)
            except (ValueError, KeyError) as exc:
                self.send({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                traceback.print_exc()
                self.send({"error": "server error"}, 500)

        def do_POST(self):
            try:
                if not self.authorized():
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 12_000_000:
                    return self.send({"error": "invalid body size"}, 413)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("expected JSON object")
                self.send(demo.command(urlparse(self.path).path, body))
            except (ValueError, KeyError) as exc:
                self.send({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                traceback.print_exc()
                self.send({"error": "server error"}, 500)

    return Handler


def serve_stream(demo):
    """Full-duplex long connection: no per-delivery WAN request round trip.

    All CUDA stays in the model thread. The connection's monotonic cursor is
    independent of the renderer playhead; at most 2s can be irrevocably in flight.
    """
    from websockets.sync.server import serve
    from websockets.exceptions import ConnectionClosed

    def handler(ws):
        finished = threading.Event()
        replies = queue.Queue()
        subscribed_session = None
        try:
            auth = json.loads(ws.recv(timeout=10))
            if not isinstance(auth, dict) or not hmac.compare_digest(str(auth.get("token", "")), demo.token):
                ws.close(1008, "unauthorized")
                return
            ws.send(json.dumps({"type": "authenticated"}))

            def receive():
                nonlocal subscribed_session
                try:
                    # Do not let slow downstream writes delay upstream prompt/clock updates.
                    for raw in ws:
                        message = json.loads(raw)
                        if not isinstance(message, dict):
                            raise ValueError("expected object")
                        if message.get("type") == "heartbeat":
                            with demo.lock:
                                s = demo.checked_session(message)
                                s.heartbeat(int(message.get("played_frame", 0)))
                                subscribed_session = s.id
                                echoed = message.get("echo_monotonic")
                                if isinstance(echoed, (float, int)):
                                    rtt = control_round_trip(
                                        time.monotonic(), float(echoed),
                                        float(message.get("echo_delay_s", 0)),
                                    )
                                    if rtt is not None:
                                        s.transport_rtt_s = (0.8 * s.transport_rtt_s + 0.2 * rtt
                                                             if s.transport_rtt_s else rtt)
                        elif message.get("type") == "command":
                            try:
                                result = demo.command(message["path"], message.get("body", {}))
                                if message["path"] == "/api/start":
                                    subscribed_session = result["session"]
                                replies.put({"type": "reply", "id": message["id"], "result": result})
                            except (ValueError, KeyError) as exc:
                                replies.put({"type": "reply", "id": message.get("id"), "error": str(exc)})
                except ConnectionClosed:
                    pass
                except Exception:
                    traceback.print_exc()
                finally:
                    finished.set()

            threading.Thread(
                target=receive, daemon=True, name="speculative-sampling-controls"
            ).start()
            session_id, cursor, last_status = None, 0, 0
            while not finished.is_set():
                while not replies.empty():
                    ws.send(json.dumps(replies.get_nowait()))
                if demo.rtc and subscribed_session is not None:
                    while not demo.rtc.messages.empty():
                        message = demo.rtc.messages.get_nowait()
                        if message.get("session") == subscribed_session:
                            ws.send(json.dumps(message))
                now = time.monotonic()
                status_due = now - last_status >= 0.25
                with demo.lock:
                    # Status is emitted at 4 Hz.  Rebuilding it at the 15 ms
                    # media/control poll cadence held the shared lock ~66 Hz;
                    # as session history grew, that starved the model thread's
                    # cooperative-cancellation checks and inflated measured
                    # DiT time.
                    status = demo.status() if status_due else None
                    s = demo.session
                    packet = None
                    if (s and not s.stopped and not s.error and subscribed_session == s.id
                            and demo.transport == "websocket"):
                        if session_id != s.id:
                            session_id = s.id
                            cursor = int(getattr(s, "delivery_resume_cursor", 0))
                        delivery_kwargs = {}
                        if getattr(s, "backend_delivery_pacing", False):
                            delivery_kwargs["paced"] = True
                        units = s.deliver(
                            cursor,
                            max_frames=getattr(s, "delivery_packet_frames", 8),
                            **delivery_kwargs,
                        )
                        if units:
                            packet = {"type": "units", "session": session_id,
                                      "units": [demo.previews[u.proposal].packet(u) for u in units]}
                            cursor = units[-1].end
                # Never hold the state lock while writing to a slow network.
                if status_due:
                    ws.send(json.dumps({"type": "status", "data": status,
                                        "sent_monotonic": time.monotonic()}, ensure_ascii=False))
                    last_status = now
                if packet:
                    ws.send(json.dumps(packet, separators=(",", ":")))
                finished.wait(0.015)
        except (ConnectionClosed, TimeoutError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            finished.set()
            with demo.lock:
                # RTC media has its own peer connection. A transient control
                # WebSocket reconnect must not destroy the media consumer.
                # Session ownership is enforced by /api/start, not by a tab.
                if demo.session and demo.transport != "rtc" and subscribed_session == demo.session.id:
                    demo.session.last_seen = time.monotonic() - 11

    server = serve(handler, demo.args.bind, demo.args.port + 1,
                   max_size=12_000_000, compression=None, ping_interval=15, ping_timeout=30)
    threading.Thread(
        target=server.serve_forever, daemon=True, name="speculative-sampling-websocket"
    ).start()
    return server
