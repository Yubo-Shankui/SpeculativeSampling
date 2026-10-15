"""Resident Wan-CF++ worker with generator-owned speculative KV state."""

from __future__ import annotations

import sys
import time
import importlib
from pathlib import Path


def _select_available_attention_backend():
    """Reject namespace-only FlashAttention installs used by newer images.

    CF++ treats any importable ``flash_attn`` package as FlashAttention 2, but
    the B300 image ships a FlashAttention 4 namespace without the FA2 entry
    point.  Leave FA3 alone and make CF++ use PyTorch SDPA when neither of its
    supported callable interfaces is actually present.
    """
    attention = importlib.import_module("wan.modules.attention")
    flash_attn = getattr(attention, "flash_attn", None)
    if attention.FLASH_ATTN_2_AVAILABLE and not callable(
        getattr(flash_attn, "flash_attn_varlen_func", None)
    ):
        attention.FLASH_ATTN_2_AVAILABLE = False
    if attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        return "flash-attention"

    def sdpa_flash_attention(
        q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
        softmax_scale=None, q_scale=None, causal=False,
        window_size=(-1, -1), deterministic=False, dtype=None, version=None,
    ):
        kwargs = {
            "q_lens": q_lens, "k_lens": k_lens,
            "dropout_p": dropout_p, "softmax_scale": softmax_scale,
            "q_scale": q_scale, "causal": causal,
            "window_size": window_size, "deterministic": deterministic,
            "fa_version": version,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype
        return attention.attention(q, k, v, **kwargs)

    # Some Wan modules import flash_attention directly rather than the
    # dispatching `attention` function. Redirect those bound names too.
    for module_name in ("wan.modules", "wan.modules.model", "wan.modules.clip"):
        module = importlib.import_module(module_name)
        module.flash_attention = sdpa_flash_attention
    return "torch-sdpa"


class WanCFPPWorker:
    model_id = "wan13b_cfpp"
    mock = False

    def __init__(self, args):
        import torch
        import torch.distributed as dist

        self.args, self.torch = args, torch
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if args.cfpp_site:
            sys.path.append(str(args.cfpp_site))
        sys.path.insert(0, str(args.cfpp_repo))
        from .loader import load_resident_pipeline

        _select_available_attention_backend()

        self.backend = load_resident_pipeline(
            repo=args.cfpp_repo, config=args.config,
            checkpoint=args.checkpoint, device="cuda",
        )
        self.pipeline = self.backend.pipeline
        if int(getattr(self.pipeline, "local_attn_size", -1)) == -1:
            # Upstream allocates 21 latent frames in this configuration. Make
            # its existing rolling-cache branch explicit for resident sessions.
            self.pipeline.local_attn_size = 21
            model = getattr(self.pipeline.generator, "model", None)
            for block in getattr(model, "blocks", ()):
                block.self_attn.local_attn_size = 21
        self.sequence_parallel = None
        if self.world_size > 1:
            from .usp import WanCachedUSP

            if int(args.usp_size) != self.world_size:
                raise ValueError(
                    f"configured USP{args.usp_size} does not match world size {self.world_size}"
                )
            self.sequence_parallel = WanCachedUSP().install(self.pipeline)
        native = int(self.pipeline.num_frame_per_block)
        if native != int(args.chunk_latents):
            raise ValueError(
                f"checkpoint/config native K={native} does not match serving K={args.chunk_latents}"
            )
        self.native = None
        self.last_session = None
        self.condition_session = None
        self.conditions = {}
        self.official_vae_state = None
        self.pending_vae_state = None

    def warmup(self):
        self.torch.cuda.synchronize()

    def encode_condition(self, session_id, version, prompt):
        """Prepare an opaque resident handle before SpeculativeSampling dispatch."""
        session_id, version = str(session_id), int(version)
        if self.condition_session != session_id:
            self.conditions.clear()
            self.condition_session = session_id
        handle = f"{session_id}:{version}"
        cached = self.conditions.get(handle)
        if cached is not None:
            return handle, cached[1]
        started = time.perf_counter()
        condition = self.pipeline.text_encoder(text_prompts=[prompt])
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.conditions[handle] = (condition, elapsed)
        # Only the current/latest few versions can still be selected.
        if len(self.conditions) > 4:
            del self.conditions[next(iter(self.conditions))]
        return handle, elapsed

    def _new_session(self, session_id):
        from .speculative_sampling_native import WanNativeSpeculativeSamplingSession

        if self.native is not None:
            self.native.close()
        self.native = WanNativeSpeculativeSamplingSession(self.pipeline, self.args.chunk_latents)
        self.pipeline.vae.model.clear_cache()
        self.official_vae_state = self._snapshot_vae()
        self.pending_vae_state = None
        self.last_session = session_id

    def _snapshot_vae(self):
        model = self.pipeline.vae.model
        return tuple(model._feat_map)

    def _restore_vae(self, state):
        model = self.pipeline.vae.model
        model._feat_map = list(state)
        model._conv_idx = [0]

    def _decode(self, proposal):
        """Advance official decoder only through accepted history, then fork."""
        self._restore_vae(self.official_vae_state)
        official_started = time.perf_counter()
        promoted = False
        if proposal.committed is not None:
            if (
                proposal.accepted_latents == proposal.previous_pending_latents
                and self.pending_vae_state is not None
            ):
                # The prior preview fork decoded exactly this complete accepted
                # proposal from the same official prefix. Promote that state;
                # re-decoding K merely to reconstruct it is redundant.
                self.official_vae_state = self.pending_vae_state
                promoted = True
            else:
                self.pipeline.vae.decode_to_pixel(proposal.committed, use_cache=True)
                self.torch.cuda.synchronize()
                self.official_vae_state = self._snapshot_vae()
        official_seconds = time.perf_counter() - official_started

        self._restore_vae(self.official_vae_state)
        preview_started = time.perf_counter()
        video = self.pipeline.vae.decode_to_pixel(proposal.values, use_cache=True)
        video = (video[0] * 0.5 + 0.5).clamp(0, 1)
        self.torch.cuda.synchronize()
        preview_seconds = time.perf_counter() - preview_started
        self.pending_vae_state = self._snapshot_vae()
        # A speculative proposal must not advance official decoder history.
        self._restore_vae(self.official_vae_state)
        return video, official_seconds, preview_seconds, promoted

    def generate(self, task, on_block):
        from speculative_sampling.runtime.media import PreviewBuilder

        if task["session"] != self.last_session:
            self._new_session(task["session"])
        started = time.perf_counter()
        handle = task.get("condition_handle")
        if not isinstance(handle, str) or handle not in self.conditions:
            raise ValueError("Wan generation requires a ready encoded condition handle")
        condition, encode_seconds = self.conditions[handle]
        proposal = self.native.advance(
            retain_until=int(task["retain_until"]),
            condition=condition,
            seed=int(task["seed"]),
        )
        if self.rank != 0:
            # Every rank owns its sharded official/speculative KV state and
            # participates in DiT collectives. Only rank zero owns media
            # decoding, persistence, and delivery to the controller.
            return None, {
                "prompt_encode": {
                    "wall_s": encode_seconds,
                    "device": "gpu",
                    "pre_dispatch": True,
                },
                "dit": {
                    "wall_s": proposal.denoise_seconds,
                    "device": f"usp{self.world_size}",
                },
                "commit": {
                    "wall_s": proposal.commit_seconds,
                    "retain_until": int(task["retain_until"]),
                },
            }, time.perf_counter() - started
        video, official_decode_s, preview_decode_s, promoted_vae_state = self._decode(proposal)
        decoded_at = time.perf_counter()
        builder = PreviewBuilder(
            task["output"],
            width=self.args.rtc_width if task["transport"] == "rtc" else self.args.preview_width,
            transport=task["transport"], proposal_frames=task["proposal_frames"],
        )
        try:
            rate = 24000
            audio = self.torch.zeros(
                2, task["proposal_frames"] * rate // 24, dtype=self.torch.float32
            )
            builder.audio(audio, rate)
            preview = builder.video(0, video)
            if on_block:
                on_block(task["proposal_frames"], preview)
            preview = builder.finish()
        except Exception:
            builder.close()
            raise
        finished = time.perf_counter()
        stages = {
            "prompt_encode": {"wall_s": encode_seconds, "device": "gpu", "pre_dispatch": True},
            "dit": {
                "wall_s": proposal.denoise_seconds,
                "device": f"usp{self.world_size}",
            },
            "commit": {
                "wall_s": proposal.commit_seconds,
                "retain_until": int(task["retain_until"]),
            },
            "commit_decode": {
                "wall_s": official_decode_s,
                "device": "gpu",
                "promoted_speculative_state": promoted_vae_state,
            },
            "vae_preview": {"wall_s": preview_decode_s, "device": "gpu"},
            "preview": {"wall_s": finished - decoded_at},
        }
        return preview, stages, finished - started

    def close(self):
        if self.native is not None:
            self.native.close()
        self.pending_vae_state = None
        self.conditions.clear()
        self.backend.close()


def create_worker(args):
    return WanCFPPWorker(args)
