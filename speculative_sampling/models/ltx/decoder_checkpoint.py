"""Portable causal state for the pinned native streaming decoder.

Only bounded histories and constructor parameters are serialized. Model
objects, CUDA pointers, diagnostic event history, and thread state never cross
the worker boundary.
"""
from __future__ import annotations

import hashlib
import json


CONFIG_FIELDS = (
    "video_history_latents",
    "audio_history_latents",
    "vocoder_history_mel",
    "value_ramp_ms",
    "expected_first_video_latents",
    "expected_video_latents",
    "expected_first_audio_latents",
    "expected_audio_latents",
)
CAUSAL_TENSOR_FIELDS = (
    "_video_history",
    "_audio_history",
    "_mel_history",
    "_previous_wave_sample",
)


def _cpu_clone(value):
    return None if value is None else value.detach().cpu().clone()


def export_decoder_state(state):
    config = {name: getattr(state, name) for name in CONFIG_FIELDS}
    for name, value in config.items():
        if value is not None and type(value) not in (bool, int, float):
            raise TypeError(f"decoder config {name} is not scalar")
    return {
        "schema_version": 1,
        "config": config,
        "block_index": int(state.block_index),
        "causal_tensors": {
            name: _cpu_clone(getattr(state, name)) for name in CAUSAL_TENSOR_FIELDS
        },
    }


def restore_decoder_state(decoder_type, models, checkpoint, *, device):
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "schema_version", "config", "block_index", "causal_tensors"
    }:
        raise ValueError("invalid decoder checkpoint structure")
    if checkpoint["schema_version"] != 1:
        raise ValueError("unsupported decoder checkpoint schema")
    config = checkpoint["config"]
    causal = checkpoint["causal_tensors"]
    if set(config) != set(CONFIG_FIELDS) or set(causal) != set(CAUSAL_TENSOR_FIELDS):
        raise ValueError("decoder checkpoint fields do not match the locked schema")
    if type(checkpoint["block_index"]) is not int or checkpoint["block_index"] < 0:
        raise ValueError("invalid decoder block index")
    state = decoder_type(*models, **config)
    for name in CAUSAL_TENSOR_FIELDS:
        value = causal[name]
        setattr(state, name, None if value is None else value.to(device))
    state.block_index = checkpoint["block_index"]
    state.events = []
    return state


def checkpoint_fingerprint(checkpoint, torch_module):
    """Hash exact causal bytes plus scalar configuration and block frontier."""
    metadata = {
        "schema_version": checkpoint["schema_version"],
        "config": checkpoint["config"],
        "block_index": checkpoint["block_index"],
    }
    digest = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":")).encode())
    for name in CAUSAL_TENSOR_FIELDS:
        digest.update(name.encode())
        value = checkpoint["causal_tensors"][name]
        if value is None:
            digest.update(b"none")
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.view(torch_module.uint8).numpy().tobytes())
    return digest.hexdigest()
