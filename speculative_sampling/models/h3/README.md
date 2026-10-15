# MiniMax H3 live plugin

This package adapts MiniMax H3 to SpeculativeSampling without changing the model. Each
proposal is a native five-second I2AV generation. The adapter decodes the
selected boundary frame at full resolution and uses that RGB frame as the next
I2V condition; replacement begins on the following frame.

## B300 deployment profile

The default profile is
[`configs/boundary_image_5s_b300_sol_h3_vae.json`](configs/boundary_image_5s_b300_sol_h3_vae.json):

- 8×B300 Ulysses, four denoise forwards;
- 124 generated frames, 832×480, 24 FPS;
- full-resolution RGB boundary re-encode;
- Sol-H3-VAE streaming decode and H.264/Opus WebRTC by default;
- prompt-rewrite lead: 0.91 seconds;
- continuation lead: 0.91 seconds;
- EMA updates disabled;
- zero delivery-credit frames;
- `transport_paced` scheduling clock: RTC uses the backend's paced committed
  frontier, while WebSocket uses extrapolated browser feedback;
- near-horizon rewrite fallback enabled, using the maximum of the latest eight
  warm first-media-ready measurements plus one frame of guard.

At 24 FPS, a 0.91-second lead rounds upward to 22 frames (0.92 second). Rewrite
and continuation values are independent backend fields even though the current
web page intentionally updates both from one control. The authenticated
`/api/scheduling/lead` endpoint also accepts
`condition_lead_seconds` and `continuation_lead_seconds` separately, so a
future UI can expose two controls without restarting the service.

These numbers are measured deployment calibration, not model capabilities or
universal SpeculativeSampling defaults. A different GPU topology, resolution, codec, or
model build must be profiled and represented by a new versioned config.

## Scheduling behavior

RTC pulls exactly one frame on each backend 24 FPS tick. That irreversible
`committed` frontier drives both operations:

```text
prompt rewrite: q = committed + rewrite_lead_frames
continuation:    start when horizon - committed <= continuation_lead_frames
```

If a prompt arrives one scheduler tick late and the ideal `q` lies just beyond
the ready horizon, the adapter compares remaining ready media with recent warm
first-media-ready time. When production still fits, it clamps `q` to the
horizon and immediately generates the new prompt. It does not generate a full
old-prompt bridge merely to fill one or two missing frames. An already-running
continuation remains reusable cover; a bridge is retained only when the ready
horizon is genuinely too close.

Browser playback feedback is used for observability and history compaction,
not to authorize RTC commits. No prompt update can replace a frame already
committed by the sender.

## Launch

Set the checkpoint path in `${H3_MODEL_PATH}` and launch the service with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3_live_demo.py \
  --model "$H3_MODEL_PATH" \
  --deployment-config speculative_sampling/models/h3/configs/boundary_image_5s_b300_sol_h3_vae.json \
  --token-file results/h3/token \
  --output results/h3/sessions \
  --reference /path/to/reference.png \
  --generation-width 832 --generation-height 480 --proposal-frames 124 \
  --vae-decode stream --rtc-width 832 --rtc-height 480
```

Use a private random token and an operator-provided ICE/TURN configuration.
Never publish tokens in URLs, logs, configs, or documentation. Each run records
the deployment policy, code revision, source hashes, environment versions, and
model manifest identity in `run_metadata.json`.

## VAE backends

`decoder.backend` selects the video codec without changing H3 sampling:

- `original` keeps the released H3 VAE and the existing seven-latent windowed
  streaming adapter.
- `sol-h3-vae` uses StreamRender H3's TAE encoder for boundary images and its
  two-latent causal ViT decoder on rank 0. The generated latent space remains
  H3's normalized 24-channel latent space.

The default config loads both backends and exposes a session-start selector in
the browser; the selected backend is fixed for that session and recorded in
`request.json`.
Set `SOL_H3_TAE_WEIGHTS` and `SOL_H3_CAUSAL_DECODER_WEIGHTS` to the separately
released StreamRender-H3 checkpoints. This backend also needs the upstream
FlashAttention-4/CUTLASS DSL environment used by the causal decoder. The
default profile selects `sol-h3-vae`. Use
[`configs/boundary_image_5s_b300.json`](configs/boundary_image_5s_b300.json) for
the original-VAE-only fallback. WebRTC is enabled by default; pass
`--disable-rtc` to use WebSocket only.

## Verify the installation

Run `pytest -q` before a GPU launch. In the browser, confirm that WebRTC
connects, playback crosses a natural continuation without interruption, and a
prompt change replaces only future video while the committed prefix continues
playing.
