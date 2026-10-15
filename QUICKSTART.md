# Quick start

Install the CPU-testable controller and run the test suite:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

For the browser runtime, install `.[rtc]`; WebRTC is the default transport.
The H3 GPU demo additionally requires
[Sol-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/Sol-H3)
on `PYTHONPATH`.

Download the released four-step checkpoint and set `H3_MODEL_PATH`:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree \
  --local-dir checkpoints/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree
export H3_MODEL_PATH="$PWD/checkpoints/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
```

Two H3 deployment profiles are included:

- `boundary_image_5s_b300_sol_h3_vae.json` is the default and selects
  Sol-H3-VAE while exposing both VAE backends at session creation;
- `boundary_image_5s_b300.json` is the original-VAE-only fallback.

The Sol-H3-VAE profile uses the TAE and causal decoder released with
[StreamRender-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/streamrender_h3):

```bash
hf download Efficient-Large-Model/StreamRender-H3 \
  --include 'tae/taeh3_official.pth' \
            'causal_decoder/causal_vit24_step6000.safetensors' \
  --local-dir checkpoints/StreamRender-H3
export SOL_H3_TAE_WEIGHTS="$PWD/checkpoints/StreamRender-H3/tae/taeh3_official.pth"
export SOL_H3_CAUSAL_DECODER_WEIGHTS="$PWD/checkpoints/StreamRender-H3/causal_decoder/causal_vit24_step6000.safetensors"
export SOL_H3_FA4_ROOT=/path/to/Sol-H3/h3_runtime/third_party/sol_attn/_vendor
```

WebRTC and Sol-H3-VAE require no launch flags. Use `--disable-rtc` for the
WebSocket fallback, or select `original` in the browser when starting a
session.

The same H3 service exposes two built-in experiences:

- `/` is the free-form prompt interface;
- `/margit-duel` is the 10-action game interface.

Pass `--game-config examples/games/my_game.js` and open `/?game=1` to load a
custom action game. See [Create a New Interactive Game in Five Minutes](docs/creating_a_game.md).

The LTX and Wan integration code is included for study and extension, but both
are still being tuned. Their deployment configs and launch instructions are
not part of this release; H3 is the supported runnable path.
