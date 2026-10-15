import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from speculative_sampling.models.h3 import H3SpeculativeSamplingLiveSession
from speculative_sampling.models.h3.deployment import SPEC as H3_SPEC
from speculative_sampling.models.h3.sol_h3_vae import (
    configure_vae_backend,
    h3_frames_from_latents,
)
from speculative_sampling.runtime.deployment import load_deployment
from speculative_sampling.runtime.web_app import Demo
from speculative_sampling.runtime import web_app


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("latents", "frames"), [(2, 5), (5, 17), (7, 22), (37, 124)])
def test_h3_latent_frame_mapping(latents, frames):
    assert h3_frames_from_latents(latents) == frames


@pytest.mark.parametrize("latents", [0, 1, 3, 4, 6, 8, 36])
def test_h3_latent_frame_mapping_rejects_non_native_lengths(latents):
    with pytest.raises(ValueError, match="5\\*n"):
        h3_frames_from_latents(latents)


def test_original_backend_proxy_does_not_import_gpu_dependencies():
    calls = []
    engine = SimpleNamespace(generate=lambda **kwargs: calls.append(kwargs) or "ok")
    selected = configure_vae_backend(engine, {"backend": "original"})
    assert selected.vae_backend == "original"
    assert selected.available_vae_backends == ("original",)
    assert selected.generate(vae_backend="original", stream_vae=True) == "ok"
    assert calls == [{"stream_vae": True}]


def test_sol_h3_vae_deployment_contract_is_loadable():
    path = ROOT / "speculative_sampling/models/h3/configs/boundary_image_5s_b300_sol_h3_vae.json"
    spec = load_deployment(path)
    assert spec.document["decoder"]["backend"] == "sol-h3-vae"
    assert spec.document["decoder"]["available_backends"] == [
        "original", "sol-h3-vae"
    ]
    assert spec.document["decoder"]["sol_h3_vae"]["cuda_graph"] is True
    assert spec.document["decoder"]["sol_h3_vae"]["fa4_root"] == "${SOL_H3_FA4_ROOT}"


def test_deployment_rejects_unknown_vae_backend(tmp_path):
    source = ROOT / "speculative_sampling/models/h3/configs/boundary_image_5s_b300.json"
    document = json.loads(source.read_text())
    document["decoder"]["backend"] = "mystery"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="decoder backend"):
        load_deployment(path)


def test_session_locks_selected_vae_backend_into_worker_tasks(tmp_path):
    token = tmp_path / "token"
    token.write_text("x" * 32)
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference)
    args = SimpleNamespace(
        live_policy=H3_SPEC.policy,
        session_class=H3SpeculativeSamplingLiveSession,
        output=tmp_path / "sessions",
        token_file=token,
        enable_rtc=False,
        mock=True,
        preview_width=448,
        vae_decode="stream",
        vae_backend="sol-h3-vae",
        vae_backends=("original", "sol-h3-vae"),
        reference=reference,
        initial_budget=3.5,
        model="h3-test",
        generation_width=832,
        generation_height=480,
    )
    demo = Demo(args)
    demo.ready = True
    with pytest.raises(ValueError, match="not loaded"):
        demo.start({"prompt": "test", "vae_backend": "unknown"})

    state = demo.start({
        "prompt": "test",
        "mode": "image",
        "transport": "websocket",
        "duration": 5,
        "vae_backend": "original",
    })
    assert demo.status()["session_vae_backend"] == "original"
    assert demo.next_task()["vae_backend"] == "original"
    request = json.loads((tmp_path / "sessions" / state["session"] / "request.json").read_text())
    assert request["vae_backend"] == "original"


def test_browser_exposes_session_scoped_vae_selection():
    html = (Path(web_app.__file__).with_name("web") / "interactive.html").read_text()
    assert 'id="vaeBackend"' in html
    assert 'vae_backend:$("vaeBackend").value' in html
    assert '$("vaeBackend").disabled=running||starting' in html


def test_browser_exposes_data_driven_game_configuration():
    html = (Path(web_app.__file__).with_name("web") / "interactive.html").read_text()
    assert '<script src="/game-config.js"></script>' in html
    assert "window.SPECULATIVE_SAMPLING_GAME" in html
    assert "gameNextPrompts" in html
    assert "action.followups" in html
