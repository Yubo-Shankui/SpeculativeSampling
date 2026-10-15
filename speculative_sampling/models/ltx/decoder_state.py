"""State operations around the pinned Fangtai decoder, not another decoder."""
import importlib.util
from pathlib import Path
import sys


def streaming_decoder_class(repo):
    source = Path(repo) / "packages/ltx-dmd/src/ltx_dmd/streaming_decoder.py"
    name = "speculative_sampling_fangtai_streaming_decoder"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load native decoder at {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    class Forkable(module.ZeroLookaheadAVStreamingDecoder):
        def feed_variable(self, video, audio):
            # Shapes follow the native absolute mapping (bootstrap/partial
            # replay may differ). Algorithm and history settings stay fixed.
            if self.block_index == 0:
                self.expected_first_video_latents = video.shape[2]
                self.expected_first_audio_latents = audio.shape[2]
            else:
                self.expected_video_latents = video.shape[2]
                self.expected_audio_latents = audio.shape[2]
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
            )
            for name in ("_video_history", "_audio_history", "_mel_history",
                         "_previous_wave_sample", "block_index"):
                setattr(forked, name, getattr(self, name))
            # Events are diagnostics, not causal state; avoid unbounded history
            # copying across every speculative fork.
            forked.events = []
            return forked
    return Forkable
