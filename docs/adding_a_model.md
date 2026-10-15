# Bring your own model

Speculative Sampling is training-free: integrating a model changes serving
state and scheduling, not model weights or the training objective. The shared
controller only needs to know which future outputs are still revocable. Native
latents, KV caches, positions, decoders, and distributed execution stay inside
the model plugin.

The sampler may be streaming or non-streaming. The serving prerequisite is
simply that it can produce usable future video faster than playback consumes
that future; no per-latent or per-frame streaming interface is required.

## 1. Choose the continuation path

Start by identifying the state from which the model can continue correctly.

### Boundary-image or boundary-latent continuation

Use this path when the model can begin a new native proposal from a decoded
image or latent at a valid boundary. H3 is the reference implementation:

- jointly sample a complete bidirectional proposal;
- publish decoded frames incrementally;
- preserve the irreversible prefix;
- choose an already-published RGB boundary and re-encode it;
- regenerate only the uncommitted suffix under the new condition.

Read [`models/h3/adapter.py`](../speculative_sampling/models/h3/adapter.py) for
boundary selection and output mapping, then
[`models/h3/session.py`](../speculative_sampling/models/h3/session.py) for the
controller-facing session.

### Retained-state continuation

Use this path when the model continues from causal KV, recurrent state, or a
native cache. Wan and LTX are the reference implementations. Keep two kinds of
state separate:

- **official state:** represents only the accepted prefix;
- **speculative state:** contains predictions that may still be discarded.

On every production command, clean-commit the accepted part of the previous
proposal, discard the rejected suffix, and sample one complete native proposal
from the retained boundary. Never promote speculative cache writes into the
official state before their outputs are accepted.

Read
[`models/wan/speculative_sampling_native.py`](../speculative_sampling/models/wan/speculative_sampling_native.py)
for a compact clean-commit example and
[`models/ltx/speculative_sampling_native.py`](../speculative_sampling/models/ltx/speculative_sampling_native.py)
for a more involved audio-video/ReRoPE path.

## 2. Create a self-contained model package

Use this layout:

```text
speculative_sampling/models/my_model/
  __init__.py
  adapter.py          valid boundaries and native continuation semantics
  output_mapping.py   native output → logical playback groups
  session.py          controller-facing live session
  worker.py           resident model and decoder execution
  deployment.py       capabilities and measured scheduling policy
  configs/            versioned deployment profiles
  README.md            model-specific dependencies and launch contract
```

Model packages may import `core` and `runtime`, but must not import another
model plugin. Do not put model cache rules into the shared controller or
transport rules into the model package.

## 3. Declare model capabilities

Declare the facts that do not change between deployments with
`ModelCapabilities` and `SamplingCapabilities`:

```python
from speculative_sampling.core import ModelCapabilities, SamplingCapabilities

CAPABILITIES = ModelCapabilities(
    model_id="my-model-v1",
    conditioning={"text": True, "image": False, "audio": False},
    continuation_kind="retained_kv",  # or "boundary_image"
    execution="monolithic",
    joint_chunk_size=8,
    supported_parallelism=(1, 2, 4),
    source_geometry={"fps": 24, "resolution": [832, 480]},
    decoder_grouping={"kind": "native_causal", "delivery_groups": 1},
    valid_branch_policy="accepted_prefix_clean_commit",
    can_fork_generator_state=True,
    can_fork_decoder_state=False,
    can_split_decode=False,
    condition_readiness="encoded_handle_required",
    initialization_policy="text_only_then_steady",
    mapping_version="my-model-v1",
    sampling=SamplingCapabilities(
        mode="streaming", native_chunk_size=8,
        requires_refill=True, refill_chunk_size=8,
    ),
    evidence=("tests/test_my_model_adapter.py",),
)
CAPABILITIES.validate()
```

Keep measured latency, GPU topology, checkpoint paths, and transport
calibration in a versioned deployment config rather than in this declaration.

## 4. Define logical positions and output mapping

The controller uses absolute, half-open logical positions. Your mapper owns the
conversion between those positions and native latents, RGB frames, audio
samples, and playback time.

Each `OutputGroup` must describe:

- logical start and end;
- native source start and end;
- presentation start and end in seconds;
- whether audio is required;
- any later native group on which decoding depends.

Do not assume one latent equals one frame. Preserve the model's exact temporal
mapping, including asymmetric first frames, causal decoder windows, audio
tails, and overlap dependencies.

## 5. Expose valid replacement boundaries

A boundary adapter answers one question: given the desired cut, the available
candidate boundaries, and the current horizon, where is the earliest safe
replacement point?

```python
from speculative_sampling.core import ContinuationKind

class MyBoundaryAdapter:
    continuation_kind = ContinuationKind.RETAINED_KV

    def select_retain_until(self, requested, candidates, horizon):
        return next(
            (p for p in candidates if requested <= p <= horizon and self.valid(p)),
            None,
        )
```

Return `None` when no valid boundary exists. Do not silently round to an unsafe
point. If decoder state or retained media crosses the boundary, report that
dependency explicitly instead of treating it as ready.

## 6. Implement one resident generation transaction

For retained-state models, expose one model-owned operation:

```text
advance(retain_until, ready_condition, condition_version, seed)
  1. promote or clean-prefill the accepted prefix
  2. discard the unaccepted prediction suffix and speculative cache writes
  3. sample one complete native proposal from retain_until
  4. retain the new proposal and forked state for the next call
  5. return source data plus stage timing and readiness metadata
```

The controller index remains absolute for the entire session. Position IDs,
ReRoPE, rolling-window eviction, sink/recent retention, decoder caches, and RNG
state are private native semantics. Never reconstruct continuation state from
the browser playhead.

For boundary-image models, the equivalent transaction selects a published
full-resolution boundary image, encodes it, and starts the next native proposal
after the repeated anchor frame.

## 7. Connect the live session and worker

The runtime repeatedly calls the session's `plan()` method and sends the task
to a persistent worker. The worker reports source readiness; the session maps
that source into publishable groups and calls `complete()` when native
production finishes.

For retained-KV models, subclass
`runtime.retained_kv_session.RetainedKVLiveSession` and provide:

- `capabilities` and a measured `live_policy`;
- `chunk_latents`;
- `mapper_factory`;
- optional split-decoder boundary logic;
- exact RGB/latent boundary conversion.

For boundary-image models, copy the smaller H3 session shape and replace only
the native mapping and worker task fields. In both cases, the runtime owns HTTP,
authentication, WebRTC/WebSocket delivery, session history, and recovery
journals.

## 8. Calibrate the deployment

Measure these values on the actual hardware, resolution, proposal length, and
decoder backend:

- condition ready → first committable media;
- continuation dispatch → first committable media;
- per-group decoder readiness;
- media preparation and sender delay;
- supported GPU topology and native parallelism.

Store them in a new versioned deployment config. Do not copy H3's lead time to
another model: scheduling lead is deployment evidence, not an algorithmic
constant.

## 9. Validate in stages

Before a public launch, verify:

1. **CPU controller tests:** bootstrap, refill, latest-condition-wins,
   replacement near the horizon, cancellation, retry, and bounded history.
2. **Native parity:** uninterrupted output matches the model's original
   pipeline at the same seed and configuration.
3. **Branch correctness:** every partial accepted-prefix length preserves
   official state and discards the suffix without cache contamination.
4. **Output mapping:** frame count, audio duration, boundary conversion, and
   decoder dependencies are exact.
5. **Live transport:** the sender never exposes an uncommitted frame and no
   prompt update rewrites the irreversible prefix.
6. **Latency calibration:** measure warm P50/P90 first-presentation latency and
   underruns on the target deployment.

The integration is ready when a condition change can occur at every supported
native boundary without discontinuity, stale condition use, state corruption,
or playback underrun.
