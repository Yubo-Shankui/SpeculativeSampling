# Architecture

The repository has three deliberately separate layers.

```text
speculative_sampling/
  core/                 model- and transport-independent algorithm
  runtime/              HTTP, WebSocket, RTC, media and workers
  models/                model-native continuation and sampling plugins
```

## `speculative_sampling/core`: the algorithm

`speculative_sampling` is a deterministic, CPU-testable state machine. It owns prompt
versions, replaceable future proposals, immutable delivered prefixes, branch
selection, refill, recovery, and provenance. It imports no model framework,
network server, codec, browser, or deployment code.

The public extension points are exported by `speculative_sampling` and implemented in
`speculative_sampling/core`. Logical positions are model-independent delivery groups. A
model adapter maps those positions to
native latents, frames, KV blocks, boundary images, or another continuation
representation.

The backend owns the irreversible frontier and advances it from the media
sender clock. Browser playback reports are telemetry and compaction evidence;
they never authorize delivery or refill. A condition update projects that
frontier forward by the measured production lead, selects the first valid
model boundary, keeps the old branch flowing to that boundary, and discards
only the replaceable suffix. If a short native proposal does not yet reach the
projected boundary, the controller extends the old replaceable branch with a
`bridge`; the new condition remains pending and wins as soon as the safe point
exists.

## `speculative_sampling/runtime`: optional systems code

`speculative_sampling/runtime` contains worker supervision, RPC/tensor transport, media types,
production-time estimation, journals, and launch configuration. A
reader interested only in SpeculativeSampling does not need to read or install it.

WebSocket, WebRTC, HTTP authentication, browser playback, and cluster launchers
belong to this layer or to applications, never to the core controller.

## `speculative_sampling/models`: plugins

Each model owns its native continuation and output mapping:

- `speculative_sampling/models/h3`: non-invasive full-resolution boundary-image continuation.
- `speculative_sampling/models/ltx`: retained-KV continuation with native ReRoPE and
  split-service decoding.
- `speculative_sampling/models/wan`: CF++ retained-KV continuation with clean accepted-prefix
  commits.

H3 is the supported runnable release. LTX and Wan are published as integration
code while deployment tuning continues; their release configs and launch
instructions are intentionally not included yet.

Retained-KV models expose a resident iterator. Each call supplies only
`retain_until`, a ready condition handle, its version, and a seed. The iterator
keeps official KV, speculative suffix state, RNG/position state, and native
cache rules such as LTX ReRoPE. Accepting a prefix clean-commits exactly that
prefix; rejecting a suffix cannot contaminate official state. Native latent
groups remain atomic for branch selection even though RTC sends their decoded
RGB frames one at a time.

Model packages do not import one another. New integrations implement the same
public contracts and can be developed without changing the controller.
