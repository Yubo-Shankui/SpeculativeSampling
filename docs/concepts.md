# SpeculativeSampling concepts

A session contains an immutable delivered prefix and a replaceable generated
future. Generation always produces one complete native proposal. The controller
may retain a safe prefix of an older proposal, discard its suffix, and attach a
new proposal generated with the latest condition.

The core distinguishes four frontiers:

- **played**: acknowledged by the renderer;
- **committed**: released to transport and no longer replaceable;
- **generated horizon**: native output exists;
- **media-ready horizon**: decoded output can actually be played.

A model adapter defines which boundaries are native-safe and supplies evidence
that continuation state and media cover exist there. The controller never
inspects tensors or assumes that native proposal size equals delivered output
size.

This distinction covers retained-KV models and boundary-image models with the
same state machine. For example, H3 samples 37 causal groups for a five-second
proposal, but a continuation publishes 36 new groups after removing its repeated
boundary image.
