# Third-party notices

The implementation in this directory is adapted from NVlabs/Sana,
`models/streamrender_h3`, revision `62da34b14a53d7289e82fa56692f34e96e089bee`.

- The TAE-H3 implementation retains its MIT notice and originates from
  madebyollin/taehv as described in its source header.
- The remaining vendored StreamRender H3 files retain their Apache-2.0 source
  headers and are distributed under the upstream Sana repository license.

Only the dependency closure needed by the TAE encoder and causal ViT decoder
is included here. Model weights are not redistributed.
