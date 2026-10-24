# Atomic fused kernels and message-pass blocks — 2026-10-24

Closes **09.4b.2i**, extending **30.1b / AR7.8**.

## Implemented

- Certify fused AXPY, multiply-add and neighbor-product aggregation: they consume
  run variables, access only selected topology and return temporary variables.
  Invalid fused neighbor direction now rejects rather than choosing a default.
- Certify ordered message-pass blocks whose body operations are implemented.
  They share the outer staged property context via preceding loads and later
  attachments, including inside nested loops. No intermediate version is created.
- Deserialize block options from their JSON payload, honoring supplied direction,
  include-self and ordered flags. The old tagged wrapper silently used defaults.
- Build directional selected CSR once per block. Preserve edge multiplicity and
  count each stored self-loop once. Optional include-self adds one contribution
  only when no selected self-loop already exists. Temporary Gauss-Seidel updates
  are visible to subsequent nodes, with existing map/column traversal order.
- Check cancellation between node/body operations, integer overflow (including
  MIN/-1), and integer/floating zero divisors. Integer division keeps its existing
  truncating convention. Failure discards all staged writes in the outer run.
- Remove the repeated linear search for a target row during block writeback;
  update the known column position instead.

## Explicit remaining work

Atomic blocks reject streaming, unordered execution, unsupported operations and
legacy `where`, which currently returns the true branch without evaluating a
condition. The latter remains a known semantic repair under 30.1 / AR7.8.
Ordinary legacy blocks retain their placeholder where behavior. Correct options,
direction and arithmetic error handling apply to legacy blocks as well; directed
results can change where the prior implementation ignored the requested direction.

Atomic defaults, effect-aware optimization, alternate batch plans, remaining step
families and outer pipeline transactions remain open. Graphiti work stays deferred.
No new policy approval is required for this implementation slice.

## Performance scope and receipt

Each block constructs a selected directional CSR; repeated loop invocations rebuild
it. Direction-aware topology cache reuse remains an optimization. Fused steps keep
their native variable kernels. Runtime property buffering stays proportional to
touched cells. No whole-graph clone is introduced.

`benches/atomic_builder_blocks.rs` records four native release ring workloads,
five samples each, at 1,000/10,000 nodes and edges, five loop iterations, shared
staged attachments/loads and one accepted state. Fixture/spec construction is
excluded; execution includes topology preparation, staging and final publication.
All output cells and clean publication are checked. Raw receipt:
`notes/benchmarks/atomic-builder-blocks-2026-10-24.json`.

Observed median milliseconds on this machine:

| Nodes / edges | Fused loop | Message-pass loop |
| --- | ---: | ---: |
| 1,000 / 1,000 | 3.379 | 4.733 |
| 10,000 / 10,000 | 23.597 | 39.933 |

These are bounded observations, not a matched speedup, FFI, allocation/RSS, memory
budget or production throughput guarantee.

## Verification

- Full offline/locked native suite: **940 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension using existing macOS dynamic lookup settings;
  existing three FFI warnings remain. No package installation.
- **36 focused atomic-builder tests passed**, including public message_pass,
  staged property loads across iterations, transient owner reads, fused arithmetic,
  in/out/both traversal, parallel edges/self-loops, early unsupported rejection,
  and rollback after block overflow. Native tests cover arithmetic boundaries and
  mode/operation capability checks.
- Full root Python API gate: **1,112 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **39 executable documentation examples passed**.
- Four release benchmark workloads passed all output/publication checks.
- `git diff --check` passed. Owner development/benchmark changes remain separate.
