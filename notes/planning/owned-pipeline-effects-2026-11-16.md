# Owned pipeline effects — 2026-11-16

Completed bounded extraction within 09.4b.2, AR3, 30.1 and AR7.8. No checklist
rows added or broad parents closed.

## Implementation

PropertyBatch::prepare consumes staging and produces an internal PreparedEffects:
the complete WriteCoordinate, an optional canonical GraphBatch, and constant-size
selection projection ranges. It moves values and explicit clocks, drops read pins,
root owner references and pending caches, and does not allocate accepted IDs, sample
a default clock or publish. It is not a public durable-execution API.

Local and outer atomic execution continue through accept_selection. The receiver
validates owner/branch/head and cleanliness, even for empty effects; nonempty
payloads use Graph::apply_batch exactly once. Projection retains the original
selected survivors plus global added positions. Empty/property-only runs retain
the original continuation. Ranges replace temporary added-ID vectors at this
boundary; no graph, history or writable scratch clone is introduced.

## Verification

Five new native regressions cover value-buffer movement and release of read/owner
references; preparation without publication/allocation; exact mixed topology and
typed value projection; explicit versus publication-default clocks; stale owner,
branch, head and dirty-state rejection for empty/nonempty effects; discarded plans
and late canonical validation failure; and unchanged prior immutable captures.

- `cargo test --offline --locked`: **1,615 passed / 54 ignored / zero failed**,
  225 suite results (shared target, incremental compilation disabled).
- Targeted PropertyBatch suite: **18 passed**.
- Strict all-target Clippy passes; Rust formatting and whitespace checks pass.
- No FFI/Python public surface changes. Artifact-backed qualification follows the
  next implementation slice and will retain its exact tested source.

## Remaining boundary

The durable pipeline read adapter still requires work. Algorithms/StepScope retain
a Subgraph paired with a mutable owner, whereas DurableGraph intentionally lends
only shared graph reads. The extraction makes payload conversion independent of
that machinery; it does not bypass journaling, expose a writable durable owner,
or implement safe durable pipeline dispatch. See
[boundary context](durable-pipeline-boundary-2026-11-16.md).
