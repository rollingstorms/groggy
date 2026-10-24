# 09.4b.2c / 27.2a — Explicit mixed table property writeback

Implements the accepted A10.1/A10.2 direction: joins can repeat source identities,
ambiguous assignments require explicit resolution, and accepted mixed node/edge
updates form one atomic batch. This adds an explicit mapped path without claiming
that all tables already propagate source identity or immutable read pins.

## Work completed

- `TableWriteback` is a native request with explicit table-ID/destination-ID maps,
  selected node/edge columns and an optional shared effective timestamp.
- `GraphTable.update_graph` checks the destination state/clean head, every
  structural row, exact one-to-one mappings, active targets and mapped endpoints.
  IDs accept native full/compact integers. Invalid structural rows never pass
  through the legacy filtered ID/tuple readers.
- Repeated table IDs, repeated destinations and duplicate selected columns reject.
  No implicit last-row-wins or creation occurs after a join. Node rows resolve
  edge endpoints; undirected reversal is accepted and parallel edges stay distinct.
- All selected inline property values are staged into columnar `GraphBatch`
  assignments. One coordinator call validates/accepts the whole update. Failure
  leaves values, deltas, entity counters and versions unchanged. Empty selections
  validate but do not publish. Explicit calls require a clean head even in manual
  compatibility mode and do not change the graph's ordinary publication policy.
- Selected Null assigns Null, while omitted columns remain unchanged. The API
  rejects structural/system/managed-ID changes and graph-local reference values
  pending explicit reference remapping. f64 and temporal properties remain typed.
- The FFI constructs the native request and returns its state ID; no graph logic
  was moved into Python or FFI. This borrowed graph mutation retains the GIL, as
  existing graph setters do. GIL-free table numerical/grouping operations keep
  their existing behavior.
- An explicit effective timestamp applies to all assignments; omission uses one
  acceptance clock. Per-assignment times remain available through GraphBatch.
  Checkpoint labeling remains separate from accepting this mutation.

## Remaining scope / audit findings

- These tables are owned data and do not carry a verified origin graph/pin through
  joins. Caller-supplied mappings assert the chosen target; they are not a proof of
  origin. 27.1, connected table views and AR7.10 therefore remain open.
- `GraphTable.to_graph` still uses legacy row lookup/warning paths that can lose
  meta/pooled properties; it requires a separate checked materialization repair.
- Reviewed ordinary algorithm score/label/distance output paths already use one
  bulk column write. This does not certify whole multi-step pipelines as atomic.
- Shared subgraph node setters still silently filter inactive targets before
  validation, unlike edge setters. Tightening that failure boundary remains in
  05.3/09.4b.2; it is not masked by the new table path.
- The planner allocates structural-ID/mapping validation sets and selected owned
  property columns. It performs no per-cell graph mutation. No FFI timing,
  memory-budget or throughput guarantee is inferred from functional tests.

See [API and executable example](../../docs/api/table-writeback.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **920 passed,
  52 ignored**, zero failures. Four new native cases cover mixed mapping/Null,
  late rejection without consumed IDs, duplicate rows/reference rejection and
  strict full/compact structural IDs.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed.
- Rebuilt the in-tree development extension via offline locked `cargo rustc`
  with the existing macOS dynamic lookup linker options. Its three pre-existing
  FFI warnings remain unchanged.
- **21 new Python writeback cases passed**. Root Python API gate excluding the
  unrelated owner algorithm benchmark: **1,035 passed, 1 skipped**. This does not
  include the separate legacy `tests/modules/` survey.
- **33 documentation examples passed**: the existing 32-example API/read-view
  set plus the new mapped mixed-update example. Storage examples ran in a temporary
  directory.
- `git diff --check` passed. No unrelated owner development notes, benchmark data
  or benchmark test changes were staged.
