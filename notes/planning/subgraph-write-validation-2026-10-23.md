# 09.4b.2d / 05.3a — Preserve complete subgraph write inputs

Closes the concrete stale-target filtering finding recorded during the table
writeback pass. No additional owner contract decision is needed: invalid bulk
writes already reject as a whole at the graph coordinator.

## Implementation

- Remove inactive-node filtering from native multi-column and single-column
  subgraph setters. Delegate the original columns to the existing owner validator,
  retaining original input positions for inactive, unknown and duplicate rows.
- Algorithm attribute attachment already uses those setters for both node maps
  and native node columns. It now propagates rejection instead of accepting only
  surviving targets. No algorithm or error translation logic moves to FFI.
- Change the native all-edges convenience setter from per-edge scalar calls to
  one columnar edge update. All edges validate before any property write, and a
  successful publishing call produces one version for the complete column.
- Existing all-node, selected-node and multi-property convenience setters inherit
  whole-operation validation. Valid empty calls remain no-ops; ordinary manual
  calls keep deferred publication. Rejected calls preserve active properties,
  pending state and published versions.
- Read behavior, retained groups and explicit restoration requirements do not
  change. Python accessors already validate supplied IDs and remain compatible.

## Evidence and scope

Four new native regression tests exercise publishing/manual bulk/column writes,
original-position diagnostics, stale node/edge convenience calls, successful
single-version bulk operations, empty calls, and algorithm attachment through both
maps and columns. They use real removals after constructing the selection.
No success can be inferred solely from the retained selection's ID set.

The release receipt `benches/subgraph_bulk_writes_2026-10-23.json` covers native
chain graphs with 1k/10k nodes, node/edge convenience writes and five observations
per case. Fixture construction is excluded. Each accepted workload verifies one
new state and a typed property result. The prior edge helper retained one version
per edge, so histories differ; this receipt makes no before/after speedup, FFI
latency, allocation/RSS or production-budget claim. Bulk payload construction and
existing coordinator validation still allocate column-sized temporary data.

09.4b.2 and 05.3 remain partial: native default/infallible compatibility, mutable
storage escape hatches, whole multi-step pipelines and legacy table-to-graph
materialization still require their own repairs/audits. This step does not add
source-pin propagation to connected views or tables.

See [API behavior and executable example](../../docs/api/subgraph-writes.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **924 passed,
  52 ignored**, zero failures across native/unit/integration/doc suites.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed, including the new benchmark.
- Rebuilt the in-tree development extension via offline locked `cargo rustc`
  with the existing macOS dynamic lookup linker options. The three existing FFI
  warnings remain; no new FFI logic or warnings were introduced.
- Root Python API gate excluding the unrelated owner benchmark: **1,035 passed,
  1 skipped**. The separate legacy `tests/modules/` survey is not part of this gate.
- **34 documentation examples passed**, including the new stale-selection example.
  Storage examples executed in a temporary directory.
- Optimized `subgraph_bulk_writes` benchmark ran four cases with five observations
  each, verifying a single accepted version per bulk node/edge operation and the
  exact f64 property result. The raw receipt is retained beside its source.
- `git diff --check` passed. Unrelated owner development notes, benchmark data and
  benchmark tests were preserved outside this commit.
