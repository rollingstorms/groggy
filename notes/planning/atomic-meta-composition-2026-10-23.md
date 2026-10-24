# 09.4b.2a — Atomic compound meta/collapse publication

Continues the accepted A03.1 ordinary publication migration, without a new
contract approval. The parent 09.4b.2 remains partial.

## Implementation

- Modern strict/lenient plans and legacy trait collapse routes share one Rust
  planner. It checks selection membership and source/target mappings, evaluates
  all node/edge aggregations, and stages the entire topology before mutation.
- The graph coordinator prepares the full batch and acceptance clock before
  installing a retained membership reference. Pool reference preparation checks
  content collisions and reserves capacity without inserting group contents.
  The accepted batch creates the meta-node with canonical identity, its properties
  and boundary edges, and removes originals if requested. One accepted version
  contains the entire operation. Manual mode uses the same prepared storage
  executor and leaves changes pending for explicit publication.
- Rejected input leaves IDs, active data, retained group contents, deltas and
  versions unchanged. Capacity reservations are allowed; process allocation aborts
  or corrupted storage through native mutable escape hatches are outside this
  guarantee. No full graph clone/rollback was added.
- Boundary planning preserves incoming/outgoing directions and actual parallel
  copies, instead of grouping all edges only by external neighbor. It respects
  count/threshold and meta-edge suppression policies. Explicit meta-edge selection
  now rejects before mutation because the API supplies no selector.
- Sorted node/edge processing defines first/last order. Compact text concatenates
  correctly; distinct concatenation is sorted. Strict missing-source calls now
  actually fail. Native explicit defaults apply only to missing observations.
- Standalone native meta-edge helpers also evaluate attributes before accepting
  a complete edge batch. FFI remains translation only, with its existing collapse
  error class preserved.

## Performance scope

Boundary planning scans active edges once and sorts by ID for deterministic
aggregation, replacing a full stored-edge scan for each selected node. It stages
boundary property maps/columns and selected-node values, so temporary memory is
not constant. The canonical removal/publication paths still inspect incident
edges for selected nodes; this change makes no overall linear-time promise.
The native release benchmark `benches/meta_composition.rs` records chain graphs
at 1k/10k nodes, selecting one/half the nodes, with extract/collapse strategies and
five observations per case. Fixture construction is excluded. It verifies one
new state and expected topology. The receipt is
`benches/meta_composition_2026-10-23.json`; it is not a before/after speedup, FFI
latency, RSS/allocation measurement or production budget.

## Remaining work

- 09.4b.2 still covers native default/infallible compatibility, mutable storage
  escapes, algorithm and table compound writeback.
- 09.4b.2b records pre-existing composer gaps: real external path contraction for
  `contract_all` (still aggregate compatibility), preset/custom identity versus
  labels, and an explicit meta-edge selector. Canonical node identity stays meta.
- Stored groups retain membership here, not complete pinned property state.
  Group revisions and portable pooled history/persistence remain their own work.
- Read-only snapshots remain usable across the accepted collapse; this does not
  migrate every legacy connected view to the pinned-state contract.

See [API semantics and executable example](../../docs/api/meta-composition.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **916 passed,
  52 ignored**, zero failures across native/unit/integration/doc suites.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed, including the new benchmark.
- Rebuilt the in-tree development extension with offline locked `cargo rustc`
  and the existing macOS dynamic lookup linker options. Three pre-existing FFI
  warnings remain; this step adds none.
- Root Python API suite, excluding the unrelated owner algorithm benchmark:
  **1,014 passed, 1 skipped**. Includes 12 new atomic-collapse cases and the existing
  retained-group/branch cases. The separate legacy `tests/modules/` survey is not
  part of this API gate.
- **32 documentation examples** executed, including the new directed collapse
  example. Storage examples ran in a temporary directory.
- Optimized benchmark: eight workload cases, five samples each. At 10k graph
  nodes, selecting one node measured medians of 0.465ms extract / 0.490ms collapse;
  selecting 5k nodes measured 1.156ms extract / 104.519ms collapse. These local
  observations highlight remaining canonical cascade/publication incident scans;
  they do not establish universal budgets or isolate each internal routine.
- `git diff --check` passed. Unrelated owner development notes/benchmark data and
  tests were left outside this change.
