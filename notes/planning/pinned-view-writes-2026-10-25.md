# Checked pinned-view writes — 2026-10-25

Closes original item **26.2**. Supports existing mutation/view architecture work
09.4b.2 / AR4 / AR7.5; those broader parents remain open. Inventory is now
191 checked / 77 unchecked. No new checkbox row.

## Contract and implementation

- Current clean Graph read pins carry in-process history identity, branch and
  expected accepted state. Arc-owned provenance follows copies, filters and
  nested selections without attaching a mutable owner or breaking Send/Sync.
- Graph.apply_view_batch and its schema variant prepare one existing GraphBatch
  against that source. Manual ordinary policy does not defer this explicit checked
  publication. Existing scalar/row owner APIs keep their existing costs/policies.
- Scope checks cover properties, identities, removals and new-edge endpoints. A
  node deletion rejects if any incident edge is outside the selection, including
  excluded parallel or boundary instances. New entities are explicit additions.
- Existing batch/schema/identity/time preparation is authoritative. The output
  state/adjacency and surviving selection are constructed before execution using
  that prepared overlay and the history allocator's next acceptance ID. No
  fallible snapshot capture runs after publication.
- Old views stay unchanged. The result retains surviving selected members and
  surviving new entities, excluding unrelated owner edges and transient removals.
  The result can be used for the next checked write; old source heads conflict.
- Foreign histories, different branches (even equal numeric heads), stale/dirty
  graphs, dirty captures, standalone history pins and unrecorded from_snapshot
  materializations reject. Empty batches retain the existing checked rejection.
- A fresh current capture validates raw storage indices/endpoints and compares
  logical values/topology with the source pin, detecting untracked changes after
  capture. This does not certify raw native escapes made before capture; complete
  storage-escape migration remains in 09.4b.2.
- DurableGraph exposes the same capture/write path. Preflight precedes append;
  installed and stricter schemas both apply; one shared helper preserves the
  existing append/sync/execute/poison-on-execution-error behavior. Ordinary journal
  records retain the batch and times, not serialized view capabilities. Recovery
  restores accepted values; a reopened store requires fresh provenance.
- Python FFI only marshals existing batch/schema inputs and result fields. Write
  ownership retains the GIL; frozen read kernels retain their prior GIL release.
  Generator-backed Graph/DurableGraph methods and the GraphView stub are updated.

These are implementation choices under the already accepted immutable-view and
checked-publication decisions, not new quoted owner answers. The public API uses
the working Graph as mutation owner and returns a new view; it never mutates the
old snapshot. Fresh retrieval, shared read copies and independent materialization
are documented separately.

## Verification

Ten native regressions cover both directions and ordinary publication policies;
mixed add/property/identity/edge-removal/time operations; exact parallel/loop scope;
cascade rejection/acceptance; no consumed IDs on failure; dirty/foreign/branch and
historical source pins; raw post-capture changes/corrupt indices; schema/duplicate/
empty failures; transient lifecycles; immutable read-worker traits; 80 chained
updates compared with actual independently captured owner state; durable no-append
rejection, recovery and reopened-owner conflicts; unrecorded snapshot rejection.

Sixteen Python cases cover these public boundaries, including installed durable
schema enforcement and immutable outputs after close. Three executable API
examples and the existing documentation corpus pass. Generator-produced stubs and
Python sources parse; no supported public name was removed.

Full gates:

- `cargo test --offline --locked --quiet`: 1,030 passed / 52 ignored.
- In-tree FFI build, staged local sources/extension; root Python: 1,746 passed /
  1 skipped (owner benchmark test and legacy modules survey excluded as before).
- 91 executable documentation examples.
- `cargo clippy --offline --locked --all-targets -- -D warnings`: passed.
- Scoped Rust formatting and diff checks.

## Bounded optimized cost

`benches/pinned_view_writes.rs`: chain with half-graph selection, f64 score and
eight-element embedding properties; five observations per case. Complete current
capture/comparison, normal preparation, scope checks, output pin/adjacency,
execution and publication measured. Initial pin/fixture setup and exhaustive
output verification excluded. Median milliseconds:

| Owner nodes | Property batch | Mixed batch | Scoped cascade |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.8440 | 0.9435 | 1.3401 |
| 10,000 | 17.5147 | 10.9546 | 16.1073 |

Receipt: `../benchmarks/pinned-view-writes-2026-10-25.json`. Property batches change
half the nodes; mixed batches add one node/edge and remove one selected edge;
cascades delete the selected interior while retaining its two boundary nodes.
All returned full read payloads are compared with actual accepted owner captures.

This explicit API owns a fresh full read payload, with cost proportional to
current values/topology, selected membership and changes. Older pins retain their
buffers until released. It is not an O(1) scalar mutation promise or an extra cache
on ordinary writes. No matched speedup, allocation/RSS, durable-fsync, FFI latency
or broad scale guarantee is established. Legacy Subgraph/table migration (26.1,
27), portable meta payloads and effective-time reconstruction remain open.
