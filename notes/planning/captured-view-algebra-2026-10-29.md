# Captured membership algebra and grouping — 26.1 / AR7.5 / 21.2

Completed the composition/grouping read dependency of the connected ownership
migration. GraphView supports exact membership union/intersection/difference,
node-membership similarity, native ordered overlap intersections and typed node/
edge grouping. Results share the original capture; no writable owner, Graph clone,
new state, group revision or retained-history copy is introduced.

## Existing defects repaired through the shared kernel

- Native subtraction no longer leaves edges whose endpoints were removed.
- Native connected algebra rejects unrelated owners with overlapping numeric IDs.
  Captured algebra requires the same capture; independent captures remain distinct
  even if weak current-state caching happens to share a frozen payload.
- Python's merge/intersection/subtraction methods no longer claim the existing
  native algebra is unimplemented; thin adapters call it.
- Connected node/edge grouping and membership algebra use the canonical captured
  kernel, retain their connected return type and preserve descriptive type metadata.
  Unique captured membership moves into connected results rather than being cloned.

Union/intersection operate on exact selected edge instances, not newly induced
edges. Difference retains graph-valid endpoint closure and isolates. Similarity
retains the existing four node-set definitions and zero-empty-denominator rule.
Grouping retains canonical typed equality/order, exact integer distinctions,
explicit-null groups, omission of absent keys and selected edge boundaries. Node
groups examine member incidences rather than rescanning every edge per group.
Edge groups retain only their grouped instances/endpoints.

## Verification and measurements

Five native integration cases include exhaustive pairs of all valid selections
of a four-node multigraph with a loop, frozen property reads through mutation/
owner drop, workers, scope/capture checks and connected adapters. Eleven Python
cases cover captured/connected operations, typed grouping, missing/null, workers,
exact parallel identity, owner rejection and nonpublication. Existing API examples
and regressions are retained; no old test is weakened or removed.

`benches/captured_view_algebra.rs` reports cold capture separately from five
samples of frozen grouping, connected grouping and three exact set operations.
Raw local release results: `notes/benchmarks/captured-view-algebra-2026-10-29.json`.
The fixture is a 1k/10k-node chain with one integer grouping key and 1/100/N groups.
This measures the new supported paths and preparation trade-off, not an old/new
speedup, logical heap, Python FFI budget or production guarantee. Captured groups
share frozen state; connected adapters temporarily capture current state and
therefore pay cold capture if no other reader retains it. Large typed keys can
make comparison expensive; no universal complexity/throughput claim is implied.

Final native/Python/docs/Clippy gates are recorded in the priority-window log.

## Fixed remainder

This completes captured composition/grouping, not default connected ownership.
The existing 26.1/AR4/AR7.5 boundary still includes connected columns/accessors,
traversal/structural reads, read/write contexts and default return migration.
Legacy filter return ownership (21.2), compound topology/custom effects (09.4b.2)
and their original policy boundaries remain open. These are dependencies inside
the existing inventory, not new checklist rows: 211 checked / 57 unchecked.
