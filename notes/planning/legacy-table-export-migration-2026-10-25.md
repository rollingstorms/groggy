# Legacy table export migration — 2026-10-25

Existing 26.1 / 27.1 / AR7.10 scope; no new checklist rows.

## Closed inventory

- Native Graph.table/nodes_table/edges_table now delegate to one immutable
  GraphView projection path in the existing Graph-owned tables module.
- Native legacy Subgraph exports capture the exact current selection. A mixed
  export uses one pin for node and edge components; stale membership rejects.
- Python full/constrained node and edge accessors share that same path. Separate
  FFI row-building implementations were removed, including stale-ID fallback
  rows, null endpoints and dropping explicitly present all-null columns.
- Python direct Graph/Subgraph and array/iterator export entry points delegate
  consistently. Capture occurs under the GIL; selected native projection runs
  after release. No Rc/RefCell graph borrow crosses that boundary.
- Outputs retain source captures and direct row IDs for existing checked updates.
  Separate exports remain independent captures; same numeric state is not enough
  for implicit mixed writeback. Dirty manual captures remain inspectable but
  read-only for publication. Import/bundle detachment semantics are unchanged.

Legacy connected Subgraph/accessor objects themselves remain connected; only
outputs are pinned here. This does not close all 26.1 live-view migration work,
change GraphTable node-subset edge retention, or certify pooled portability.
Constrained export order is now stable unique ascending IDs like other native
projections; stale IDs reject and explicit-null properties remain represented.

## Evidence and costs

Six native cases and twelve Python cases cover fresh and mixed capture identity,
exact/null properties, selected edge instances, current versus old values,
stale selection rejection, manual read-only captures, empty sources and mixed
publication through the existing checked coordinator.

API examples: `docs/api/legacy-table-exports.md`; existing source-pin contract
corrected in `docs/api/table-source-pins.md`.

`benches/table_export_capture.rs` records three current-cost observations per
operation at 1k/10k/100k parent nodes plus chain edges. Fixtures, reused-view
capture, assertions and output destruction are excluded. Complete fresh capture,
selection/projection/source binding are timed as applicable. Receipt:
`notes/benchmarks/table-export-capture-2026-10-25.json`.

At 100k nodes, medians were 190.42 ms fresh node export, 199.58 ms fresh bundle,
88.68 ms nodes from an existing view, 100.28 ms bundle from an existing view,
and 108.25 ms for a fresh selected bundle containing 1% of parent nodes.
These compare current capture/reuse paths, not previous-version speedup.
Fresh capture remains proportional to parent state and retains that payload;
this migration is not COW or O(1), and no RSS or fixed FFI budget is certified.
Reuse one read_view for repeated exports when the same state is wanted.


### Legacy table output pin migration — 2026-10-25

Closed existing native/Python Graph, connected Subgraph and full/constrained
accessor table export paths under 26.1/27.1/AR7.10. All share pinned projection,
source row identity, stable selected IDs and explicit-null preservation. Mixed
exports share one capture; stale selections reject; dirty working captures stay
read-only. Removed separate FFI row-building logic. Legacy source objects remain
connected; GraphTable edge-retention policy is unchanged.

Evidence: `legacy-table-export-migration-2026-10-25.md`; 6 native / 12 Python cases.
Gate: **1,114 native passed / 51 ignored; 1,855 root Python passed / 1 skipped;
121 executable docs passed; strict all-target Clippy passed**. Fresh capture costs
are proportional to parent state; current capture/reuse measurements are in
`notes/benchmarks/table-export-capture-2026-10-25.json`. No COW, O(1), RSS or fixed
FFI budget claim. No new rows: **192 checked / 76 unchecked**. Broad legacy-view
and composite adapter exits remain open.
