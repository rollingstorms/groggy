# Frozen matrix exports — 2026-10-22

AR7.5c propagates the accepted immutable read state into owned numerical results.
It does not close AR7.5/AR4/26.1: existing table/query/algorithm adapters still need
read-owner migration. No workbook policy is selected by this change.

## Ownership and implementation

Native GraphView.to_matrix<T> and adjacency_matrix<T> read the shared pin and
selected membership. They never borrow the working graph or capture state again.
Live Graph exports and frozen exports call storage/matrix/graph_projection,
sharing scalar-column discovery, missing-value conversion, edge-instance handling
and the existing accurate matrix adjacency accumulator. Live graph reads still
resolve GraphSpace references against one borrowed pool, rejecting bad referents.
The frozen reader borrows attributes directly; discovering names does not clone
text, vector or other attribute payloads.

Results own their storage and row/column metadata. Python exports use f64; native
exports retain the existing NumericType/FromAttrValue bounds. FFI only marshals
arguments/errors and releases the GIL for Rust projection, passing owned Send
NumericalMatrixData back before constructing the Python matrix. This adds neither
a writer/concurrency policy nor history events/checkpoints.

## Shared public policies

- Rows follow sorted selected node IDs; feature columns follow sorted names.
  A feature column requires at least one selected numeric scalar/Boolean value.
  Missing, null and nonnumeric cells are zero-imputed. Vector/text-only columns
  stay excluded. This preserves existing Graph export behavior and conversion
  precision; it does not redefine strict explicit-matrix input policies.
- Adjacency contains only selected edge instances, retains selected isolates,
  preserves direction, sums parallel contributions and counts each loop once.
  It supports dense and CSR output with node_<id> labels. A missing weight is
  one; explicit null is zero. Explicit numeric text uses existing parse rules.
- An existing encoding bug was exposed: pooled short text becomes CompactText,
  while FromAttrValue previously accepted only Text. Shared f64/i64/Boolean
  conversions now accept Text/CompactText/CompressedText. f32/i32 inherit these
  adapters. Bad numeric text and corrupted compressed data return errors.
- Empty views export 0x0; one node without scalar attributes exports 1x0 features.
  Later owner mutation/deactivation cannot alter captured values or matrix results.

## Verification

Full `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet` passes, including
four new native projection/encoding fixtures. Strict all-target core Clippy passes.
Rebuilt the in-tree native Python extension without package installation; existing
three FFI warnings remain. Combined numerical/matrix, read-view, temporal,
property, identity, table and atomic-write regressions: **671 passed, 1 existing
skip**. New Python tests cover dense/CSR, directed/undirected, scoped parallel
instances, labels, frozen feature discovery, zero imputation, text weights, empty
shapes and worker-thread exports. All three examples in
`docs/immutable-read-views.md` execute successfully.

Bounded native release timing receipts are in
`notes/benchmarks/frozen-matrix-export-2026-10-22.md`. They report observed costs,
not a speedup against the previous live exporter, FFI latency, memory budgets or
production scale guarantees.
