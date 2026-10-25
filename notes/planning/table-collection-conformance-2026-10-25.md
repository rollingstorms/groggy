# Table collection conformance — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. This closes reduction/group-key validation
and supplied-table flattening in the existing TableArray family. No new checkbox
rows; broad table parents remain partial for specialized merges and broader
collection ownership/conversion adapters.

## Completed fixed inventory

| Component | Contract / exit |
| --- | --- |
| Native reduction coordinator | sum/mean/count/min/max/all and explicit agg share checked column reductions; default numeric methods exclude boolean/text and group keys. |
| Group metadata | Exact key/table count, consistent sorted key names, constant physical keys verified using property identity. Projected-away keys may remain metadata; malformed/stale metadata rejects. |
| Output schema | Group keys emitted once, sorted; defaults preserve input non-key order, explicit specs sorted; key/spec collision and unsupported/missing-everywhere specs reject. |
| Group/global semantics | Grouped: one row/table. Native/core ungrouped: one row across all observations with correctly weighted means. Legacy Python constructor keeps per-table behavior; columns= is projection, not grouping. |
| Numeric and logical kernels | Shared streaming checked array sum/mean/extrema; original mixed numeric scalar extrema, checked counts, logical non-null nunique, fully validated boolean all. All-null/empty reductions Null, counts zero. |
| Flatten | Union of supplied column schemas, stable order, null padding, checked row totals, zero-column row counts and no loss of later-only columns. |
| Source captures | All supplied table captures retained read-only in aggregate/flatten outputs; no guessed output writeback identity. |
| Python adapters | Both group wrappers and node/edge adapters delegate reductions to Rust with GIL released. Removed Python-side reduction/eligibility business logic; edge group metadata now survives delegation. |

Implementation: private `collection_reductions.rs` owned by TableArray, plus one
crate-only iterator adapter for the existing array extrema kernel and an internal
BaseTable collection-output constructor. No new public collection type. Existing
TableArray public fields remain, and corrupted metadata is checked at reduction.
`TableArray::from_group_columns` performs native validation for the legacy bridge.

Explicit specs support sum, mean/avg, count, min, max, nunique and all. Numeric
specs reject nonnumeric/nonfinite observations rather than silently dropping them.
Logical unique counts retain property equality, including canonical NaN identity.
Boolean all validates later cells even if an earlier observation is false.

## Verification

- Seven native regressions: key emission, strict defaults, weighted global means,
  integer cancellation across chunks, bounded f64 means, precision-preserving
  extrema, null/empty groups, logical uniques, boolean/nonfinite rejection,
  malformed metadata/specs, heterogeneous flatten, zero-column shape, distinct
  source captures and direct grouped row oracles over 29 fixture sizes.
- Twelve Python cases exercise both group wrappers, native/global versus legacy
  per-table constructors, projection, counts/nulls, strict errors, flatten,
  immutable sources and specialized edge group metadata.
- Full native: 1,061 passed / 51 ignored.
- Root Python API suite against rebuilt in-tree extension: 1,789 passed / 1 skipped.
- 103 executable documentation examples, including three collection examples.
- Strict all-target core Clippy, scoped rustfmt, Python/stub AST and diff checks pass.
- In-tree macOS FFI artifact uses dynamic Python symbol lookup; no wheel/package
  install. Three preexisting unrelated FFI warnings remain.

## Current native costs

`cargo bench --offline --locked --bench table_collections` records five optimized
observations per size in `notes/benchmarks/table-collections-2026-10-25.json`.
Two columns, 100 existing groups. Metadata validation and full result construction
are timed; fixture/group creation and correctness checks are excluded.

| Input rows | Sum ms | Mean ms | Count ms | Unique ms | Flatten ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.0714 | 0.0685 | 0.0650 | 0.0928 | 0.0435 |
| 10,000 | 0.1430 | 0.1391 | 0.1241 | 0.2756 | 0.1228 |

These are current-cost medians with no matched speedup, RSS/allocation or FFI
latency claim. Native reductions stream columns without flattening. Unique counts
allocate a set; outputs own columns and retain source payloads. The legacy Python
bridge clones BaseTables before creating its native collection, which adds copy
cost outside these measurements. No whole-reduction 100ns or large-scale claim.

## Remaining existing scope

Specialized graph-domain merge/remapping, collection ownership when transformations
remove every table, and broader conversion adapters remain under 27/AR7.10.
A zero-table collection has no input capture/schema to recover; this slice retains
captures belonging to supplied tables, including supplied empty tables. It does
not certify every legacy collection transform or merge merely because reductions
now share a coordinator. Checklist stays 191 checked / 77 unchecked.

Follow-up: [specialized merge conformance](graph-table-merge-conformance-2026-10-25.md)
closes all six graph-table merge strategies and federated naming. Broader collection
ownership/conversion and existing pooled portability work remain open.
