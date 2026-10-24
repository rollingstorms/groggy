# 09.4b.2e / 27.2b — Checked table-to-graph materialization

Removes the legacy row-lookup/warning conversion recorded in the writeback audit.
It follows A03.1's accepted publication migration and preserves undirected default
materialization while making direction an explicit option.

## Implementation

- Replace whole-table cloning and repeated per-ID row lookup with columnar staging
  into one native GraphBatch. Fresh dense IDs follow row order; properties and
  topology are accepted in one state. Empty results remain state 0. Both native
  and Python factories now return clean publishing graphs.
- Share strict structural-column readers with table writeback. Reject malformed,
  duplicate and dangling IDs instead of using filtered tuple readers that can
  silently omit rows. Native table loop/parallel-edge policy remains enforced;
  undirected reversed pairs share the same pair identity.
- Register external IDs through batch identity bindings, including one global
  node/edge namespace. Preserve inline typed properties and remap NodeArray and
  EdgeArray references through source-row/destination-ID maps. Missing referenced
  entities reject. No unsupported pooled reference is copied to another pool.
- Remove meta-to-base fallback and warning-only property failures. Missing source
  pool contents reject explicitly. Invalid kind properties and identity conflicts
  propagate from preparation before destination entity allocation.
- Preserve null edge-ID assignment after the maximum declared ID, with checked
  arithmetic/range. The explicit auto_assign_edge_ids helper shares that routine;
  both paths avoid the old per-null row search and overflow. Materialization never
  repairs the source table in place.
- Ordinary table null cells become Null. Null storage identity/managed-ID/labels
  cells represent undeclared metadata; labels remain absent because Null is not
  a legal graph labels value. Projection's pre-existing null/absence ambiguity is
  documented rather than described as a fully lossless graph/history roundtrip.
- Tables carry no direction metadata. Python adds directed=True with an unchanged
  undirected default; Rust exposes to_graph_with_config. The FFI only translates
  the option and result. Borrowed graph construction retains its existing GIL
  boundary; no blanket GIL/FFI latency improvement is claimed.

## Scope

This accepts active table data as a new graph with new acceptance clocks. It does
not restore source history, inactive entities, pinned states, group revisions or
pool contents. Those remain 10.3/12.2/26/27.1. Native default/infallible APIs,
mutable storage escapes and whole-pipeline publication still keep 09.4b.2 open.

The native release benchmark uses 1k/10k node tables, descending sparse source IDs,
directed chain edges and one/four typed property columns per component. Five
observations per case exclude fixture construction and verify topology, exact f64
values and one clean accepted state. The receipt is
`benches/table_materialization_2026-10-24.json`. No before/after speedup, RSS/
allocation, FFI or production budget follows from those local observations.

See [API and executable example](../../docs/api/table-materialization.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **929 passed,
  52 ignored**, zero failures across native/unit/integration/doc suites.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed, including the new benchmark.
- Rebuilt the in-tree development extension through offline locked `cargo rustc`
  with the existing macOS dynamic lookup linker options. The three pre-existing
  FFI warnings remain unchanged.
- **28 new Python materialization cases**; full root API gate excluding the
  unrelated owner algorithm benchmark: **1,063 passed, 1 skipped**. The separate
  legacy `tests/modules/` survey is not included in this API gate.
- **35 documentation examples passed**, including materialization with explicit
  direction and managed IDs. Storage examples used a temporary directory.
- Four optimized benchmark cases, five samples each, verified clean single-state
  graphs, mapped topology and exact typed results. Raw receipt retained.
- `git diff --check` passed. Unrelated owner notes/benchmark data/tests remain
  outside the change.
