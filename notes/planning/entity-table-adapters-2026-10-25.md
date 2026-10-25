# Entity table adapter completion — 2026-10-25

Existing 27 / AR7.10, with structural reader consistency under AR2. No new rows.

## Closed inventory

- Native ID-mapped property replacement shares BaseTable preparation for node
  and edge columns. Table IDs must be checked and unique; unknown inputs and
  duplicate records reject; unspecified rows are null. All validation precedes
  output construction. Direct source row identity/capture is retained.
- Python NodesTable.with_attributes is enabled, and EdgesTable exposes the same
  native operation. Dictionary and id/value-record inputs marshal once; native
  validation runs with the GIL released. Existing positional call shape stays.
- Node IDs and edge IDs/endpoints share strict nonnegative integer extraction.
  SmallInt is supported consistently. Independent filtered-column compaction no
  longer misaligns rows. Incomplete/null edge IDs require explicit assignment.
- Native tuple extraction validates columns and zips once, replacing repeated
  membership/position searches. Endpoint filters retain physical row/source order.
- Option-returning row lookup/iteration retain skip-invalid semantics but no
  longer wrap negative IDs or ignore small integers.
- Python edge ID/source/target arrays retain exact int64 values above 2^53,
  correcting the previous float64 conversion. FFI tuples/readers/filter calls
  release the GIL. Stubs and executable examples reflect the corrected dtype.

Six native cases cover mapping/null/order/type behavior, invalid/duplicate IDs,
lookup/iteration, endpoint filters, exact wide IDs and source context. Thirteen
Python cases cover input marshaling, lossless IDs, failure behavior, retained
sources and explicit mixed node/edge publication as one delta with one supplied
effective timestamp. Existing source/writeback suites also certify stale/foreign
captures, schema rejection, duplicate destinations, exact structural identity,
read-only derived rows and synced durable recovery.

## 27.2 exit review

The accepted immutable source-bound table update contract is complete: direct
row identity survives supported table operations; explicit chosen columns publish
one checked mixed batch; old captures remain immutable; derived/combined rows
retain read-only sources and require explicit mapping before update. Attached
property columns now flow through this complete path from Python as well.

Mark 27.2 complete. Remaining legacy connected-view migration stays in 26.1;
composite conversion semantics stay in 27.1/AR7.10; portable retained pools/history
stay in 07.3/10.3/12.2. This completion does not claim those other exits.

## Performance and compatibility

`benches/entity_table_adapters.rs` records five optimized current-cost observations
at 1k/10k/100k rows. Fixture/entry construction, assertions and destruction are
excluded. Complete checked operations and output allocation are timed. At 100k
rows medians are 0.734 ms tuples, 12.298 ms ID-mapped attribute column and 1.107 ms
two-source filtering on this machine. Tuple work is now O(rows); ID-mapped
replacement is O(rows + entries), including copied output columns. No O(1),
matched speedup, RSS or FFI budget claim. Receipt:
`notes/benchmarks/entity-table-adapters-2026-10-25.json`.

Compatibility corrections: malformed structural readers now error instead of
silently dropping values; unknown attribute IDs now error instead of being
ignored; edge ID arrays now carry int64 rather than float64 values. General raw
tables remain inspectable through BaseTable. API contract/examples:
`docs/api/entity-table-adapters.md`.


### Entity table adapters and 27.2 completion — 2026-10-25

Completed node/edge ID-mapped property attachment, typed structural readers,
linear tuple extraction, exact Python integer ID arrays and source-aware
attribute publication. Unknown/duplicate attribute IDs reject; malformed
structural reads no longer compact rows independently. Existing source capture,
head, schema and durable publication gates are reused.

27.2 is now checked: the immutable source-bound explicit table update contract
is complete, including Python mixed property updates with one delta and explicit
effective time. Remaining legacy live-view migration and composite conversions
stay in 26.1/27.1/AR7.10; full retained-pool portability keeps its original items.

Evidence: `entity-table-adapters-2026-10-25.md`; six native and thirteen Python
regressions. Gate: **1,104 native passed / 51 ignored; 1,843 root Python passed /
1 skipped; 118 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/entity-table-adapters-2026-10-25.json`.
No new checklist rows: **192 checked / 76 unchecked** after closing 27.2.
