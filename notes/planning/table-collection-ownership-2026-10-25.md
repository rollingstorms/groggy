# Collection ownership closeout — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. Completed the supported collection
selection/schema/callback/join ownership inventory below. No checklist rows added.
The remaining table scope is broader graph/table conversion adapters, not these
finished ownership, reduction, generic join or specialized merge families.

## Fixed completed inventory

| Component | Exit / evidence |
| --- | --- |
| Shared context | Private Arc context retains an empty prototype, source captures, declared group names and observed numeric/boolean roles. No synthetic property rows or graph publications. |
| Empty table selections | take/skip/filter/native select_tables retain schema/source ancestry with zero tables. Native predicates run once per actual table. |
| Empty grouping | Core grouping and both Python wrappers, including node/edge adapters, retain source and declared keys. |
| Schema transforms | select/drop/rename work on zero-table schemas; renamed group metadata follows physical aliasing and collisions reject. Unknown fields and duplicate projections reject. |
| Row transforms | head/tail/sample retain context; real empty groups preserve group count and known column roles. |
| Callbacks | map/apply execute only on real tables; empty output schema is not guessed by invoking callbacks on prototypes. Source ancestry survives imported callback outputs. Python filter callback/truth exceptions propagate. |
| Pairwise joins | Real BaseTable column planning is reused for empty schemas, column roles and group aliases. Both source contexts survive; invalid/malformed metadata cannot be hidden by empty selection. |
| Checked outputs | Aggregates/flatten retain context sources without output row writeback identity. Missing specs against a retained schema reject even with zero tables. |

Source payloads remain immutable and alive through supported transformations.
Role inference uses actual observations when available; known roles survive
removal of all observations, while new/unobserved columns retain the existing
all-null/unknown policy. Context preserves source ancestry, including captures of
inputs later excluded by table selection, just as frozen views retain sources.

## Native compatibility

TableArray now has private context; external Rust struct literals must use
new/from_tables/from_tables_with_keys/from_group_columns instead. Public tables
and group_keys fields remain, with metadata validation at checked boundaries.
Direct clearing/replacing raw fields is not equivalent to supported ownership
transformations. Python signatures remain unchanged. The native with_empty_source
adapter accepts zero-row schema/source metadata and rejects property rows.

## Verification

- Eight native regressions: source lifetime and immutability, take/skip/filter,
  empty grouping, declared keys, role/schema propagation, invalid-metadata
  laundering, key/projection/rename checks, zero-column and row-empty outputs,
  callback counts/imports, empty joins, and the legacy empty-source bridge.
- Ten Python cases exercise core and legacy group wrappers, specialized node/edge
  adapters, callback error propagation, empty maps, schema operations and captures.
- Full native: 1,079 passed / 51 ignored, including the final unknown-role and
  callback-empty refinement.
- Root Python API suite: 1,812 passed / 1 skipped, rebuilt in-tree extension.
- 109 executable documentation examples, including three ownership blocks.
- Strict all-target core Clippy, scoped rustfmt, Python AST and diff checks.
- In-tree macOS extension linking uses dynamic Python symbol lookup; no wheel or
  package installation. Three preexisting unrelated FFI warnings remain.

## Performance receipt

`benches/table_collection_context.rs` records take(0)+flatten, empty schema
projection/rename+sum, and filter-all+sum at 1,000 / 10,000 rows and 100 groups.
Five optimized native observations include context/role preparation and outputs;
source/group fixtures and correctness checks are excluded. Receipt:
`notes/benchmarks/table-collection-context-2026-10-25.json`.

| Input rows | Take empty + flatten ms | Empty projection ms | Filter all + sum ms |
| --- | ---: | ---: | ---: |
| 1000 | 0.0840 | 0.0356 | 0.0876 |
| 10000 | 0.1906 | 0.0282 | 0.1909 |

Context preparation/validation may scan cells. Selected table clones still copy
owned columns; empty prototypes contain empty arrays and shared source captures.
No O(1), matched speedup, RSS/allocation or Python/FFI latency claim is established.
Retained payload lifetime is intentional and follows the final context owner.

## Checklist accounting

191 checked / 77 unchecked entries: 63 numbered rows and 14 overlapping
architecture rows; nine numbered rows are explicitly deferred. Of the numbered
rows, 54 are not explicitly deferred. These are overlapping parents/partial gates,
not 77 independent implementations. This pass completes the fixed ownership family
inside existing parents; broader conversion/legacy view and portable storage
requirements keep their original checklist entries open.
