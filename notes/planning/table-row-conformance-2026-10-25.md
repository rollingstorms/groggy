# Table row conformance — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. This closes the fixed generic row family
below; broad table parents remain partial. No new checkbox rows are added.

## Completed inventory

| Component | Contract / verified exit |
| --- | --- |
| Shared keys | Joins, groups, dedup and set operations use one typed tuple-key extractor and column validator in BaseTable's private row module. No debug-string/delimiter identity. |
| Both grouping APIs | Stable first-key occurrence order, stable rows per group, identical logical key equality, exact retained row identity. Existing distinct collection wrappers and empty-key behavior preserved. |
| Deduplication | First row per key, exact source IDs retained; null/storage/NaN/zero identity matches joins; empty subset and zero-column rows explicit. |
| Union/intersection | Stable distinct rows, identical ordered schemas, both source contexts retained without guessed write identity. |
| Stack/concat | Explicit ordered row append or positional column combination; zero-column row counts preserved, concat padding/order fixed, generated-name collisions reject. Native contracts now have thin Python stack/concat adapters. |
| BaseTable group aggregate | Uses shared stable groups; sorted aggregation columns, rejects group/aggregate output collisions and invalid columns/functions even for empty input. Existing checked reduction kernels retained. Ungrouped aggregate output columns also sorted. |
| Python execution | BaseTable group_by/groupby/groupby_single, dedup, union/intersection and stack/concat release the GIL over native work. Existing group_by_agg already did so. |

One private `row_ops` module owns the mechanisms, alongside the existing private
join module. There is no new public table type. Selected group/dedup rows can
write through the normal checked source coordinator; aggregate and combination
outputs keep read-only source contexts and require explicit identity mapping.

## Verification

- Seven native regressions: storage representations, logical types, canonical
  float identity, delimiter-collision regression, stable groups/sets, empty and
  zero-column cases, repeated/missing keys, concat collision/order/padding,
  grouped aggregate preflight and exact selected source writeback.
- Independent equality oracle: 144 small input patterns for each of union,
  intersection and stack (432 output comparisons), with duplicate/null and
  Int/SmallInt representations.
- Ten Python cases exercise both group wrappers, sets, new combination adapters,
  validation, aggregate rows and graph source/writeback boundaries.
- Full native: 1,054 passed / 51 ignored. One old ignored grouping doctest was
  removed from its obsolete location; executable docs now cover the public API.
- Root Python API suite against rebuilt in-tree extension: 1,777 passed / 1 skipped.
- 100 executable documentation examples, including three new row-operation blocks.
- Strict all-target core Clippy, scoped rustfmt, Python/stub AST and diff checks pass.
- No package/wheel installation; in-tree macOS FFI artifact uses dynamic Python
  symbol lookup. Three preexisting unrelated FFI warnings remain.

## Current native costs

`cargo bench --offline --locked --bench table_rows` records five observations per
size in `notes/benchmarks/table-rows-2026-10-25.json`. Two-column fixtures have 100
repeated keys. Full output construction is included; fixtures and correctness
checks are excluded.

| Input rows | Group ms | Dedup ms | Union ms | Intersection ms | Stack ms | Concat ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.4215 | 0.0858 | 0.1625 | 0.1943 | 0.0307 | 0.0176 |
| 10,000 | 1.7835 | 1.0391 | 2.0610 | 2.4621 | 0.3190 | 0.1434 |

These are medians of current optimized native observations. No matched speedup,
RSS/allocation or FFI-latency claim. Hashing includes cell content; groups/indexes,
row plans, owned outputs and retained snapshots consume memory. No streaming or
total allocation budget is established.

## Remaining existing scope

Grouping/dedup/set/stack/concat and the generic join engine are complete for their
specified contracts. Specialized GraphTable domain merges/remapping, TableArray
aggregation and group-key policies, and broader conversion adapters remain under
27/AR7.10. Legacy connected reads remain under 26.1. Existing broad checklist
count stays 191 checked / 77 unchecked; these completed families reduce the
remaining implementation inventory without manufacturing parent completion.
