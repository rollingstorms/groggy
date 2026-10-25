# Table join conformance — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. The five-family generic join engine is
complete for the contract below. No checklist rows added; broad parents remain
partial because specialized graph merges and other table adapters are separate.

## Fixed implementation inventory

| Component | Status / boundary |
| --- | --- |
| Shared core | Complete. One Rust hash-key planner and columnar materializer replaces five duplicated generic join paths; public single-key methods delegate. |
| Typed key identity | Complete. AttrValue Eq/Hash replaces debug strings, covering storage variants, typed values, canonical signed zero/NaN and null-null matches. No numeric coercion across property types. |
| Composite keys | Complete. Ordered, nonempty, unique key lists with equal arity; existing Python string/list/dictionary syntax delegates to core. |
| Ordering and unmatched rows | Complete. Stable left-major matches/unmatched rows, followed by right-only rows for outer; right retains swapped-left behavior. Outer shared keys coalesce on right-only rows. |
| Output schema | Complete. Existing family naming policies preserved, column order retained, generated-name collisions reject before row expansion. Zero-column cross products retain row count. |
| Source provenance | Complete for generic joins. Existing exact-capture source dedup remains; joined rows have no automatic graph write identity. |
| Execution adapters | Complete for this family. Python releases the GIL over native join execution. Native TableArray positional single-key joins support all five families, including empty-input family validation. |

Implementation lives in `src/storage/table/joins.rs`, privately owned by the
BaseTable module. This separates the mechanism without adding another table type
or public module. The Python bridge marshals key lists, family names and errors;
no join execution logic lives in FFI. Invalid family names retain ValueError;
other native join failures retain RuntimeError. The Python call shape is unchanged.

## Verification

- Seven native regression tests, including a nested-loop oracle over 75 small
  input-pattern combinations for each of four keyed families (300 comparisons).
- Storage variants (small integers, compact/compressed text and compressed vectors),
  signed zero, NaN payloads, distinct numeric/property types, null keys,
  duplicate fanout, composite mismatches, missing/empty/repeated keys, collisions,
  empty schemas, zero-column products and five native TableArray families.
- Eight Python cases cover public composite syntax, deterministic full outer
  rows/coalescing, collisions without input mutation, invalid keys and immutable
  source capture retention after owner updates.
- `cargo test --offline --locked --quiet`: 1,047 passed / 52 ignored.
- Root Python API suite against rebuilt in-tree extension: 1,767 passed / 1 skipped.
- 97 executable documentation examples, including three new join examples.
- Strict all-target core Clippy and scoped rustfmt/diff checks pass.
- In-tree macOS extension linked with dynamic Python symbol lookup; no wheel
  installation. Three preexisting unrelated FFI warnings remain.

## Current native costs

`cargo bench --offline --locked --bench table_joins` records five observations
per size in `notes/benchmarks/table-joins-2026-10-25.json`. Inputs have duplicate
keys internally but disjoint key ranges, so inner produces zero rows and both
outer variants produce twice the input row count. Fixture construction and
correctness checks are excluded; full join output construction is timed.

| Rows per input | Inner median ms | Outer median ms | Composite outer median ms |
| --- | ---: | ---: | ---: |
| 1,000 | 0.1727 | 0.2520 | 0.4223 |
| 10,000 | 1.5071 | 2.4349 | 4.1153 |

These are current native cost observations, without a matched baseline or speedup,
RSS/allocation, FFI latency or streaming/scale guarantee. The eager row plan and
result can grow with duplicate fanout. Row-plan reservation and cross multiplication
are checked, but no total allocation budget is promised.

## Remaining existing work

The generic join family no longer appears as remaining join-key work. Specialized
GraphTable domain merges/remapping, grouping and set-operation key semantics,
TableArray aggregation/group-key policies and other conversion adapters remain
under existing 27/AR7.10; legacy connected reads remain under 26.1. This pass
neither expands those requirements nor claims broader table completion.
