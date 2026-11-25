# Portable filtering and ordering steps — 2026-11-25

Finite component of existing 03.2, 09.4b.2, 30.1 and AR7.8. All four filtering
module steps now share memory/durable selected reads. Stable flat counts remain
217 checked / 51 open; broader execution parents remain partial.

## Completed component checks

- [x] Port node and edge predicate masks without changing entity membership.
- [x] Port attribute sorting and owned-source top-k with stable ID rank ties.
- [x] Preserve typed equality, exact mixed numeric ordering, containment/range
  predicates, comparable-domain checks and null/missing placement.
- [x] Read pending properties and selected surviving/provisional rows through the
  shared executor, including nested ordinary bodies.
- [x] Preserve complete-result installation, zero-k validation, empty/read-only
  no-op history and full rejection/allocation rollback.
- [x] Compare memory, worker, expected values and effective-time reopening, with
  selected-outside isolation and parallel/self-loop edge instances.

## Contract and implementation

Owner adapters delegate to the sealed read kernels; predicate and checked-sort
implementations remain unchanged. No filtering algorithm moved into FFI and no
new query/index policy is introduced. Predicate results are owned ID-to-one maps,
not graph selections or destructive deletes. Sort/top-k return ID-to-rank maps.
Only explicit attachment changes graph properties.

Missing attributes are skipped by predicates; explicit null is a typed value for
equality/inequality. Equality remains typed. Numeric ordering/range preserve
mixed integer/floating comparison. Sorting requires one comparable numeric/text/
Boolean domain; null/missing values sort last in either direction and ties use
ID order. NaNs and mixed incompatible domains reject. Top-k zero still validates
all inputs before producing an empty result.

## Evidence

`tests/durable_filtering_steps.rs` runs every kernel in nested ordinary bodies,
comparing exact results above f64 integer precision, worker execution, memory and
reopening. It covers selected sparse IDs, nulls, pending creation/weights,
parallel/self-loop edges and outside-selection preservation. Negative tests
reject mixed domains even at zero-k without journal/ID consumption. Read-only
sorting/filtering makes no retained version.

`tests/test_durable_filtering_steps.py` covers both sort directions and k limits,
all six typed comparison predicates, containment/range/empty matches, exact mixed
numeric comparisons, pending edge values and late rejection. The executable
contract is in [durable pipelines](../../docs/api/durable-pipelines.md).

Local qualification passes **1,687 native tests** (54 ignored), **3,558 packaged
Python tests** (one skipped, 23 expected warnings) and **300 documentation
examples**. Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax
and all 62 packaged Python/stub byte comparisons pass. All 500 production source
files match runtime `8c29687d`.
See [local receipt](../benchmarks/portable-filtering-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-filtering-wheel-qualification-2026-11-25.json).

The build cache and earlier external Python environment disappeared during
qualification. A cold native rebuild and an isolated ignored Python 3.9 environment
restored the gates; the receipt records its actual interpreter build. No new
platform or performance claim is made for this adapter migration.

## Remaining scope

Other producers/containers, retained history, edge-aware registers, normalization
options, fixed-child/provisional topology and remaining alternate IR/streaming
routes stay in existing execution parents. A14 naming/cache/history and A15
production budgets remain unanswered.
