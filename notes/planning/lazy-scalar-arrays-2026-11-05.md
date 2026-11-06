# Checked native lazy scalar arrays — 2026-11-05

Completes an existing scalar execution component of **29.3 / 30.1 / AR7.7 /
AR7.8**, with bounded allocation evidence under **29.6 / 29.26 / 32.3**.
No checklist rows are added and no broad parent closes.

## Demonstrated problems and implementation

The old native lazy filter ignored its query and selected AttrValue rows with a
random 70% heuristic; other element types passed through. The optimizer moved
skip/filter/sample/take/map/collapse operations across one another and merged
filters without preserving boolean grouping. Take/skip folding could overflow
and change results. Unsupported map/graph transforms returned fabricated empty or
unchanged data; an empty intermediate could hide an invalid later operation.

Execution now validates the full plan, compiles each scalar predicate once and
applies filters, take, skip and sampling in statement order. Filtering/slicing
reuse the owned input allocation. Explicit sample_seeded uses a private RNG;
unseeded sampling retains a random default. Primitive numeric types retain their
original values and dtype. Unknown types, invalid predicates and reserved graph
or type-changing operations reject even after take(0) or on empty input.

Array QueryEvaluator and BatchQueryEvaluator now share the graph lexer, numeric
literal parsing and boolean precedence/parenthesis grammar through typed token
adaptation. Quoted operators and AND/OR text remain literal. Array parity and
bare text compatibility are adapted as tokens, not raw string replacements.
Graph modulo predicates still reject. No row reparses a query. Array mathematical
numeric equality and infinity/NaN literals remain distinct from graph property
identity equality and finite-only graph literals. Text ordering remains lexical,
and mismatched array scalar domains remain nonmatches. Null and logical compact/
compressed lengths retain explicit behavior. Constructor signatures remain
infallible; validate/evaluate/collection expose compilation errors.

The old filter-size heuristic is replaced by a statement-order upper bound.
Exact filtered cardinality is obtained by collection. This is deferred vector
execution, not a streaming engine: ArrayOps.lazy_iter still copies its source.

## Evidence and remaining boundary

The focused native suite exhaustively compares all 512 three-operation chains
from the fixed eight-operation set with independent eager Vec semantics. It
covers scalar widths, wide integer/floating boundaries, graph ordering parity,
quoted tokens, constants, parity, invalid controls/empty branches, local seeded
sampling and explicit rejection of unsupported graph/type-changing operations.
Existing property_numeric_predicates tests protect mathematical equality and
nonfinite literal compatibility.

Native generic eager marker-trait methods, full typed graph/table intermediates,
streaming/cancellation and broader execution lowering remain in existing
30.1/AR7.8. No simulation or synthetic timing is reported as correctness evidence.
The release numerical corpus gains --lazy-arrays with real requested-allocation,
peak-additional-live-byte and wall-time samples, with source clone, compilation,
execution and assertions included. Fixtures are excluded; no RSS/FFI claim is made.

## Verified gates and measured limits

- Full native test gate: **1,530 passed / 54 ignored**, including the executable
  native documentation example; strict all-target root Clippy passed.
- Rebuilt release in-tree extension: maintained Python gate **2,581 passed /
  1 skipped**, plus **256 Python documentation examples**. Three pre-existing
  FFI warnings remain.
- Release corpus: `cargo bench --bench numerical_completion -- --lazy-arrays`
  (offline/locked with the shared target); timed executable with builds quiet.
  Raw wall/allocation samples: `notes/benchmarks/lazy-scalar-arrays-release-2026-11-05.json`.
  Aggregate process CPU/RSS: `notes/benchmarks/lazy-scalar-arrays-process-2026-11-05.txt`.
- At one million rows, median i64 chain **21.453 ms**, independent direct eager
  reference **0.776 ms**, AttrValue chain **24.265 ms**, f64 filter **26.159 ms**,
  seeded sample **1.634 ms**. i64 chain requests **45 allocations** at 1k, 100k
  and 1m rows, with **8,002,473 additional peak live bytes** at 1m: owned input
  copy plus about 2.5KB of plan overhead. AttrValue input is larger (48-byte
  values). Output vectors may retain input capacity; caller compaction is explicit.

The interpreted predicate path is roughly **28x slower** than its direct eager
reference on that fixture. This is correctness and bounded allocation evidence,
not numerical throughput certification or a speedup over the old random filter.
Typed predicate lowering/vector execution remains existing 29.6/29.26/AR7.8 work.
No further semantic reordering is justified by this performance gap.
