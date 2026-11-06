# Typed scalar predicate lowering — 2026-11-05

Continues existing **29.6 / 29.26 / AR7.8**, following the measured 21.453 ms
million-row lazy i64 chain. No checklist row is added.

The validated query AST now lowers to a numeric predicate plan. i32/i64/f32/f64
column dispatch happens once; rows no longer construct dynamic AttrValue wrappers
or repeatedly classify target literals. Numeric cells in heterogeneous AttrValue
columns use the relevant integer/floating plan, with all other domains on the
original generic path. usize retains checked value conversion. The same existing
exact scalar comparator evaluates numeric predicates; no numeric comparison math
is reimplemented and no unsafe casts or new dependencies are introduced.

The authoritative scalar ConversionPolicy::Exact performs opportunistic computational threshold conversion:
integer to floating with a checked round trip, or finite integral floating to an
in-range integer. Original stored values and literal identity remain untouched.
Control validation, statement order, errors, source dtype and reservoir sampling
are unchanged. Numeric lowering allocates in proportion to query size, not rows.
Generic QueryEvaluator.evaluate remains an independent execution reference.

The differential suite compares integer, f32/f64 and heterogeneous collections
with the generic evaluator across wide integer/float boundaries, null/bool/text
mismatches, parity, length, constants, logical expressions, infinity and NaNs.
Primitive floating outputs are compared by bits, including signed zero,
subnormals and NaN payloads. Existing exact property predicate gates remain.

This does not close all numerical performance work or the full pipeline execution
boundary. Small-group captured collection projection costs remain documented in
`subgraph-array-projection-2026-11-05.md`.

## Gates and bounded release measurements

- Full native gate: **1,533 passed / 54 ignored**; strict root all-target Clippy
  passed. Release extension rebuilt in tree; three pre-existing FFI warnings.
- The unchanged release corpus --lazy-arrays was run with builds quiet.
  Raw per-case wall/allocation samples:
  `notes/benchmarks/typed-scalar-lowering-release-2026-11-05.json`;
  aggregate process CPU/RSS: `typed-scalar-lowering-process-2026-11-05.txt`.
- One million rows, median wall compared with the prior checked interpreter:

| Operation | Before | After |
| --- | ---: | ---: |
| i64 lazy chain | 21.453 ms | 8.497 ms |
| AttrValue lazy chain | 24.265 ms | 14.672 ms |
| f64 filter | 26.159 ms | 14.787 ms |

The i64 chain is approximately **2.5x faster**, while its measured additional
peak live requested bytes remain **8,002,473**. Allocation count rises from 45 to
47 for query-sized numeric plans and remains independent of row count. AttrValue
plans add four calls (45 → 49), also independent of row count. No per-row heap
allocation is introduced for numeric values. Source copying, compilation and
output assertions are included; fixture construction is excluded.

The direct eager i64 reference is **0.742 ms** in this run, so a substantial
interpreter gap remains. These are bounded scalar-interpreter/ownership receipts,
not SIMD, streaming, JIT or general production numerical certification.

Rebuilt Python compatibility gate: **2,591 passed / 1 skipped**, plus **257
executable Python documentation examples**. No Python facade or persisted format
changes are introduced by this optimization.
