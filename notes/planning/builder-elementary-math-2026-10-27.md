# Checked elementary builder math — existing 30.1 / AR7.8 component

Implemented the existing abs/sqrt/exp/log and element-wise min/max facade methods
for registered step pipelines. No rows added or closed; the fixed inventory stays
**210 checked / 58 unchecked**.

## Shared native implementation

- One private UnaryMathStep in the existing arithmetic module delegates numerical
  work to the canonical checked expression evaluator. Scalars, node maps and edge
  maps preserve kind. Inputs are processed in deterministic ID order, complete
  outputs replace variables only on success, and cancellation is checked before,
  during and after map work. Property publication remains the existing overlay.
- Integer absolute values remain exact i64 and reject minimum-i64 overflow.
  Sqrt/exp/log use checked finite real f64; invalid domains/nonnumeric/nonfinite
  values/results reject. Natural log is default. Explicit bases validate before
  execution (including empty inputs): numeric, finite, positive and not one.
- Min/max extend the existing binary executor with the canonical exact numeric
  comparator. No conversion of large integers to f64 for ordering. Numeric ties
  choose the left value and retain its type; scalar pairs remain scalar. These
  are element-wise operations, not reductions.
- Legacy and IR lowering now handle all six operations. IR distinguishes binary
  extrema from reductions and infers map/scalar kind correctly for subsequent
  fusion. Python/FFI contains no numerical algorithm. No new core module is added.

Guide: `docs/api/builder-elementary-math.md` (three executed examples).

## Verification and evidence

Three native tests cover scalar/node/edge kind, exact integers, logarithm bases,
real domains, invalid controls on empty inputs, cancellation, output preservation,
exact mixed-number extrema and left ties. **68 new Python cases** cover legacy/IR,
optimization/local-atomic modes, selected scope, public methods, atomic failure,
scalar fusion/broadcast, large integers and extrema dtype identity.

Final gates: **1,416 native passed / 56 ignored**, **2,212 root Python passed /
1 skipped**, **207 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The in-tree CPython
3.9 full-ABI release extension was rebuilt and atomically staged; three existing
unrelated FFI warnings remain. Native commands use offline locked cargo with
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`. No package installation
or external graph library. Owner benchmark/notebook/review files are excluded.

An initial concurrent-build Python run reported the existing sparse Cholesky GIL
worker-progress assertion (2,211 passed / 1 failed). All five isolated GIL tests
and the complete suite passed after builds finished, with no matrix/test change.
This observed scheduling sensitivity remains evidence under the existing 29.26
GIL/performance gate, not a new checklist row or a claim of a matrix repair.

Reproducer: `cargo bench --offline --locked --bench builder_unary_math` with the
same target environment. Five release samples validate every output for all four
unary kernels at 1k/100k node rows. Median 100k step costs: abs **9.30 ms**, sqrt
**9.10 ms**, exp **8.92 ms**, base-4 log **9.68 ms**. Receipt:
`../benchmarks/builder-elementary-math-2026-10-27.json`. Construction/history and
Python marshalling are excluded, map allocation/sorting/replacement included.
These are bounded absolute observations, not before/after/RSS/production claims;
extrema are not separately profiled by this unary receipt.

## Remaining fixed scope

Other unfinished facade methods, alternate batch executors, complete effects/
alias/lowering and certified numeric folding/fusion domains remain 30.1/AR7.8.
Compound topology acceptance remains 09.4b.2. This completes the named elementary
methods without a blanket claim that every builder API or optimization is done.
