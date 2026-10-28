# Checked builder power — existing 30.1 / AR7.8 component

The builder power gap identified during the preceding order repair is now closed
for registered step pipelines. No new checklist row; broader builder/effect work
remains unchecked. Inventory stays **210 checked / 58 unchecked**.

`CoreOps.pow`, VarHandle `**` and reflected `**` lower through the existing binary
step. Native `core.pow` delegates numeric evaluation to the same checked expression
kernel, preserving i64 integer powers and validating f64 real powers. Node/edge
maps accept matching maps or either scalar side; scalar pairs stay scalar and can
be explicitly broadcast. No numerical algorithm is implemented in Python/FFI.
Atomic support follows the existing property-only staging contract. Alias/operand
encoding accepts current IR/legacy fields and legacy base/exponent spelling;
scalar inference recognizes power before arithmetic fusion.

## Verification

Three native tests cover node/edge maps, exact integers above 2^53, fractional and
negative exponents, scalar operand order/result identity, overflow/invalid real/
nonfinite/nonnumeric rejection, missing/mixed map inputs, empty outputs,
cancellation and unchanged earlier result variables. Twenty-five rebuilt Python
cases cover legacy/IR, optimization/local atomic modes, selected graph scope,
explicit/reflected operators, exact property round trips, scalar fusion/broadcast,
and failure preserving graph/history/IDs and discarding staged prefixes.

Final gates: **1,413 native passed / 56 ignored**, **2,144 root Python passed /
1 skipped**, **204 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The final in-tree
CPython 3.9 full-ABI release extension was rebuilt and atomically staged; three
existing unrelated FFI warnings remain. No packages or external graph libraries
were installed. Owner benchmark/review work remains excluded from staging.

Guide: `docs/api/builder-power.md` (three executed examples). Native gates use
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target` and offline locked cargo.
Python gate covers root test files, excluding the owner's separate algorithm
benchmark, plus executable API documentation.

## Bounded release evidence and limits

Reproducer: `cargo bench --offline --locked --bench builder_power` with the same
target environment. Five samples validate every 3^9 integer output over 1k/100k
node-map rows. Median registered native step cost: **0.0625 ms / 10.73 ms**.
Graph construction/history acceptance and Python marshalling are excluded; map
sorting/allocation/replacement is included. Receipt:
`../benchmarks/builder-power-2026-10-27.json`. No before/after, RSS, large-exponent
throughput or production claim is inferred.

Integer nonnegative powers reject i64 overflow or exponents outside u32; they do
not silently promote/round. Negative exponents/floating operands use checked real
f64 and reject nonfinite inputs/results. Zero to zero is one. Complex powers are
not introduced. Empty/missing map-key policy follows existing binary steps.

Other unfinished math facades, alternate batch-plan certification, full typed
effects/alias/lowering and compound topology acceptance remain their existing
30.1/AR7.8/09.4b.2 scope. The preceding order receipt's power-gap observation is
historical and superseded by this implementation.
