# Builder operand and effect-order repairs — AR7.8 / 30.1

Continuation of the existing execution-contract priority after finite queries
(21.1) closed. No new checklist rows and no claim of complete typed effects or
compound topology acceptance. Inventory stays **210 checked / 58 unchecked**.

## Reproduced failures and changes

1. Default optimized `attach(score, 2); load(score); attach(copy, loaded)` moved
   the load before the first attachment, copying 7 instead of 2. Legacy lowering
   builds a value graph and appends output-free effects, so a later outer atomic
   run cannot repair already changed compilation order. Ordinary algorithm and
   sampler builds now retain the original program whenever an output-free effect
   precedes a later computation. A terminal output suffix still permits value
   optimization; explicit atomic builds continue to bypass optimization entirely.
2. Common-expression signatures sorted inputs, identifying subtraction/division
   with their reversed operands. `(10-2)+(2-10)` returned 16 instead of 0. Signatures
   now preserve ordered operands and structured metadata boundaries. Even
   commutative operations require a future explicit canonicalization proof.
3. Expression reuse now admits only listed value operations and clears its reuse
   cache at effects or uncertified operations. Graph/property reads and custom
   operations are not assumed reusable without source/effect identity. Execution
   and unknown-domain blocks are observable for dead-code elimination.

These are frontend compilation repairs; graph computation and property staging
remain in existing native kernels. Ordinary runs still accept writes separately;
explicit atomic execution remains required for all-or-nothing behavior. Transient
loads continue reading the original graph. The fallback deliberately forgoes
fusion on interleaved-effect programs until the IR represents their dependencies.

## Verification and bounded costs

**16 new Python cases** cover subtraction/division, all ordered signatures,
conditional branch positions, persistent/transient/local-atomic modes, default
compiled children within an outer atomic run, sampler order, execution-block
liveness and retained arithmetic reuse. Full root contract gate:
**2,119 passed / 1 skipped**, **201 executable documentation examples**. Python
syntax and diff checks pass. Native code is unchanged from the immediately prior
verified **1,410 passed / 56 ignored** and strict all-target Clippy baseline.

Reproducer: `python benches/builder_order.py`, using the in-tree release extension.
Five alternating optimize=False/True samples at 1k/10k nodes check all reported
results. Fixture graph construction is excluded, compilation and persistent
execution are separate; ordinary publication costs are included in execution.
Receipt: `../benchmarks/builder-order-2026-10-27.json`.

At 10k nodes the terminal value program compiles in median 0.175 ms optimized vs
0.019 ms without optimization, and executes in 12.64 vs 13.28 ms. The interleaved
program retains six original statements in both modes; compilation medians are
0.020 vs 0.017 ms, execution 37.22 vs 33.74 ms. These small local samples show
current costs, not before/after speedups or statistically established regressions.
No allocation/RSS/production guarantee is inferred.

## Fixed remaining scope

AR7.8/30.1 still own complete typed read/write/topology effects, alias/redefinition
identity, all lowering paths/custom plans, and certified fusion/folding domains.
09.4b.2 still owns compound topology acceptance. This repair makes no blanket
optimizer or atomic-topology claim.

An execution probe also confirms an existing unfinished builder facade: `a ** b`
raises NotImplementedError and `core.pow(...)` emits a step rejected as unsupported
by BuiltAlgorithm encoding, in both optimization modes. That remains part of the
existing 30.1 capability/lowering inventory; the matrix/array power APIs are
separate. No extra checklist row is created. Power signatures are tested for
operand identity without claiming builder power execution is implemented.
