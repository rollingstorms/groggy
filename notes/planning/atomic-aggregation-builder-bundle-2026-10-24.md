# Checked aggregation builder bundle — 2026-10-24

Closes **09.4b.2o / 09.4b.2p / 09.4b.2q**, extending **30.1 / AR7.8**.

Owner requested slices about three times larger and then confirmed continuing this
bundle. This delivery groups three related workstreams into one reviewable commit;
subsequent slices should continue at that larger cadence. Broader parents remain
partial; this is not a claim to finish all builder or roadmap work.

## 1. Statistical reductions

Certify reduce_nodes, std, median and quantile. Numeric inputs require finite
integer/float scalars; invalid present values and nonfinite outputs error. Both
reduce_nodes and the previously separate reduce_scalar path now use one native
reduction kernel. Sum/mean accumulate integer high/residual parts in the exact
product accumulator; mean divides the accumulator before final rounding. Standard
deviation uses the shared centered sample kernel; median/quantile use shared
checked exact-order interpolation. No duplicated naive sum/variance/quantile path
is left in these builder steps.

Preserve empty conventions: reduce_nodes/std/median/quantile return None;
reduce_scalar keeps its zero empty result and scalar pass-through. Singleton std
is zero. Quantile q validates during preparation and runtime. Scalar statistical
outputs remain f64. New public CoreOps.std/median/quantile/reduce_nodes methods
lower through IR and legacy builders, including ordinary optimized execution.

## 2. Categories and histogram ownership

Certify mode and entropy with typed serialization keys instead of debug keys,
deterministic iteration and mode ties chosen by the smallest contributor node ID.
Mode preserves category type; entropy is Shannon bits, with empty=0. Nonfinite
floating categories reject. Public mode_values distinguishes the global statistic
from existing per-node list mode; public entropy is also available.

Histogram bins formerly masqueraded as node IDs, allowing unrelated node property
writes. Histogram now produces a scalar IntList variable plus an owned
AlgorithmOutput::Histogram count vector exposed as outputs["histograms.<target>"].
The FFI only marshals that vector. Direct node-map attachment rejects. Explicitly
broadcasting the scalar vector is a caller choice. Empty input returns requested
zero bins; constants go to bin zero. Validate positive bins and allocation, retain
narrow integer/float differences, and scale extremely wide ranges to avoid width
overflow. This deliberately repairs the old histogram output contract; the prior
Python regression asserting accidental bin-ID writes was replaced with an ownership
regression. Public histogram documentation now describes the corrected result.

Owned histogram outputs are observable even without property attachment. Until IR
effects represent those outputs, histogram-containing builds retain their ordered
program, including ordinary non-atomic builds and nested loop bodies. No DCE pass
may silently remove the output.

## 3. Neighbor aggregation and builder integration

Certify all five standalone neighbor reducers: sum/mean/min/max/mode. Build sorted
selected directional CSR, validate active nodes/edges and endpoint closure, preserve
parallel-edge multiplicity and one contribution per stored self-loop. Direction is
explicit over stored endpoints, default in, with out/undirected/both available.
Missing source-map rows remain omitted. Present invalid numeric values error;
weighted sum requires a finite numeric node multiplier for each contributor.
Non-sum weights reject. Supplied reducer/aggregation/direction/weight selectors
must have their declared text type; malformed selectors no longer silently choose
defaults. Numeric sums retain residuals and weighted products use the
exact accumulator, avoiding intermediate product overflow when the final result
is representable. Empty sum/mean=0; min/max/mode=None. Mode shares the category and
contributor-ID tie contract.

Public GraphOps.neighbor_agg now accepts direction and preserves weight operands.
Both Python lowering paths and optimizer rebuilds retain these operands. Fixed the
metadata-only weight reference loss by tracking the second input variable.
Message-pass bodies still implement unweighted sum; other aggregation/weight
requests there now reject explicitly instead of silently ignoring metadata.

Removed the invalid optimizer rewrite aggregate(values)*scale ->
aggregate(values*scale). The fused neighbor kernel operates on contributor scales,
while the original expression scales completed node results. This rewrite changed
results and ignored aggregation type. Explicit fused primitives remain available;
other arithmetic fusion paths remain in place. Optimized/unoptimized parity tests
cover all five reducers and nonuniform output scales.

## Costs and remaining work

Statistics sort/allocate where needed; histogram count vectors require O(bins)
variable/output storage. Standalone neighbors rebuild CSR and allocate node result
maps; mode/entropy retain category frequency maps. Exact accumulation costs more
than naive addition. Removing the invalid rewrite can retain additional temporary
variables. These are explicit correctness costs, not measured whole-project
regression/speedup claims. Native owners and graph history are not cloned.

Cancellation is checked around kernels and before result installation/publication;
individual numerical kernels/sorts do not promise bounded interruption latency.
Streaming, remaining uncertified families, arbitrary outer transactions, complete
optimizer effects and atomic-default migration remain parent work.

## Release receipt

`benches/aggregation_builder_steps.rs` covers 14 native modes at 1,000/10,000 nodes
and edges, five samples each: sum, mean, std, median, quantile, mode, entropy,
histogram, five neighbor reducers and weighted neighbor sum. Fixture/spec creation
is excluded; calculation, owned histogram or graph attachment and checked acceptance
are included. Every output is checked, including zero histogram publication and
one publication for property results. Raw receipt:
`notes/benchmarks/aggregation-builder-bundle-2026-10-24.json`.
Ring/uniform fixtures do not characterize dense or skewed neighbor workloads.
No matched speedup, allocation/RSS, FFI or production budget claim is made.

## Verification

- Full offline/locked native suite: **953 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension plus changed Python sources; existing three FFI
  warnings remain. No package installation.
- Combined focused aggregation/atomic/core builder suites: **266 passed**.
- Full root Python API gate: **1,312 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **48 executable documentation examples passed**, including four new public
  statistical/histogram/neighbor/optimizer examples.
- **28 release workloads, five samples each**, passed all output/publication checks.
- `git diff --check` passed. Owner development/benchmark changes remain separate.

Tests cover empty/singleton statistics, exact mixed residuals, integer spread beyond
2**53, overflow-safe median/mean/weighted products, typed mode ties, entropy, narrow
and wide histogram ranges, bin/node identity rejection, sparse/selected neighbors,
all five reducers in three directions, IR/legacy/optimized public APIs, output
scaling parity, invalid controls and rollback. Prior unsupported-step fixtures now
use sampling, which remains uncertified, rather than newly supported histograms.

## Current local release observations

Median milliseconds at 10,000 nodes/edges, five samples per mode. Histogram is an
owned-output workload with no graph publication; other rows include property
attachment/publication. These distinct workloads are not a speedup comparison.

| Mode | Median ms |
| --- | ---: |
| sum | 11.876 |
| mean | 11.840 |
| std | 11.818 |
| median | 11.734 |
| quantile | 11.872 |
| mode | 12.328 |
| entropy | 12.343 |
| histogram | 1.861 |
| neighbor_sum | 14.576 |
| neighbor_mean | 14.438 |
| neighbor_min | 13.108 |
| neighbor_max | 13.164 |
| neighbor_mode | 13.906 |
| neighbor_weighted_sum | 14.769 |
