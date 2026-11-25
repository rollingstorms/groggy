# Portable structural steps — 2026-11-24

Finite component of existing 03.2, 09.4b.2, 30.1, 30.3 and AR7.8. This completes
the five-step `steps/structural.rs` native-read inventory: `core.node_degree` was
already portable; the remaining four now use the same boundary. The stable flat
list remains 217 checked / 51 open; broader execution and adjacency audits stay
partial.

## Completed component checks

- [x] Route `core.weighted_degree` through the sealed selected-read kernel.
- [x] Route `core.edge_weight_sum` through the same incidence-sum implementation.
- [x] Route `core.triangle_count` through its existing checked triangle kernel.
- [x] Route `core.k_core_mark` through its existing checked peeling algorithm.
- [x] Use the same implementation for owner memory execution and detached durable
  workers, including nested ordinary loop bodies and pending weights/topology.
- [x] Cover sparse selected IDs, outside-selection isolation, creations, endpoint
  revisions, parallel edges, loops, signed/missing weights, invalid domains,
  overflow/failure rollback, one publication and exact recovery.

## Contract preserved

Weighted degree and edge-weight sum both compute total incidence. Both endpoints
contribute on directed and undirected inputs; self-loops contribute twice.
Missing weights are one, present values must be finite numeric scalars, signed
weights remain valid, and nonfinite accumulated results reject. This is distinct
from directed outgoing unweighted `core.node_degree`.

Triangles use the simple loop-free projection. K-core peeling retains selected
edge-instance multiplicity and counts a self-loop once. These two steps require
undirected input. No directed definition, weight restriction or result type was
changed. Existing cancellation and complete-result-before-output behavior remain.

Each public owner adapter now delegates to its sealed read kernel. The helpers
read through `ReadStepScope` rather than accessing `Subgraph` owners; algorithms
remain in Rust and FFI only uses the already-established worker/coordinator
boundary. This does not add a second algorithm implementation.

## Verification

Targeted native tests include the existing exhaustive five-node references and
`tests/durable_structural_steps.rs`. The new worker test compares every result
with explicit expected values and memory execution, then reopens the journal to
check effective-time boundaries. The failure matrix covers directed-domain,
nonnumeric-weight and accumulated-overflow rejection with unchanged IDs/history.
The existing unsupported-step preflight fixtures now use the still-unported
median step, preserving negative coverage after weighted degree becomes portable.

Eight new Python cases exercise both weighted step names on directed/undirected
inputs, triangle/k-core loops, exact recovery and directed rejection rollback.
The new executable example is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Final local qualification passes **1,672 native tests** (54 ignored), **3,458
packaged Python tests** (one skipped, 14 established warnings), and **295
documentation examples**. Strict all-target Clippy, root formatting, whitespace,
changed Python 3.8 syntax and all 62 packaged Python/stub byte comparisons pass.
All 500 production source files match runtime `9ae820e8550c9878eb1543202839b8d7996909c9`.
See [local receipt](../benchmarks/portable-structural-local-qualification-2026-11-24.json)
and [wheel receipt](../benchmarks/portable-structural-wheel-qualification-2026-11-24.json). No new platform
qualification or structural performance improvement is claimed. The earlier
[ordinary-loop timings](../benchmarks/ordinary-loop-performance-2026-11-24.json)
remain evidence for their recorded runtime and workload.

## Remaining scope

Other unported aggregations/history producers and composition containers,
edge-aware registers, normalization options, fixed-child/provisional topology,
remaining expression/IR lowering and streaming stay in the existing execution
parents. A14 naming/cache/history and A15 production budgets remain unanswered.
