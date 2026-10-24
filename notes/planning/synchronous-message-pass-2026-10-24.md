# Synchronous message-pass execution — 2026-10-24

Closes **09.4b.2k**, extending **30.1 / AR7.8**.

## Contract and implementation

- `ordered=False` now honors synchronous Jacobi semantics in ordinary and atomic
  message-pass builders. Each node reads the target values captured at block
  entry, including self/neighbor reads. Later blocks and loop iterations see the
  completed result. Computation is serial; no parallel execution claim is made.
- Keep pending outputs separate from the input column and install them only after
  every node succeeds and a final cancellation check passes. No graph clone or
  intermediate graph version is created. Failure leaves the original run variable
  intact; the atomic outer executor also discards earlier staged attachments.
- `ordered=True` retains Gauss-Seidel visibility: each completed node update is
  visible to subsequent nodes within that block. Both modes use the same selected
  directional topology, body evaluator, conditionals and canonical attachments.
- Preserve maps as maps and columns as columns with their original row order.
  Synchronous results do not depend on traversal order when reading the target.
  Parallel edges contribute repeatedly; stored self-loops contribute once;
  include-self and direction semantics match ordered execution.
- Empty bodies preserve input values. Empty graph attachment emits no assignments
  or accepted state. Node/body cancellation checks remain, with a check immediately
  before completed output installation.
- Python API documentation now describes synchronous serial evaluation rather
  than promising parallel Jacobi execution.

## Cost and scope

Synchronous mode adds one output buffer proportional to target node count, on top
of the existing temporary input column, selected CSR and outer property staging.
Output values move into the retained column after success. This buffer is needed
for fixed-input semantics; it is not a whole-graph snapshot. Streaming, remaining
unsupported step families, alternate batch plans, default policy and effect-aware
optimization remain parent work.

The release benchmark extends `benches/atomic_builder_blocks.rs` with synchronous
ring loops at 1,000/10,000 nodes and edges, five iterations and five samples per
case. Its original fused/ordered cases are rerun as current observations. Every
output and one clean publication are checked. Fixture/spec construction is outside
timing. It does not establish matched speedup, parallelism, allocation/RSS budgets
or production throughput. Raw receipt:
`notes/benchmarks/synchronous-message-pass-2026-10-24.json`.

## Verification

Native tests check reversed/permuted column order and fixed neighbor inputs, plus
failure on a later node without changing the original column. Python tests check
public ordinary/atomic builders with both visibility modes, direction/multiplicity/
self-loops, staged repeated iterations, prefix rollback, empty bodies/graphs and
streaming rejection. Documentation contains a runnable directed-chain example.
Full gate results and benchmark observations follow below.

- Full offline/locked native suite: **943 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension and changed Python sources; existing three FFI
  warnings remain. No package installation.
- **60 focused atomic-builder cases passed**.
- Full root Python API gate: **1,136 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **41 executable documentation examples passed**.
- Six release ring workloads passed all output and publication checks. Observed
  medians (ms), 1,000/10,000 nodes: fused 3.431/25.370, ordered 4.656/41.012,
  synchronous 3.957/40.972. These timings are current local observations of distinct
  kernels, not evidence that one visibility mode is universally faster.
- `git diff --check` passed; owner development/benchmark changes remain separate.
