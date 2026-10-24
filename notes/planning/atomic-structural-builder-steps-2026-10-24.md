# Atomic degree and incident-weight builder steps — 2026-10-24

Closes **09.4b.2l**, extending **30.1 / AR7.8**.

## Implementation

- Certify `core.node_degree`, `core.weighted_degree` and `core.edge_weight_sum` for
  atomic property pipelines and nested loops. These produce temporary variables;
  only explicit attachment steps stage graph properties.
- Share one native incident-weight kernel between the two weighted primitives.
  Read through StepScope's overlay so earlier edge attachments are visible.
  Transient pipelines continue to read owner properties and return owned outputs.
- Preserve weighted total incidence semantics on directed/undirected graphs:
  both endpoints receive each edge weight; a self-loop contributes twice.
  Missing weights default to one. Signed finite scalar weights are valid.
- Reject present null/nonnumeric/nonfinite weights and nonfinite sums instead of
  silently treating malformed values as unit weights. This deliberate validation
  repair applies to ordinary pipelines too. f64 inputs retain their precision;
  integer weights convert to f64 for these sums.
- Degree uses one selected-edge pass, checked i64 counters and cached edge order.
  Preserve directed outgoing counts, undirected endpoint counts, parallel edges
  and once-per-loop unweighted counts. Explicit source variables must exist and
  contain selected active nodes; incorrect sources no longer silently fall back
  to the full node selection. Endpoint errors and cancellation propagate.
- Weighted sums also use cached edge ID order for repeatable accumulation. Initial
  ordering may sort the selection; later calls reuse the cache. No graph clone,
  additional storage format, FFI business logic or publication per node is added.

## Remaining scope

Triangle and k-core builder primitives still need their own semantic review and
atomic certification. Other step families, streaming, alternate batch plans,
effect-aware optimization and default atomic policy remain parent work. Existing
unsupported-step tests now use triangle_count so they continue to test an actual
uncertified descendant rather than a newly supported degree step.

## Performance receipt

`benches/structural_builder_steps.rs` covers three native builder primitives at
1,000/10,000 nodes and edges on a directed ring, five samples each. Timed execution
includes staged edge weight assignment, structural calculation, node-result
staging and one publication. Fixture/spec construction is excluded. Every node
result and clean publication are checked. Raw receipt:
`notes/benchmarks/structural-builder-steps-2026-10-24.json`.

These are bounded current observations, not matched before/after speedup,
allocation/RSS, FFI or production-budget guarantees.

## Verification

Focused Python cases cover public atomic degree, directed/undirected multiedges
and loops, staged f64 weights, transient owner reads, nested loops, missing/signed
weights, invalid weight types, nonfinite weights and sum overflow, missing degree
sources, and rollback of mixed prefix writes. Native cases cover selected-edge
scope, foreign source nodes and cancellation without replacing existing results.
Documentation contains a runnable public degree builder example. Full results
and receipt observations are recorded below.

- Full offline/locked native suite: **945 passed, 52 ignored**.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension using existing macOS dynamic lookup settings;
  existing three FFI warnings remain. No package installation.
- **83 focused atomic-builder cases passed**.
- Full root Python API gate: **1,159 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **42 executable documentation examples passed**.
- Six release workloads passed all output/publication checks. Observed medians
  (ms), 1,000/10,000 nodes: degree 2.758/23.617, weighted_degree 2.604/23.342,
  edge_weight_sum 2.286/23.364. Distinct workload timings do not imply universal
  speedups or throughput budgets.
- `git diff --check` passed; owner development/benchmark changes remain separate.
