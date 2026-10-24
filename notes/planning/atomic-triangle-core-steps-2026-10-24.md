# Undirected atomic triangle and k-core steps — 2026-10-24

Closes **09.4b.2m**, extending **30.1 / AR7.8**.

## Implemented semantics

- Certify `core.triangle_count` and `core.k_core_mark` for atomic property builders
  and nested loops. Both emit temporary node maps; attachments enter the outer
  batch. Empty graph output creates no assignments/version.
- Build checked selected undirected topology, validating active nodes/edges and
  endpoint closure. Preserve original IDs, parallel edges and one adjacency entry
  per stored self-loop. Sort rows for deterministic traversal.
- Triangle counting uses the simple projection: ignore self-loops and collapse
  parallel adjacency before orienting neighbors by node ID. Intersect forward rows
  to visit each distinct three-node triangle once and increment all three checked
  i64 counts. Prior duplicate-neighbor/self-loop artifacts are repaired.
- K-core uses queue-based peeling with the existing undirected degree convention:
  parallel edges contribute repeatedly, each stored self-loop once. Remove nodes
  whose remaining selected degree is below k, updating each retained neighbor for
  every removed incidence. k=0 includes isolates. Reject negative/out-of-range k
  during instantiation rather than casting negative values to huge usize values.
- Check cancellation during preparation, peeling and triangle intersections, and
  before installing results. Errors leave existing result variables intact; an
  atomic outer run discards earlier staged property attachments.

## Directed boundary and compatibility

Directed graph calls explicitly reject for these two primitives, in ordinary and
atomic builders. The prior triangle path mixed outgoing neighbors with closure
checks, while k-core decremented outgoing neighbors instead of predecessors. A
particular directed cycle/transitive-triangle/core definition is not inferred from
that inconsistent behavior. Directed variants remain unsupported and visible under
30.1/AR7.8; this does not mark those parents complete or add a policy approval gate.

Existing undirected simple-graph behavior is preserved. Multigraph triangle counts
are corrected; k-core retains the established multiedge/self-loop degree model.
Preflight rejection tests now use the still-uncertified histogram family, ensuring
that newly supported structural steps are not used as unsupported fixtures.

## Complexity and release receipt

Topology preparation uses O(V+E) memory and sorts adjacency rows. K-core's peeling
pass is O(V+E) over those prepared lists. Triangle forward intersections avoid
repeatedly counting triangles but can still require cubic work on dense graphs;
no universal throughput bound is implied. Local topology storage is computation
scratch, not a graph clone or a retained snapshot.

`benches/structural_builder_steps.rs` now includes undirected triangle and 2-core
ring cases at 1,000/10,000 nodes, alongside the three earlier structural cases.
Five samples per case include edge-weight staging, structural evaluation, result
staging and one publication. Every node result and clean publication are checked;
fixture/spec construction is outside timing. Ring cases do not characterize dense
triangle workloads. Raw receipt:
`notes/benchmarks/triangle-core-builder-steps-2026-10-24.json`.

## Verification

Native exhaustive references cover all 1,024 simple undirected graphs on five nodes:
independent triple enumeration for per-node triangles, and synchronous fixed-point
pruning for every k from 0 through 5. Additional native checks cover selected
noncontiguous IDs and cancellation preserving old result variables. Python checks
ordinary/atomic and transient/persistent triangle output, multiedges/self-loops,
core peeling cascades, repeated staged iterations, directed rejection with prefix
rollback, negative k and empty graphs. Full gate results follow below.

- Full offline/locked native suite: **947 passed, 52 ignored**, including exhaustive
  simple-graph references.
- Strict all-target native Clippy with warnings denied: passed.
- Rebuilt/staged in-tree extension using existing macOS dynamic lookup settings;
  three existing FFI warnings remain. No package installation.
- **96 focused atomic-builder cases passed**.
- Full root Python API gate: **1,172 passed, 1 skipped**, excluding the unrelated
  owner benchmark test and separate legacy modules survey.
- **43 executable documentation examples passed**.
- Ten release ring workloads passed all result/publication checks. Observed
  medians (ms), 1,000/10,000 nodes: triangles 2.248/22.266,
  k-core 2.214/22.819. No matched speedup, allocation/RSS or production-budget claim.
- `git diff --check` passed; owner development/benchmark changes remain separate.
