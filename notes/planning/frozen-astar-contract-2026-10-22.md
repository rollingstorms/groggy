# Frozen checked A* paths — 2026-10-22

AR7.5h completes the pinned weighted-path slice after checked Dijkstra. Broader
component/centrality/registry/pipeline migration remains open in AR7.5/28.16.
No workbook policy is selected here.

## Shared computation and validation

AlgorithmRead/GraphView.astar_path use the captured selection and prepared CSR.
Live/frozen readers share heuristic value validation and one search kernel;
weights use the common checked Dijkstra domain/minimum parallel-pair map. The
kernel consumes CSR row indices directly, removing the old linear node-ID search
on each heap pop. Strict improvements, finite cost/priority checks, cancellation,
unreachable errors and the existing heap tie order remain explicit.

Absent heuristics are zero; present values must be numeric, finite, nonnegative.
The goal must be zero. Each selected traversable pair must satisfy
h(u) <= weight(u,v)+h(v), including reverse pairs for undirected graphs. All
selected weights/heuristics are validated, including isolated/unreachable inputs
and start-equals-goal. Null/Boolean/text/vector heuristic values fail. Existing
weight rules and f64 conversion/output scope are unchanged. Missing checked
weights are errors rather than an implicit kernel fallback to one.

## Owned path and identity

The output reuses query::Path: ordered original nodes, actual selected edge IDs,
f64 total_weight and existing PathMetadata (algorithm/is_simple/edge-count length).
Path instance reconstruction scans selected edge IDs only, not unrelated captured
adjacency. For each path pair it chooses the minimum weight; equal f64-cost
parallel instances resolve to the lowest selected edge ID. Zero-cost cycles do
not create predecessor cycles; the returned path is simple. An unreachable goal
fails; a valid trivial path contains one node, no edges and zero cost.

Python GraphPath is a thin immutable wrapper over that owned native result.
Node/edge getters return independent lists. GraphView preparation/search/path
assembly release the GIL. Native prepared inputs can be reused and accept an
atomic cancellation flag. Current graph changes/deactivation/owner destruction
cannot change captured properties or the owned result. Calls write no graph
attributes/history. Source pins remain with the read input; Path is an owned
result, not a connected graph view or writeback owner.

The legacy A* pipeline keeps marker resolution, path-index outputs and cost/
performance telemetry, while sharing the computation. This does not freeze the
whole pipeline or migrate its publication/writer contract. Preparation/value
validation/sorting do not provide a bounded cancellation-latency promise.

## Evidence

Three new native fixtures compare all 512 three-node edge sets in both directions,
all start/goal pairs, sparse IDs and zero/positive parallel weights to independent
Floyd–Warshall costs. They verify actual oriented edge instances and deterministic
equal-cost IDs, simple paths/cost sums, heuristic domains/consistency/goal zero,
isolated invalid values, selected scope, cancellation, owner mutation/drop and
reader threads. Existing A*/Dijkstra wrapper validation/output tests also run.
Python verifies immutable GraphPath data, copies, directions, captured values,
threaded reads, errors and trivial paths. Benchmark costs are retained separately
from correctness: `notes/benchmarks/frozen-astar-2026-10-22.md`.

Executed gates: full locked offline native suites, final focused weighted-path
fixtures, strict all-target core Clippy and the rebuilt in-tree Python extension
(three existing FFI warnings). Combined weighted-path/traversal/core/read/history/
property Python suite: **325 passed, 5 existing skips**. All eight read-view
examples execute successfully. Black was unavailable in the existing environment;
no package was installed, and the new Python test was formatted manually.
