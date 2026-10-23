# Immutable algorithm input and traversal — 2026-10-22

AR7.5f introduces a prepared immutable read boundary for algorithm kernels and
migrates BFS/DFS integer-row computation to it. Broader algorithm/pipeline read
migration remains open under AR7.5, AR4, 26.1 and 28.16.

## Input ownership

AlgorithmRead owns a GraphView and shares sorted node IDs, an ID-to-row index and
CSR through Arc. Clones reuse the preparation and captured state; there is no
working-graph reference. Construction uses the existing CSR builder, validates
selected endpoints, includes isolates, retains parallel occurrences and loops,
and adds reverse adjacency only for undirected graphs. Sorted original node IDs
establish stable row identity; algorithms do not mistake sparse IDs for indices.
The input's view retains captured properties for later reader migrations.

Current dirty captures and recorded historical snapshots can supply the same
GraphView/read input. No implicit refresh, checkpoint or mutation delta occurs.
Prepared input lifetime is independent of current graph membership/owner lifetime.
There is no global topology-version/head-ID cache: those identifiers cannot safely
identify dirty property state. Preparation is explicit and reusable natively.

## Shared kernels and public results

- GraphView.bfs_distances(start) returns reachable (node ID, hop distance) rows
  sorted by original ID. GraphView.dfs_order(start) returns (node ID, preorder
  index) rows in discovery order with ascending-neighbor recursive DFS semantics.
  The implementation is iterative, including for long paths.
- Results contain exact i64 integer values. Loops and parallel connections do not
  discover a node twice. Traversal follows directed outgoing edges or undirected
  adjacency and cannot cross the selection's edge scope.
- Unknown/unselected starts fail. A selected isolate returns its own ID and zero;
  an empty input cannot supply a start. Neither method writes graph attributes.
- AlgorithmRead exposes the same operations with optional atomic cancellation.
  Kernels poll at entry, during work and before returning owned rows; cancellation
  returns an error without partial output. CSR preparation/allocation and a row's
  sort are not interruptible through this flag; no bounded latency is promised.
- Legacy pipeline BFS/DFS share the kernels via the existing Context cancellation
  callback. Marker validation, telemetry and explicit persist/nonpersist behavior
  remain in their current-state adapters. This does not freeze those whole
  pipelines or migrate their writer/concurrency contract.

Python methods are thin Rust adapters and release the GIL for input preparation
and traversal. Their results are owned integer rows, not graph traversal-tree
selections. Native callers can retain view.algorithm_input() and clone it cheaply;
Python convenience methods prepare per call. Depth/node-limit options and the
existing graph selection-returning traversal API remain distinct.

## Evidence and limits

Five native integration fixtures cover all 512 three-node edge sets in both graph
directions and all starts, sparse IDs, parallel/loop occurrences, scoped excluded
shortcuts/outside connections, isolates/empty inputs, exact rows, cancellation,
owner mutation/drop, thread transfer, a 10,000-node iterative path and recorded
historical topology. An additional native kernel test deterministically cancels
during work. Existing legacy wrapper reference/cancellation/persistence tests run
against the shared kernels. New Python fixtures verify captured scope/direction,
ordering, exact ints, worker readers, input errors and no property writes.

Executed: full locked offline native suites; strict all-target core Clippy;
rebuilt in-tree Python extension (three existing FFI warnings). Combined
traversal/wrapper/core/subgraph/accessor/algorithm/read/history/property Python
suite: **276 passed, 5 existing skips**. All six read-view documentation examples
execute successfully. Bounded native costs are retained in
`notes/benchmarks/frozen-traversal-2026-10-22.md`.

Weighted paths, centrality, components and the remaining registry/pipeline APIs
still need pinned-input migration. This is not arbitrary effective-time graph
reconstruction, checked view writeback, universal algorithm inheritance or a new
production memory/latency guarantee. No workbook policy is chosen here.
