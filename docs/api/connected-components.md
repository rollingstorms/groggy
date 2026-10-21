# Connected components

Connected components operate on the selected node and edge IDs. Every selected ID
must be active and every selected edge must have both endpoints in the selection.
Validation precedes empty returns and cache reads. Inactive IDs remain retained
for history but cannot participate in a live algorithm run. No input topology is
modified. Results are written through one existing bulk property setter after
successful computation; this does not introduce a new durable transaction boundary.
Cancellation rejects before final property writes.

## Semantics

`algorithms.community.connected_components(mode="undirected", output_attr="component")`:

- `undirected` and `weak` both ignore orientation, including on directed graphs.
- `strong` requires mutual reachability on directed graphs. On undirected graphs,
  it returns the same partition as ordinary connectivity.
- Loops and parallel instances do not change component membership. Isolates form
  singleton components; empty input has no components.
- Numeric component IDs are labels, without a stable cross-version ordering
  contract. Compare partitions using membership rather than numeric label values.
- BFS handles weak/undirected components; iterative Tarjan handles strong ones.
  Neither uses recursion. Computation supports cancellation. The current dense
  labels/DFS indices require fewer than `u32::MAX` selected nodes, checked before
  traversal. Sparse external IDs use a hash index without allocating their span.
- Shared CSR caches distinguish reverse-edge policies. Component caches distinguish
  modes and require the graph-space version to match. Validation is performed
  even when an entry is cached. A retained selection containing a removed entity
  rejects; create a new live selection to omit removed IDs.
- Nonpersistent execution returns `community.connected_components.components`
  in the native context and leaves properties unchanged. Persistent component
  caches retain internal selected edge instances, including loops/parallels.

The traversal kernels take O(V + E) time and memory. Whole-call checked selection
and deterministic ID ordering additionally sort IDs. This release does not claim
performance gains, bounded cancellation latency during every setup phase, backend
parity, or production-scale verification.

## Evidence

Native `tests/components_validation.rs` checks all 512 three-node edge sets
for both graph directions, all three modes, and repeated execution against
independent transitive closure. It also covers invalid selections, cancellation,
sparse IDs, parallel/loop fixtures, nonpersistent output and a 10,000-node
directed path. `tests/test_components_validation.py` checks the rebuilt Python
surface. NetworkX/Neo4j comparison and larger-scale verification remain open.
