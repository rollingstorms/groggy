# BFS and DFS algorithm wrappers

The registered algorithms retain their Rust constructors and Python signatures:

```python
from groggy import algorithms
from groggy.pipeline import apply

result, profile = apply(
    selection,
    [algorithms.pathfinding.bfs("start", "distance"),
     algorithms.pathfinding.dfs("start", "discovery_time")],
    persist=False,
    return_profile=True,
)
distances = profile["outputs"]["pathfinding.bfs.distances"]
preorder = profile["outputs"]["pathfinding.dfs.order"]
```

## Start selection and graph boundaries

Exactly one selected node must carry the named start property. Its value is
ignored: false, zero and explicit null still mark the source. The value is not
interpreted as another node ID. Missing or multiple markers reject before output
writes. Empty input therefore rejects for lack of a source; an isolated marked
singleton returns `(node, 0)`. This tightens the former arbitrary choice among
multiple markers and corrects misleading metadata/source-value wording.

Both algorithms validate every selected node and edge for active membership and
every selected edge for endpoint closure before reading caches or traversing.
A retained selection containing deactivated IDs rejects; it is not implicitly a
historical snapshot. Directed runs follow outgoing edges, and undirected runs
allow both orientations. Traversal remains within the selected node/edge set.
Self-loops and parallel edge instances do not repeat a node in output. Weights
are ignored; these wrappers perform unweighted traversal.

## Values and ordering

- BFS returns minimum unweighted hop distances, with source distance zero.
  Nonpersistent rows are ordered by original node ID.
- DFS returns zero-based recursive-style preorder indices, with source index
  zero. Visit neighbors in ascending original node-ID order. This ordering is
  independent of edge insertion and shared CSR-cache history. Nonpersistent rows
  follow preorder, rather than node-ID order.
- DFS sorts a private CSR copy and uses explicit recursive frames. It does not
  mutate the shared cache or recurse on the machine call stack.
- Only reachable nodes appear. Persistent runs bulk-write integer properties for
  reached nodes; previous output values on unreached nodes are retained. Use the
  returned rows to distinguish the current run's reachability from old properties.
- Nonpersistent runs return owned exact `(NodeId, i64)` pairs under the profile
  keys shown above, using native `NodeIntegers`; no properties change. The FFI
  only translates the pairs. Integer overflow rejects instead of wrapping.

Cancellation is checked before work, within queue/frame traversal and before the
final bulk setter. Validation/cancellation errors do not publish result properties.
No fixed cancellation-latency bound is promised for every validation/build/sort or
high-degree neighbor scan. The existing property setter is not a new durable
transaction boundary, and state-pinned historical input migration remains open.

## Costs and evidence

BFS and the DFS traversal kernel take O(V+E). DFS additionally copies O(V+E) CSR
storage and sorts each adjacency row, costing up to O(E log E). Its traversal
frames occupy O(V). Ordered collection and checked preflight add ID sorting and
validation costs. No speedup or production-scale capacity is claimed.

`tests/traversal_wrappers.rs` checks all 512 three-node edge sets for both graph
directions, every source, loops/parallel instances and repeated cached execution.
It compares BFS with direct adjacency traversal and DFS with an independent small
recursive reference. Reversed edge insertion tests deterministic ordering; actual
returned selections are carried between runs to exercise populated CSR caches.
Other fixtures cover missing/ambiguous markers, explicit null/false/zero markers,
invalid/stale/cancelled selections, sparse IDs, a singleton and a 20,000-node path.
`tests/test_traversal_wrappers.py` checks rebuilt Python registry/pipeline APIs,
exact nonpersistent integers, marker semantics and persistent results.

The closeness alternating-direction fixture also now carries returned selections
between executions, so its cache-isolation assertion exercises populated caches.
Direct Graph traversal APIs and lower-level utility/step callers have separate
contracts; this slice does not certify every such caller. The [Dijkstra wrapper](dijkstra.md) now has its own unique-source, cancellation
and nonpersistent-output follow-up; lower-level/step callers retain separate
audit requirements. External
backend, scale and state-pinned-input certification remain open under item 28.
