# Node sampling on graph selections

`sample(k, seed=None)` chooses up to `k` distinct nodes uniformly without
replacement. It retains exactly the selected edges whose two endpoints are in
the sampled nodes. Parallel edge IDs, direction and self-loops are preserved;
edges excluded by an input selection stay excluded.

The integer form works on `Graph`, connected `Subgraph`, and immutable
`GraphView`. Zero yields an empty selection; counts at or above the population
retain the complete selection. Counts are nonnegative integers and seeds are
unsigned 64-bit integers; booleans are rejected. The same seed and sorted input
IDs reproduce node membership across connected and captured inputs. Seeded
sampling uses a local RNG and leaves the shared random stream unchanged.

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes(8)
for node in range(8):
    graph.add_edge(node, node)
connected = graph.view()
frozen = connected.pin_view()
a = connected.sample(3, seed=17)
b = frozen.sample(3, seed=17)
assert sorted(a.node_ids) == b.node_ids
assert sorted(a.edge_ids) == b.edge_ids
assert b.node_count() == b.edge_count() == 3
assert graph.sample(0).node_count() == 0
assert frozen.sample(100).node_ids == frozen.node_ids
assert b.write_state_id == frozen.write_state_id

removed = b.node_ids[0]
graph.remove_node(removed)
assert b.node_count() == 3  # immutable membership and values survive owner writes
try:
    connected.sample(0)
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("An invalid current-owner scope must reject even at k=0")
```

Connected sampling validates the complete active endpoint-closed input even for
zero or oversized requests. Its output remains a connected selection with live
owner reads; capture explicitly for frozen state. Immutable sampling shares the
original capture, clocks, retained group catalog and checked-write provenance.
It never reconstructs a writable graph or silently recaptures a later owner state.
Neither form allocates graph entities or publishes deltas. An explicit subsequent
`Graph.apply_view_batch` still enforces current source and selected write scope.

`Subgraph.sample(sampler)` keeps its existing pipeline behavior. Set pipeline
seeds in the sampler specification; passing a separate seed with a sampler
object rejects explicitly. Immutable `GraphView.sample` currently supports the
integer form only. Whole sampler pipelines and effectful facade migration retain
their existing roadmap scope.

Existing component, neighborhood and subgraph collection `sample(k)` calls now
sample each input rather than cloning it. Counts cap separately per input. Invalid
inputs raise an error instead of silently disappearing from the returned array;
no partial array is returned. Collection signatures remain unchanged.

## Implementation and costs

The Rust core and pipeline node/edge mask steps share Algorithm R. Output reservoir
space is O(min(k,N)); input ID ordering and edge induction have separate costs.
Connected views reuse sorted IDs and validate live scope; immutable views derive
sorted IDs from the capture. Induction stays inside selected edge membership.
Zero requests skip ID sorting and edge induction after required live validation.
Oversized requests share the existing selection after live validation where needed.
Initial immutable capture still materializes owner values and topology. Native
release measurements are recorded separately from Python conversion and GIL costs.
Immutable sampling releases the GIL; connected sampling retains it because its
owner uses `Rc<RefCell<Graph>>`.
