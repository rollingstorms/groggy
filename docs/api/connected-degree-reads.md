# Connected degree and neighbor reads

Legacy `Subgraph` objects still select from their current owner. Their neighbor
and degree readers now validate the entire selected scope before using retained
pool endpoints or cached topology. Removing a selected node/edge or moving a
selected edge endpoint outside the selection makes that connected selection
invalid; reads reject rather than returning historical neighbors or zero degrees.
An explicit frozen `GraphView` remains readable in its original state.

```python
import groggy as gr

graph = gr.Graph(directed=True)
graph.add_nodes(3)
graph.add_edge(0, 1)
graph.add_edge(1, 1)
graph.add_edge(2, 1)
selected = graph.subgraph_from_edges([0, 1])
frozen = selected.pin_view()
assert selected.degree([1, 0, 1]).to_list() == [2, 1, 2]
assert selected.in_degree().to_list() == [0, 2]
assert selected.out_degree().to_list() == [1, 1]
assert selected.degree(1, full_graph=True) == 3

graph.remove_node(1)
try:
    selected.neighbors(0)
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("A stale connected selection must reject")
assert frozen.degree(1) == 2
assert frozen.neighbors(0) == [1]
```

## Counting rules and result alignment

- `degree` counts incident edge instances. Directed incoming edges contribute;
  every parallel edge contributes and a self-loop contributes once.
- Directed `in_degree`/`out_degree` count incoming/outgoing instances, including
  each self-loop once. On an undirected input both equal incident degree.
- No argument returns ascending selected node IDs, matching `node_ids`.
  An explicit ID sequence preserves its order and duplicates. An empty sequence
  returns an empty numerical array after scope validation.
- Invalid IDs reject the whole request. Older Python degree adapters sometimes
  skipped them, shortening/reordering the effective results; that behavior is
  intentionally removed.
- `full_graph=True` counts edges from the current full owner, while requested IDs
  must still belong to the original valid selection. It cannot bypass liveness.

These rules also repair the older native `SubgraphOperations::degree` override,
which counted only outgoing directed instances. Neighbor traversal continues to
use outgoing occurrences for directed inputs; degree and neighbor length can
therefore differ on a directed graph.

## Native preparation and ownership

`Subgraph::degree_read(DegreeMode, full_graph)` returns `SubgraphDegreeRead`.
Its `counts(&ids)` method reads immutable CSR topology, retains no owner or
property/history payload, and can run on another thread. Python prepares the
input on the owner thread and releases the GIL for row computation. The input
retains that prepared topology if the owner subsequently changes or is dropped;
a new preparation validates the new owner state.

Local incident/outgoing CSR and selected-scope validation reuse the existing
instance/version cache, including restoration with equal numeric versions.
Incoming degree uses incident minus outgoing plus self-loop occurrences. Full
owner preparation explicitly builds topology for the wider domain; it does not
persist that wider cache in the original selection. First/invalidated preparation
costs scale with the selected topology; warm native scalar incident degree keeps
its cached row lookup. Requested rows use sorted-ID lookups.

This closes the reproduced stale-neighbor/degree compatibility defect. It does
not make legacy connected objects immutable by default or complete roadmap 26.1.
