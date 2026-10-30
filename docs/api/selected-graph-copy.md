# Independent selected graph copies

`view.to_graph()` creates an independent publishing graph containing exactly the
selected nodes, selected edge instances and frozen property values. It preserves
original IDs, direction, parallel instances, loops and explicit null/absence.
Default `Subgraph.to_graph()` and `GraphView.to_graph()` copy their original state
after owner changes. An explicit `graph.connected_view().to_graph()` first
captures and validates its current selection.

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes([{}, {"score": 0.123456789012345}, {}, {}])
graph.add_edge(1, 3, weight=2.5)
graph.add_edge(1, 3, weight=100)
source = graph.read_view().select_edges([0])
copy = source.to_graph()
assert sorted(copy.node_ids) == [1, 3]
assert copy.edge_ids.to_list() == [0]
assert copy.get_edge_attr(0, "weight") == 2.5
assert copy.current_state_id == 1
copy.set_node_attr(1, "score", 9)
assert copy.current_state_id == 2
assert source.node_attr(1, "score") == 0.123456789012345
```

Nonempty copies publish one complete initial import with new engine acceptance
clocks. Empty selections return an empty publishing graph without a synthetic
state. Later mutations use ordinary automatic publication. The copy has default
engine configuration with the source direction; source branches/history, inactive
entities and unrelated identity reservations are outside a selected copy. Use an
explicit owned snapshot for full storage/history/configuration recovery.

Typed NodeArray/EdgeArray references must resolve inside the selected local graph;
otherwise copying rejects instead of retaining dangling references. Ordinary
numbers, vectors and JSON remain opaque values. Selected managed external IDs are
re-established in the independent graph. ID allocation continues above retained
selected IDs; sparse gaps retain the pool's normal allocation cost.

Referenced frozen group sources retain their own qualified node IDs, values and
nested revisions. Only referenced root groups are installed; unrelated source
catalog entries are not imported. Group source nodes do not become active root
nodes merely because a meta-node references them. The independent graph can save,
reopen and inspect those sources after the original owner is gone.

The Rust core reuses snapshot installation, identity validation, complete import
staging and canonical publication. The FFI only wraps the resulting owner. Copying
is an explicit materialization with new buffers and an independent graph history;
it is not a constant-time clone. Current Rc-based construction runs on the owner
thread; no additional GIL-release guarantee is introduced here.
