# Checked subgraph bulk writes

Native `SubgraphOperations::set_node_attrs` and `set_node_attr_column` pass the
complete supplied assignments to the owning graph's validator. Inactive,
unknown and duplicate targets reject the entire call. Original row positions
remain in the error; no active row is written when another row is invalid.
The subgraph layer no longer silently filters out inactive targets. Native
algorithm map/column attachment uses these same entry points.

The native `Subgraph::set_edge_attribute_bulk` helper also sends one complete
edge column to the graph coordinator. On a publishing graph, a successful bulk
node or edge operation accepts one version. Valid empty writes remain no-ops.
On a manual graph, successful ordinary setters retain the existing deferred
publication behavior. Rejected writes do not publish or change pending data.

Python subgraph/accessor bulk setters already validate their explicit targets;
these native changes align algorithm/convenience paths with that rejection
behavior. A stale selection is not an instruction to update only its surviving
members. Choose a fresh selection explicitly when that is the intended update.

```python
from groggy import Graph

graph = Graph()
a, b = graph.add_nodes(2)
selection = graph.induced_subgraph([a, b])
selection.set_node_attrs({"score": {a: 1, b: 2}})
prior = graph.read_view()
graph.remove_node(b)
before = graph.current_state_id
try:
    selection.set_node_attrs({"score": {a: 10, b: 20}})
except ValueError:
    pass
else:
    raise AssertionError("inactive target must reject the complete write")
assert graph.current_state_id == before
assert graph.get_node_attr(a, "score") == 1
assert prior.node_attr(b, "score") == 2
assert not graph.has_uncommitted_changes()
```

This concerns each bulk operation and algorithm output attachment. Full
multi-step pipeline publication boundaries, migration of every native mutation
entry point and legacy table materialization remain separate checklist work.
Historical reads and retained group membership are unchanged.
