# Degree tables from captured pipeline inputs

`graph.degrees_table(source=..., output=...)` computes a structural summary from
an explicit immutable GraphView variable. It reads that view's selected topology,
including pending endpoint revisions/removals captured earlier in an atomic run.
It never substitutes the writable owner's current topology or liveness.

```python
from groggy import Graph
from groggy.builder import step_pipeline

graph = Graph(directed=True)
graph.add_nodes(3)
graph.add_edges([(0, 1), (0, 1), (1, 1)])
head = graph.current_state_id
_, profile = graph.apply(step_pipeline("pending degree summary", [
    {"id": "graph.remove_edges", "params": {"ids": [0]}},
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.degrees_table", "params": {"source": "pending", "output": "degrees"}},
    {"id": "table.emit", "params": {"source": "degrees", "name": "degrees"}},
], atomic=True), return_profile=True)
table = profile["outputs"]["degrees"]
assert graph.current_state_id == head + 1
assert table.source_entity_ids == [0, 1, 2]
assert table.column("degree").to_list() == [1, 2, 0]
assert table.column("in_degree").to_list() == [0, 2, 0]
assert table.column("out_degree").to_list() == [1, 1, 0]
```

The owned table has four columns in this order:

| Column | Definition |
| --- | --- |
| `node_id` | Original selected node IDs in ascending order, including isolates. |
| `degree` | Selected incident edge instances, including incoming directed edges; each loop counts once. |
| `in_degree` | Incoming directed instances, with a loop once; incident degree when undirected. |
| `out_degree` | Outgoing directed instances, with a loop once; incident degree when undirected. |

Distinct parallel edge IDs contribute separately. Unselected edges do not count,
even when their endpoints are selected. These definitions reuse GraphView's
native degree methods. An existing node property named `degree` is not used or
modified. This does not alter the older `core.node_degree` builder primitive's
contract or execution-source selection.

The result retains exact source IDs and the captured view as provenance. A
pending source remains provisional, with no accepted-state write pin. Editing a
returned Python table does not change its captured source or the owner graph.
An empty selection returns shape `(0, 4)`. Integer conversion is checked rather
than truncating IDs/counts into table cells.

This is a certified immutable-input computation. It can follow staged topology,
or use a retained historical source on a graph whose current entities differ.
Unlike the topology/capture stages, the degree-table stage itself requires no
local atomic scope or persistence. It also supports the existing property-only
outer atomic scope when its source is already explicit. It adds no delta by
itself. A late type/selection/projection error discards the surrounding local
atomic run's staged writes and produced outputs.

Computation stays in Rust and materializes four owned columns. Large inputs can
fill independent rows through the existing native worker pool; writable owners
and Context remain on their execution thread. It traverses the
captured selected incidence/adjacency rows and does not copy unrelated property
columns. Sorting selected IDs, owner-wide incidence rows retained by a narrow
view, and materializing output columns remain real costs. Cancellation is checked
per result node and around the stage; one high-degree row is not interruptible.
The pipeline still runs on the owner thread. General algorithms over staged
scopes, additions, broader lowering and execution policies remain unfinished.
