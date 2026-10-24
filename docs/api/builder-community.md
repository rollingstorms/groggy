# Checked community builder helpers

`core.community_seed`, `core.label_propagate_step` and `core.modularity_gain`
support property-atomic builders and nested loops. They operate on selected
identities and install owned node columns only after success. Persistent
attachments share one outer publication; transient attachments return owned
columns. These are individual helpers, not a complete community optimizer.

## Seeding

`community_seed` takes strategy, target, optional k and seed. Strategies are
singleton, degree/degree_based, and random. Singleton labels are checked i64 node
IDs. Random requires positive k, assigns labels in [0,k), and uses stable sorted
IDs with a local RNG. Supplied serialized seeds are nonnegative i64 integers;
native constructors accept u64 seeds. Seeded calls neither reset nor consume the
shared thread RNG. Malformed supplied k/seed controls reject, even for an empty
selection; supplied k must be positive for every strategy.

Degree seeding ranks selected nodes by outgoing degree (undirected incidence),
then ascending node ID. Parallel edges count separately and stored loops once.
The first ceil(N/10) nodes, at least one for a nonempty selection, become hubs
with their own IDs as labels. Remaining nodes are visited in ascending ID order
and adopt the first sorted neighbor's already assigned label, or their own ID.
This deterministic initialization is a heuristic, not a community quality bound.

The following low-level handle exposes the existing native step interface:

```python
import json
from groggy import Graph
from groggy.algorithms.base import AlgorithmHandle

class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}

graph = Graph()
graph.add_nodes([{} for _ in range(4)])
for source, target in [(0, 1), (1, 2), (2, 3)]:
    graph.add_edge(source, target)
seed = {"id": "core.community_seed", "params": {"strategy": "singleton", "target": "labels"}}
attach = {"id": "core.attach_node_attr", "params": {"source": "labels", "attr": "community"}}
before = graph.current_state_id
graph.view().apply(Steps([seed, attach]), persist=True)
assert [graph.get_node_attr(n, "community") for n in range(4)] == [0, 1, 2, 3]
assert graph.current_state_id == before + 1
```

## Synchronous label propagation

`label_propagate_step` reads labels and writes target. A pass reads a fixed input
partition; later nodes do not observe earlier output values. Each node receives
one explicit self vote plus one vote per selected outgoing edge (both directions
for undirected edges). Parallel edges vote separately; a stored self-loop adds
one vote in addition to the explicit self vote. Isolated nodes retain their label.
Ties choose ascending canonical typed serialization keys. Nonfinite labels reject.

Both node maps and columns are accepted. Maps output sorted IDs; explicit columns
retain their input order. Sparse input rows remain sparse outputs; an omitted
selected neighbor supplies its own checked node-ID label. Foreign/duplicate input
IDs reject. Source and target may alias without changing synchronous semantics.

```python
propagate = {"id": "core.label_propagate_step", "params": {"labels": "labels", "target": "result"}}
attach_result = {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "propagated"}}
graph.view().apply(Steps([seed, propagate, attach_result]), persist=True)
assert [graph.get_node_attr(n, "propagated") for n in range(4)] == [0, 0, 1, 2]
```

## Actual local modularity gains

`modularity_gain` takes partition and target. Partition labels must be integers.
Both layouts and sparse row/output conventions above apply; omitted selected
nodes are singleton labels given by their IDs. Use noncolliding explicit labels
if sparse fallback IDs would otherwise coincide with a caller's community IDs.

The result is the maximum positive objective change from moving each output node
into another existing community connected by a non-self selected edge; staying
has gain zero. Directed candidate communities include incoming and outgoing
contacts. This does not scan nonneighbor communities, create a fresh singleton,
return the winning destination, or apply moves. Each result is evaluated against
the unchanged input partition. Empty-edge selections yield zero.

The unweighted resolution-1 objectives are:

- Undirected: `Q = sum_C [L_C/m - (K_C/(2m))^2]`.
- Directed: `Q = sum_C [L_C/m - Kout_C*Kin_C/m^2]`.

Here m is stored edge count and L is within-community stored edges. Undirected
loops contribute twice to degree; directed loops once to each in/out degree.
Parallel edges retain multiplicity. These are the standard objectives documented
in the [official modularity reference](https://networkx.org/documentation/networkx-3.4.1/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html).
The local delta implementation is derived by subtracting the objective before
and after a move. Independent native/Python references recompute the full
objective rather than reusing that local formula. Groggy graph APIs construct
and execute all fixtures; no external graph library is used for execution.

```python
single_edge = Graph()
single_edge.add_nodes([{}, {}])
single_edge.add_edge(0, 1)
gain = {"id": "core.modularity_gain", "params": {"partition": "labels", "target": "gain"}}
attach_gain = {"id": "core.attach_node_attr", "params": {"source": "gain", "attr": "gain"}}
single_edge.view().apply(Steps([seed, gain, attach_gain]), persist=True)
assert [single_edge.get_node_attr(n, "gain") for n in range(2)] == [0.5, 0.5]
```

The old helper returned an edge-fraction proxy. Scores now include removal from
the current community and the degree null-model term, so previous proxy values
are deliberately corrected. Weighted/resolution-controlled variants remain
outside this helper's interface.

```python
before = graph.current_state_id
_, profile = graph.view().apply(Steps([seed, propagate, attach_result]), persist=False, return_profile=True)
assert [value for _, value in profile["outputs"]["node_attributes.propagated"]] == [0, 0, 1, 2]
assert graph.current_state_id == before
```

## Costs and boundaries

Singleton/random seeds validate selected nodes without scanning edges. Other
helpers build checked selected adjacency, rejecting inactive entities and edges
whose endpoints escape the selection. Degree seeding sorts hubs; propagation
counts typed categories; modularity precomputes degrees/community totals and
visits incident contacts, avoiding a partition scan for every candidate. Working
storage includes selected adjacency, input labels and output columns. No owner or
history clone is introduced. Cancellation checks surround preparation, node
passes and result installation; individual sorts/large rows do not promise
bounded interruption latency. Release receipts are bounded native observations,
not matched speedups, RSS/FFI or production-scale guarantees. Broader composer,
whole-pipeline and remaining step-family audits stay open.
