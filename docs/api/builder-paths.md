# Checked path builder helpers

`core.shortest_path_map`, `core.k_shortest_paths` and `core.random_walk` support
property-atomic algorithms, loops and certified for-each bodies. A shared native
preparation path validates active selected nodes/edges and endpoint closure, reads
weights through the staged-property overlay and builds stable selected adjacency.
Directed traversal is outgoing; undirected edges work in both directions, with
stored self-loops once. Excluded owner edges cannot reappear through a cache.

Weights are finite nonnegative numeric scalars, including either stored integer
width and f32/f64. An absent property defaults to 1; explicit null, Boolean, text,
collections, negative or nonfinite weights reject. Validate all selected weights,
including unreachable edges and zero-sized requests. Weighted distances/costs use
f64, with ordinary floating rounding; unweighted distances retain integer values.
An explored nonfinite accumulated cost errors rather than returning infinity.

The following handle uses existing native step specs:

```python
import json
from groggy import Graph, AttrValue
from groggy.algorithms.base import AlgorithmHandle

class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}

def scalar(name, value):
    return {"id": "core.init_scalar", "params": {"target": name, "value": value}}

def expose(source):
    return [
        {"id": "core.init_nodes", "params": {"target": "reference", "value": 0}},
        {"id": "core.broadcast_scalar", "params": {"scalar": source, "reference": "reference", "target": "result"}},
        {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "serialized"}},
    ]

graph = Graph(directed=True)
graph.add_nodes([{} for _ in range(4)])
for u, v, weight in [(0, 1, 10.0), (0, 1, 2.0), (1, 2, 1.0), (0, 2, 8.0), (2, 3, 1.0)]:
    graph.add_edge(u, v, weight=weight)
shortest = {"id": "core.shortest_path_map", "params": {"source": "source", "weight_attr": "weight", "output": "distances"}}
attach = {"id": "core.attach_node_attr", "params": {"source": "distances", "attr": "distance"}}
before = graph.current_state_id
graph.view().apply(Steps([scalar("source", 0), shortest, attach]), persist=True)
assert [graph.get_node_attr(n, "distance") for n in range(4)] == [0.0, 2.0, 3.0, 4.0]
assert graph.current_state_id == before + 1
```

## Reachable distances and staged weights

Shortest-path maps accept a source variable containing an integer node ID or
unsigned decimal ID text. IDs must be nonnegative and active within the selection.
BFS is used without weight_attr; Dijkstra otherwise. Only reachable selected nodes
appear. Parallel arcs are considered separately, so the least-cost arc wins.

```python
steps = [scalar("source", 0),
    {"id": "core.init_edges", "params": {"target": "weights", "value": 4.0}},
    {"id": "core.attach_edge_attr", "params": {"source": "weights", "attr": "weight"}},
    shortest, attach,
]
before = graph.current_state_id
graph.view().apply(Steps(steps), persist=True)
assert [graph.get_node_attr(n, "distance") for n in range(4)] == [0.0, 4.0, 4.0, 8.0]
assert graph.current_state_id == before + 1
```

## K loopless node paths

`k_shortest_paths` takes source/target scalar variable names, nonnegative integer
k, optional weight_attr and output. The output remains a **Text scalar containing
JSON** `[[node_sequence, cost], ...]`; it is not a node map. The example broadcasts
that text explicitly for inspection. Large workloads can use a sparse one-row
reference instead of broadcasting to every entity.

Yen's root/spur procedure excludes earlier root vertices, excludes accepted next
arcs sharing a prefix, deduplicates candidate sequences and uses full f64 cost
ordering. It no longer rounds costs to integer micro-units or lets duplicate
candidates consume a requested result slot. Returned paths are simple (no repeated
nodes), distinct and ordered by cost then node sequence. Equal-cost cutoff choices
are deterministic under stable topology; no global lexicographic tie-rank promise
is made. This implements the loopless-path problem described in
[Yen's original paper](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.17.11.712).

Parallel arcs collapse to their minimum traversal cost for this node-sequence
interface. It does not enumerate distinct edge-ID sequences for the same nodes.
k=0 gives []; unreachable targets give []; positive k with source=target gives
only the zero-cost one-node path. Return fewer than k when no more paths exist.
Source/target/selected weights still validate for k=0.

```python
paths = {"id": "core.k_shortest_paths", "params": {"source": "source", "target": "target", "k": 10, "weight_attr": "weight", "output": "paths"}}
graph.view().apply(Steps([scalar("source", 0), scalar("target", 3), paths] + expose("paths")), persist=True)
assert json.loads(graph.get_node_attr(0, "serialized")) == [[[0, 2, 3], 8.0], [[0, 1, 2, 3], 12.0]]
```

## Seeded edge-incidence walks

`random_walk` accepts a start_nodes variable (node map or scalar node ID),
nonnegative integer length, optional restart_prob in [0,1], weight_attr and seed,
and output. Map keys sort by ID, regardless of their values. Each walk includes
its start and up to length transitions. At dead ends it stops; restart is checked
first, so restart_prob=1 keeps repeating the start even at a dead end. Length=0
returns just each valid start.

Uniform walks sample stored edge incidences, so parallel arcs retain their
multiplicity. Weighted walks retain each arc's weight, including undirected
reverse traversal. Normalize each row by its maximum weight before summing to
avoid overflow on large finite weights. Zero-weight arcs have no weighted chance
when positive weights exist; an all-zero row retains the existing uniform fallback.
Very small relative weights can underflow at f64 precision. Supplied seed uses a
local RNG and neither consumes nor resets the shared seeded generator; unseeded
runs initialize a local RNG without a repeatability promise. Serialized seeds are
nonnegative i64; native constructors accept u64.

```python
walk = {"id": "core.random_walk", "params": {"start_nodes": "source", "length": 3, "restart_prob": 1.0, "seed": 19, "output": "walks"}}
graph.view().apply(Steps([scalar("source", 3), walk] + expose("walks")), persist=True)
assert json.loads(graph.get_node_attr(0, "serialized")) == [[3, 3, 3, 3]]
```

## Validation, cost and remaining work

Supplied selectors and numeric controls require their declared types and ranges;
negative counts/IDs/seeds no longer wrap. Input/kernel/serialization errors and
cancellation preserve previous variables and discard an enclosing atomic property
batch. Persistent writes accept once; transient properties remain owned outputs
and do not simulate later property reads.

The three helpers share selected adjacency/weight preparation, avoiding separate
unchecked pair maps and stale owner-CSR paths. BFS/Dijkstra use frontier/heap maps;
Yen repeats shortest searches per spur and retains candidates, so k can be costly
on graphs with many simple paths. Walks retain every generated node sequence.
Adjacency, path/result vectors and JSON serialization allocate working memory.
Cancellation covers validation, searches, spur loops and walk transitions; row
sorting and serialization do not promise bounded interruption latency. No owner
or history clone is introduced. Private cache-warm timing keys from the previous
implementation are not retained as fictitious work. Release receipts are bounded
native observations, not matched speedups, FFI/RSS or production-scale guarantees.
Temporal helper certification and broader native defaults/storage escapes,
optimizer effects, source pins and standalone sampler transactions remain open.
