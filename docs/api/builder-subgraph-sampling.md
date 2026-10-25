# Bounded subgraph sampler selectors

`sample.iterate_nodes`, `sample.iterate_edges`, `sample.neighbors` and
`sample.emit_subgraphs` are certified inside property-atomic builder algorithms
and loops. They construct selection objects without changing graph topology or
publishing a version. Later property attachments use the enclosing atomic batch.
The standalone sampler pipeline still rejects atomic=True; its whole execution
boundary, including nested for-each bodies, is separate work.

## Selection scope and ownership

Each invocation prepares active selected nodes/edges and validates endpoint
closure. Every source selector must share the exact owner and stay within this
selection, including its own edge endpoint closure. Foreign owners are rejected
even when their numeric IDs coincide. Induction only uses selected edges, so
parallel edges/self-loops excluded by a parent cannot reappear.

Outputs are **legacy selectors connected to the owner**, with independently owned
membership sets. They are not pinned immutable value snapshots. Later owner
property changes remain visible through these selectors. The broader migration
to pinned views/source provenance remains open; use the existing explicit pinned
read APIs when retained values are needed.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph(directed=True)
graph.add_nodes([{} for _ in range(4)])
first = graph.add_edge(1, 2)
graph.add_edge(1, 2)  # excluded parallel edge
graph.add_edge(1, 1)  # excluded self-loop
graph.add_edge(2, 3)  # outside the parent selection
parent = graph.subgraph_from_edges([first])
builder = AlgorithmBuilder("bounded")
seeds = builder.iterate_nodes()
expanded = builder.neighbors(seeds, hops=1)
builder.emit_subgraphs(expanded, induced=True)
before = graph.current_state_id
samples = parent.sample(builder.build_sampler())
assert [(sorted(s.node_ids), sorted(s.edge_ids)) for s in samples] == [([1, 2], [first]), ([2], [])]
assert graph.current_state_id == before
```

## Iteration and neighborhoods

Node iteration emits one singleton per ascending selected node ID, including only
its selected self-loop edges. Edge iteration emits one selector per ascending
selected edge ID with that exact edge and its endpoints. Parallel edges remain
separate seeds.

Neighborhood expansion accepts a subgraph array, node-map keys (one node per seed)
or edge-map keys (the edge endpoints per seed). Map values do not filter keys.
Arrays retain input order; map seeds sort by ID. Validate every source before
installing results. Source and target may alias.

Expansion uses outgoing edges on directed graphs and both directions on
undirected graphs. Each result contains seed nodes and all nodes reachable within
hops, then induces selected edges on that visited set. At hops=0, seed nodes still
receive all selected edges between them. Negative/malformed hops reject. An empty
frontier stops expansion even for a large hop limit. Empty seed arrays return no
results; an explicit empty selector produces one empty result.

```python
builder = AlgorithmBuilder("mask_neighbors", use_ir=False)
mask = builder.sample_nodes(count=2, seed=19)
expanded = builder.neighbors(mask, hops=1)
builder.emit_subgraphs(expanded, induced=False)
samples = parent.sample(builder.build_sampler(optimize=False))
assert len(samples) == 2
assert all(set(s.node_ids) <= {1, 2} and set(s.edge_ids) <= {first} for s in samples)
```

## Emission and induction

`emit_subgraphs` accepts node maps, edge maps or selector arrays. Mode is per_item
(default, with per-item alias) or unified. Per-item map output is in sorted ID
order; array output retains its input order. Unified array emission unions its
memberships. `induced=True` applies consistently to every source and mode:
selected edges with both endpoints in each output node set are included.

With induced=False, node-map selectors have no edges, edge-map selectors preserve
sampled edges, and array selectors preserve their exact edges. With induced=True,
an edge seed can include additional parallel edges/self-loops only when those
edges also belong to the invocation selection. Direct iteration remains exact;
a later emission can deliberately induce it.

```python
builder = AlgorithmBuilder("exact_edge_seeds")
seeds = builder.iterate_edges()
builder.emit_subgraphs(seeds, induced=False)
samples = parent.sample(builder.build_sampler())
assert [(sorted(s.node_ids), sorted(s.edge_ids)) for s in samples] == [([1, 2], [first])]
```

Empty map conventions remain: per_item returns no selectors; unified returns one
empty selector. An empty selector array returns no selectors in either mode.
Supplied mode and induced controls require the declared text/Boolean types.
Malformed controls, inactive IDs, foreign rows/owners and invalid endpoint closure
reject before result installation. Previous output variables survive failure or
cancellation. Standalone sampler validation recognizes subgraph outputs without
issuing a false missing-property-attachment warning.

## Atomic property composition

These selection steps may appear in property-atomic builder algorithms/loops even
though standalone sampler pipelines have a separate execution contract:

```python
builder = AlgorithmBuilder("selectors_and_properties")
seeds = builder.iterate_nodes()
builder.emit_subgraphs(builder.neighbors(seeds), induced=False)
values = builder.init_nodes(default=7)
builder.attach_as("answer", values)
before = graph.current_state_id
parent.apply(builder.build(atomic=True), persist=True)
assert graph.current_state_id == before + 1
assert [graph.get_node_attr(n, "answer") for n in [1, 2]] == [7, 7]
assert graph.get_node_attr(3, "answer") is None
```

## Cost and next boundary

Prepare the selected adjacency/endpoints once per step, then induce from visited
rows rather than scan the whole owner graph for each seed. Iteration allocates
one selector per entity. Per-seed expansion/emission costs scale with visited
membership/incidences and result count; overlapping outputs duplicate membership
storage. Row/ID sorting and content IDs add work. Large hops can still visit the
entire invocation selection. Cancellation covers preparation, traversal and final
installation; allocation/sorting do not promise bounded interruption latency.
No owner/history clone or mutation is introduced. Bounded native release receipts
are not matched speedups, RSS/FFI or production-scale guarantees.

Nested sample.for_each remains uncertified. Its legacy executor truncates an
explicit child array to its first result and treats an explicit empty array as
fallback input. Child-effect preflight, output flattening, transient output
ownership and its transactional boundary are the next audit. This delivery does
not certify that path or close the view/pipeline parent workstreams.
