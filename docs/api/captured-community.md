# Communities on immutable graph views

`GraphView` supports Louvain, Leiden, Infomap, Girvan–Newman and label propagation.
These methods use the existing Rust community kernels. They return owned rows in
ascending node-ID order, including selected isolates, and never attach a property
or publish a delta. Input comes from the view's captured node/edge membership,
direction, seed labels and edge lengths. No working graph owner is required.

```python
import groggy as gr

graph = gr.Graph()
a, b, isolated = graph.add_nodes(3)
graph.add_edge(a, b)
graph.set_node_attr(a, "seed", "group")
graph.set_node_attr(b, "seed", "group")
view = graph.read_view()
head = graph.current_state_id
assert len(view.louvain()) == 3
assert len(view.leiden(seed=7)) == 3
assert len(view.infomap(seed=7)) == 3
assert len(view.girvan_newman(num_levels=0)) == 3
labels = dict(view.label_propagation(seed_attr="seed", seed=7))
assert labels[a] == labels[b] == "group"
assert graph.current_state_id == head
assert not view.has_node_attribute(a, "community")
```

## Controls and existing algorithm boundaries

All controls are keyword arguments in Python. `None` for an optional iteration
count selects the default shown below. Boolean, negative, fractional and
out-of-range counts/seeds are rejected. Iteration budgets must be positive;
Girvan–Newman `num_levels=0` deliberately returns initial connected components.
Seeds are unsigned 64-bit integers; omitting a seed preserves the existing
algorithm's deterministic default.

| Method | Controls and defaults | Domain |
|---|---|---|
| `louvain` | `max_iter=20`, `max_phases=10`, `resolution=1.0` | Undirected; unit edge-instance weights. |
| `leiden` | `max_iter=20`, `max_phases=10`, `resolution=1.0`, `seed=None`, `theta=0.01` | Undirected; existing bounded refinement implementation. |
| `infomap` | `teleportation=0.15`, `num_trials=10`, `max_iter=100`, `flow_max_iter=10000`, `seed=None` | Directed or undirected; existing bounded two-level map equation, with convergence errors preserved. |
| `girvan_newman` | `num_levels=None`, `modularity_threshold=None`, `weight_attr=None` | Undirected, intended for small graphs. Named lengths must be positive finite numbers; missing lengths default to one. Modularity uses unweighted original edge instances. |
| `label_propagation` | `max_iter=20`, `tolerance=0.0`, `seed_attr=None`, `seed=None` | Directed outgoing or undirected neighbors; asynchronous assignments. Arbitrary seed values retain their types. |

Parallel selected edges and loops follow the existing kernel's semantics.
Community numbers are algorithm-local identifiers; equal numbers from different
runs or algorithms do not define the same group. These methods return labels,
not retained meta-nodes or new group revisions.

## Frozen input and exact selection

Selecting edges keeps those instances only. Selecting nodes on a captured view
keeps the parent's internal edges and includes selected isolates. A captured
algorithm never reads excluded edges or later owner changes.

```python
edge = view.edge_ids[0]
selected = view.select_edges([edge])
expected = selected.label_propagation(seed_attr="seed")
graph.set_node_attr(a, "seed", "changed")
graph.remove_edge(edge)
del graph
assert selected.label_propagation(seed_attr="seed") == expected
assert selected.node_ids == [a, b]
```

The same methods work on `read_view_at(effective_at, recorded_by=...)`, retained
state views and captured group views. Historical seed/weight values remain pinned
through corrections, storage close/reopen and owner release. Python releases the
GIL during input preparation and computation; owned labels are translated after
computation.

## Native reuse and cancellation

Rust callers can prepare `view.algorithm_input()` once, then call its five
community methods with existing algorithm objects and a mutable `Context`. This
reuses the captured CSR and ID mapping across computations. Context supplies
telemetry/cancellation only: neither `persist_results=true` nor a staged working
property overlay changes captured values or causes writes. Cancellation also
rejects empty inputs. Use the existing checked view-batch API for explicit
writeback against an accepted source state.

Convenience methods prepare the captured CSR per call. Louvain, Leiden, Infomap
and Girvan–Newman additionally prepare their existing specialized kernel inputs;
this costs memory/time proportional to selected topology. No full `Graph`, pool,
retained history or writable owner is cloned. See `benches/captured_communities.rs`
for connected, convenience and reused-input measurements.

This completes the captured input surface for the thirteen classic registry
algorithms. Connected `Subgraph` defaults and builder outputs still have their
existing live ownership; their complete immutable-default migration remains open.
