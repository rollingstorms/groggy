# Composer presets, explicit targets and contraction

This completes the three existing composer requirements in **09.4b.2b**.
Composition still stages the complete result, validates it and accepts one batch.
Failure leaves IDs, source nodes/edges, retained groups and versions unchanged.

## Presets and semantic identity

Python collapse now distinguishes omitted options from explicit overrides.
Preset defaults apply first; supplied edge_strategy, include_edge_count,
mark_entity_type and entity_type override them. Without a preset, omitted options
retain their previous defaults. The same forwarding applies to component/subgraph
arrays; those retain their existing per-selection acceptance behavior.

The native ComposerOptions type and collapse_with_options provide the same
resolution. Existing fully specified native collapse and MetaNodePlan builders
remain available. Explicit edge aggregation overrides win by attribute name.

A custom/preset entity_type describes a semantic label. The created node always
keeps entity_type=meta, preserving expansion and type checks. Social_network uses
community, org_hierarchy uses department, and flow_network uses junction. Explicit
entity_type=meta requests no additional domain label. Other nonempty names are
merged into an existing text-vector labels aggregation; blank names reject.
mark_entity_type controls generated edge markers and does not erase mandatory
node identity or an explicitly requested semantic label.

```python
from groggy import Graph

graph = Graph()
graph.add_nodes(3)
graph.add_edges([(0, 2, {"weight": 2.0}), (1, 2, {"weight": 4.0})])
before = graph.current_state_id
meta = graph.induced_subgraph([0, 1]).collapse(preset="org_hierarchy")
assert graph.current_state_id == before + 1
assert graph.get_node_attr(meta.id, "entity_type") == "meta"
assert graph.node_labels(meta.id) == ["department"]
edge = next(e for e in graph.edge_ids if meta.id in graph.edge_endpoints(e))
assert graph.get_edge_attr(edge, "weight") == 6.0
assert graph.get_edge_attr(edge, "edge_count") is None
assert set(meta.expand().node_ids) == {0, 1}
```

## Explicit meta-edge targets

collapse(meta_targets=None) preserves automatic connectivity to existing meta
nodes. meta_targets=[ids...] includes only those existing external meta targets;
meta_targets=[] suppresses every existing meta target. Ordinary external nodes
remain eligible. Targets must be distinct, active meta nodes outside the selected
nodes. Unknown, inactive, duplicate, ordinary or internal targets reject before
writes. Supplied unconnected valid targets do not invent edges.

Native EdgeAggregationConfig::with_meta_targets and MetaEdgeStrategy::selected
provide the same selector. The legacy bare Explicit policy still errors without
a target sequence. MetaEdgeStrategy gains a Selected variant and
ExternalEdgeStrategy gains Contract; downstream exhaustive enum matches need to
handle these additional variants. Existing configuration struct fields are intact.

```python
outer = graph.induced_subgraph([2]).collapse(edge_strategy="drop_all")
graph.add_edge(0, outer.id)
selected = graph.induced_subgraph([0, 1]).collapse(meta_targets=[])
pairs = [graph.edge_endpoints(e) for e in graph.edge_ids]
assert (selected.id, 2) in pairs
assert (selected.id, outer.id) not in pairs
```

## contract_all: one edge per reachable external pair

The owner selected one derived edge per reachable external endpoint pair.
contract_all now implements that connectivity rather than aggregate-to-meta.
Directed A → selected nodes → B produces A → B. Undirected results use increasing
endpoint-ID order. A and B must be distinct; self-return routes are not emitted.
Both boundary crossings are required; the internal segment may have zero edges
when the crossings meet at one selected node.

Internal traversal uses exactly the selection's internal edge set. Omitted owner
edges cannot reconnect disconnected selected pieces. Boundary crossings use active
owner edges incident to selected nodes, honoring the meta-target policy. Directed
routes preserve orientation. Cycles terminate by visited-node reachability;
parallel routes produce one derived edge per pair. Pre-existing direct external
edges are retained as separate facts with their original IDs.

The summary meta-node remains isolated. node_strategy=extract keeps original
nodes/edges too; collapse removes them and their incident edges in the same batch.
Retained group membership still contains the original selected internal entities.

```python
flow = Graph(directed=True)
flow.add_nodes(4)
for source, target, capacity, amount in [(0, 1, 8, 9), (1, 2, 5, 7), (2, 3, 6, 3)]:
    flow.add_edge(source, target, capacity=capacity, flow=amount)
head = flow.current_state_id
summary = flow.induced_subgraph([1, 2]).collapse(preset="flow_network", node_strategy="collapse")
assert flow.current_state_id == head + 1
assert flow.node_labels(summary.id) == ["junction"]
assert set(flow.node_ids) == {0, 3, summary.id}
edge = flow.edge_ids[0]
assert flow.edge_endpoints(edge) == (0, 3)
assert flow.get_edge_attr(edge, "capacity") == 5
assert flow.get_edge_attr(edge, "flow") == 19
assert all(summary.id not in flow.edge_endpoints(e) for e in flow.edge_ids)
```

## Attribute aggregation and limits

The implementation defines a deterministic **reachability corridor** for each
output pair: boundary entries that can reach the destination, exits reachable
from the source, and selected internal edges reachable from the source whose
other end can reach the destination. Undirected edges can participate either way.
Each original edge contributes once, including parallel instances and selected
loops. Cyclic detours in that corridor contribute too; this is not simple-path
enumeration or path-frequency weighting.

Configured edge reducers apply to each attribute's participating observations,
ordered by original edge ID. Missing attribute observations remain absent. Default
unlisted attributes use the existing First policy. edge_count, when enabled, is
the distinct contributing-edge count; min_edge_count uses that count. source and
target metadata describe the actual new external endpoints. Managed external IDs
cannot be duplicated. Reducer failures reject the whole composition.

Flow_network's min capacity / summed flow are these descriptive column reductions.
They do not establish an equivalent max-flow capacity or enforce conservation.
Preview counts remain unbound estimates; execution determines actual connectivity.

Reachability avoids exponential path enumeration but output may still contain
quadratically many external pairs. For S entry endpoints and T exit endpoints,
the current bounded-memory implementation can traverse the selected topology for
each pair, plus aggregate its contributing attributes. It scans owner edges to
find boundary ports. Memory includes selected adjacency, visited sets and staged
output rows/payloads; large dense boundaries can be expensive. There is no new
cancellation/bounded-latency or billion-node claim. Bounded release measurements
are recorded in composer-completion-2026-10-24.md.
