# Subgraph computation over one pending graph

The existing `sample.iterate_nodes`, `sample.iterate_edges`, `sample.neighbors`,
`sample.emit_subgraphs` and certified `sample.for_each` bodies can follow pending
topology in an explicit atomic builder. They also compose inside compute loops
and across an outer atomic pipeline.

The batch owns the original execution root. Each child executes with a fixed
node/edge selection. Refreshing a child reads the root's pending values and then
projects that exact selection. It does not import every provisional node or
reject valid assignments made by a sibling.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes(2)
graph.add_edge(0, 1)
head = graph.current_state_id
body = [
    stage("core.load_node_attr", attr="counter", default=0, target="values"),
    stage("core.init_scalar", value=1, target="one"),
    stage("core.add", left="values", right="one", target="next"),
    stage("core.attach_node_attr", source="next", attr="counter"),
    stage("graph.capture_staged", output="pin"),
    stage("graph.emit", source="pin", name="child"),
]
_, profile = graph.apply(step_pipeline("pending edge frames", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("graph.add_edges", edges=[[1, {"added": 0}]], output="edge"),
    stage("sample.iterate_edges", target="seeds"),
    stage("sample.for_each", source="seeds", target="mapped", body=body),
], atomic=True), return_profile=True)
assert graph.current_state_id == head + 1
assert [graph.get_node_attr(n, "counter") for n in [0, 1, 2]] == [1, 2, 1]
first = profile["outputs"]["for_each.mapped.0.child"]
second = profile["outputs"]["for_each.mapped.1.child"]
assert sorted(first.node_ids) == [0, 1]
assert sorted(second.node_ids) == [1, 2]
assert first.node_attr(1, "counter") == 1
assert second.node_attr(1, "counter") == 2
```

## Selection and immutable outputs

Node iteration retains only internal selected edges, including stored self-loops.
Edge iteration retains exactly one selected edge and its endpoints. Neighborhoods
follow the selected graph's direction and stay inside the enclosing frame's edge
scope. Induced neighborhood output includes all internal selected edges, including
a self-loop on a newly reached node. Parallel edge instances remain distinct.

Emission preserves explicit `induced=False` and per-item/unified behavior. Node or
edge maps act as selections by their keys. Membership, owner and endpoint closure
are checked before any child runs. A seed created before a later removal is not
silently filtered into a different selection.

Within a batch, produced child subgraphs are immutable values pinned to the state
at production. Sibling writes do not change them. A child's ordinary loads refresh
working values; an explicit capture/table remains its supplied immutable input.
A missing explicit child output preserves its input; an explicit empty output
emits zero children. Later assignments to overlapping entity/property keys retain
the existing deterministic item-order behavior.

## Shared execution and nested bodies

For-each uses the same native step executor as top-level and loop frames. Exact
frame membership propagates through nested loops and nested for-each calls,
including explicit staged captures and source-bound table attachments. Supported
children share one canonical acceptance; errors or cancellation discard all
staged topology/properties and restore prior outputs.

Loops and for-each now share nested-body parsing and top-level parameter inference.
Empty programs stay empty programs, literal vectors retain their typed values,
and declared dependency names survive parsing. Malformed parameter objects and
non-string dependency arrays reject instead of silently defaulting.

General topology mutations inside child bodies, alternate execution plans,
streaming and uncertified callbacks remain guarded. Legacy owner-reading steps
remain restricted to execution before pending topology. Explicit capture/table
attachment bodies require persistent local atomic execution as before.

## Cost

Frame setup does not clone a writable graph or create per-child history versions.
Pending read materialization currently copies selected root values before exact
projection; repeated child writes can therefore make large runs expensive.
Use the published native benchmark receipt for the measured fixtures. No constant
cost, zero-copy property capture or scale guarantee follows from this capability.
