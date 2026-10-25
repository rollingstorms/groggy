# Checked builder expressions and transformations

Declarative `core.map_nodes`, `core.neighbor_mode_update` and
`core.update_in_place` support explicit `build(atomic=True)` property staging,
including nested loops. A successful persistent run accepts its property writes
once; an error/cancellation discards them. Native opaque `MapNodesStep` callbacks
remain uncertified: callbacks can access owner objects beyond the checked scope.
Callback errors now propagate instead of silently keeping an existing value.

## Expressions

Python `map_nodes` retains its small expression-string subset: a variable or
numeric literal, `variable +|-|*|/ literal`, selected neighbor values/IDs, and
sum/mean/mode/count over supported inputs. It consumes the entire string and
rejects unsupported trailing text. Integer literals stay integers. For example:

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_nodes([{"score": 2**53}, {"score": 2**53 + 1}])
builder = AlgorithmBuilder("increment")
values = builder.load_attr("score")
result = builder.map_nodes("x + 1", inputs={"x": values})
builder.attach_as("answer", result)
before = graph.current_state_id
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.current_state_id == before + 1
assert graph.get_node_attr(1, "answer") == 2**53 + 2
```

The native structured `Expr` AST additionally supports property reads, binary
add/sub/mul/div/mod/pow, comparisons, logical operations, unary operations,
min/max and conditional expressions. Native tagged constants and literal JSON
constants both decode; integers beyond i64 reject. Mixed numeric comparisons use
exact integer/float ordering. Integer operations are checked; division by an
actual zero and nonfinite scalar/list inputs/results error. Tiny nonzero divisors
remain valid. Float/mixed arithmetic returns f64 with ordinary floating rounding.
Integer list sum uses checked wide accumulation and requires an i64 final result;
float sum/array mean share the checked accumulator used by builder reductions.
An empty array mean is 0. Mode ties use the lowest numeric value; typed empty
IntList/FloatList mode results retain their respective zero types.

`if` evaluates only the selected branch; Boolean `and`/`or` short-circuit.
Conditions accept Booleans or finite numeric values, with zero false. Staged
properties are visible through AST `attr` reads during persistent atomic runs.
Transient attachments are owned outputs, so they do not simulate new properties
for later AST reads.

## Synchronous and asynchronous maps

Synchronous mapping evaluates against the original variables and installs its
result after all nodes succeed. Async mapping updates a private source map in
ascending selected node-ID order: subsequent nodes observe earlier local updates.
Only after success are source and target installed. Python chooses the first
input variable as the async source/output. This is an intentional in-place effect.

```python
graph = Graph()
graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
graph.add_edge(0, 1)
graph.add_edge(1, 2)
builder = AlgorithmBuilder("async_neighbors")
values = builder.load_attr("score")
result = builder.map_nodes("sum(x[neighbors(node)])", inputs={"x": values}, async_update=True)
builder.attach_as("answer", result)
graph.view().apply(builder.build(atomic=True), persist=True)
assert [graph.get_node_attr(n, "answer") for n in range(3)] == [2, 5, 5]
```

Topology reads validate active selected nodes/edges and endpoint closure. Directed
expressions use outgoing edges; undirected edges contribute both ways, with a
stored self-loop once. Neighbor IDs are sorted. `neighbors(node)` and the native
`neighbor_count` preserve edge incidences, including parallel edges.
`x[neighbors(node)]` retains its distinct-neighbor convention: each selected
neighbor, including a stored self-loop, contributes once; missing variable rows
are omitted. Present nonnumeric values reject, and numeric lists flatten. Mixed
integer/float neighbor collections promote to FloatList, so integers then have
f64 precision. The standalone `neighbor_agg` primitive instead retains per-edge
contributions and exact scalar accumulation; choose according to the workload.

```python
builder = AlgorithmBuilder("neighbor_counts")
values = builder.load_attr("score")
counts = builder.map_nodes("count(x[neighbors(node)])", inputs={"x": values})
builder.attach_as("neighbor_count", counts)
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(1, "neighbor_count") == 2
```

## Label updates and explicit copies

`neighbor_mode_update` remains asynchronous. Both ordering settings now use stable
ID traversal; `ordered=False` does not request synchronous evaluation. Directed
votes are outgoing; undirected votes use both directions. Parallel edges vote
separately, stored loops once. `include_self=True` adds a separate self vote.
Missing label rows are omitted. Typed labels use serialized identity; nonfinite
labels reject. `keep` chooses the first winning vote (self first when included,
then sorted neighbors). `lowest`/`highest` require comparable scalar tied winners,
use exact numeric ordering, and break numerically equal typed ties by ascending
serialized type/value key. Supplied flags/tie selectors/output names validate.
Updates and optional output aliases install only after the full pass succeeds.

`update_in_place` copies sparse source values into an existing target, retaining
other target rows. Source IDs must belong to the active selection. Source, target
and output may alias; inputs remain unchanged on validation failure/cancellation.
Its `ordered` argument remains accepted, with stable ID traversal in either mode.

```python
builder = AlgorithmBuilder("copy_values")
source = builder.load_attr("score")
target = builder.init_nodes(default=0)
result = builder.core.update_in_place(source, target=target, ordered=True)
builder.attach_as("copied", result)
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(2, "copied") == graph.get_node_attr(2, "score")
```

## Costs and boundaries

Scalar maps borrow existing variables; async maps clone only the source map once.
Label/copy steps prepare a private target map; distinct output aliases require a
copy. Neighbor expressions build sorted selected adjacency once per map pass;
scalar-only maps avoid the edge scan. Individual expression kernels/neighbor rows
are not a bounded-latency cancellation guarantee. Owner graphs/history are not
cloned. These correctness costs have a bounded native release receipt, without a
matched before/after speedup, RSS/FFI or production-scale claim. Temporal steps and whole outer-pipeline transactions remain under separate
audits. Checked community and bounded sampler
contracts are documented in their corresponding API pages.
