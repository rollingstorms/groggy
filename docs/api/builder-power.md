# Checked builder power

`builder.core.pow(base, exponent)`, `values ** exponent` and `base ** values`
execute through the native registered `core.pow` step. The ordinary, legacy and
IR builder paths share the checked expression evaluator; Python only lowers the
operation and resolves variable names.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_node()
builder = AlgorithmBuilder("power")
values = builder.init_nodes(default=3)
builder.attach_as("exact", values ** 34)
builder.attach_as("reverse", 2 ** values)
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(0, "exact") == 16677181699666569
assert graph.get_node_attr(0, "reverse") == 8
```

Node and edge maps accept matching maps or a scalar on either side. A scalar pair
produces a scalar result; broadcast it explicitly when a node column is needed.
Map kind, missing-entry and empty-map behavior follows the existing arithmetic
step contract. Results use the left map's keys; mixed node/edge operands reject.

```python
builder = AlgorithmBuilder("scalar power")
reference = builder.init_nodes(default=0)
scalar = builder.core.pow(3, 4)
builder.attach_as("broadcast", builder.core.broadcast_scalar(scalar, reference))
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(0, "broadcast") == 81
```

Integer bases with nonnegative integer exponents produce checked i64 results.
Integer overflow or an exponent outside the checked u32 exponent range rejects;
it does not silently round/promote. Negative exponents and floating operands use
checked f64 real arithmetic. Nonfinite inputs/results, invalid real powers and
nonnumeric values reject. `0 ** 0` is one, matching the existing expression kernel.
This operation does not produce complex powers.

```python
head = graph.current_state_id
builder = AlgorithmBuilder("overflow")
builder.attach_as("prefix", builder.init_nodes(default=9))
builder.attach_as("invalid", builder.init_nodes(default=2) ** 63)
try:
    graph.view().apply(builder.build(atomic=True), persist=True)
except RuntimeError:
    pass
else:
    raise AssertionError("Checked i64 power must reject overflow")
assert graph.current_state_id == head
assert graph.get_node_attr(0, "prefix") is None
```

Power follows existing selected-scope and property-overlay rules. Explicit atomic
runs discard preceding staged writes on failure; ordinary runs retain their
existing separate-publication policy. Alternate batch-plan executor certification,
other unfinished math facades and the complete typed-effects contract remain
AR7.8/30.1 work. This guide covers the registered step pipeline paths tested here.
