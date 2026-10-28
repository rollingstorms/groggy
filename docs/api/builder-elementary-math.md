# Checked elementary builder math

`builder.core.abs`, `sqrt`, `exp`, `log`, `min` and `max` now execute through
registered native steps. Unary operations share the existing checked expression
evaluator. Element-wise extrema share the exact numeric comparator and retain
the selected input value/type. Python performs lowering and variable resolution.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_node()
builder = AlgorithmBuilder("elementary math")
builder.attach_as("absolute", builder.core.abs(builder.init_nodes(default=-9007199254740993)))
builder.attach_as("root", builder.core.sqrt(builder.init_nodes(default=9)))
builder.attach_as("exponential", builder.core.exp(builder.init_nodes(default=0)))
builder.attach_as("logarithm", builder.core.log(builder.init_nodes(default=16), base=4))
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(0, "absolute") == 9007199254740993
assert graph.get_node_attr(0, "root") == 3.0
assert graph.get_node_attr(0, "exponential") == 1.0
assert graph.get_node_attr(0, "logarithm") == 2.0
```

Unary operations preserve scalar/node-map/edge-map kind. Integer absolute values
remain exact checked i64; abs(i64::MIN) rejects. Square root, exponential and
logarithm return checked finite f64. Invalid real domains, nonnumeric or nonfinite
values/results reject. Exponential underflow to finite zero is allowed. Logarithm
is natural by default; an explicit base must be numeric, finite, positive and
different from one. Base validation occurs before execution, even on empty maps.

Extrema accept matching maps or a scalar on either side; scalar pairs stay scalar.
They compare i64/f64 values exactly, reject nonnumeric/nonfinite inputs, and choose
the left operand on numerical ties, retaining its type. They are element-wise
operations; use explicit reductions to find a column's minimum or maximum.

```python
builder = AlgorithmBuilder("exact extrema")
values = builder.init_nodes(default=9007199254740993)
builder.attach_as("minimum", builder.core.min(values, 9007199254740992.0))
builder.attach_as("maximum", builder.core.max(values, 9007199254740992.0))
graph.view().apply(builder.build(), persist=True)
assert graph.get_node_attr(0, "maximum") == 9007199254740993
assert isinstance(graph.get_node_attr(0, "maximum"), int)
assert isinstance(graph.get_node_attr(0, "minimum"), float)
```

Map scope/key behavior follows the existing registered arithmetic steps. Empty
maps yield empty maps. Complete checked outputs replace prior step variables only
on success; unary kernels check cancellation before/during computation and before
publishing their output. Explicit property-atomic runs discard earlier staged
attachments on failure. Ordinary runs retain their separate-publication policy.

```python
head = graph.current_state_id
builder = AlgorithmBuilder("invalid square root")
builder.attach_as("prefix", builder.init_nodes(default=9))
builder.attach_as("invalid", builder.core.sqrt(builder.init_nodes(default=-1)))
try:
    graph.view().apply(builder.build(atomic=True), persist=True)
except RuntimeError:
    pass
else:
    raise AssertionError("Real square root must reject a negative input")
assert graph.current_state_id == head
assert graph.get_node_attr(0, "prefix") is None
```

Legacy/IR and optimized/unoptimized registered step pipelines are covered. This
completion does not certify alternate batch executors, all mathematical facades,
all numeric fusion/folding domains or the broader typed-effects contract in
AR7.8/30.1. See also [checked power](builder-power.md).


## Tiny divisors

Registered builder division rejects exact positive or negative zero. A finite
nonzero divisor remains valid even below machine epsilon, including subnormals.
For example, equal tiny values divide to one:

```python
builder = AlgorithmBuilder("tiny division")
values = builder.init_nodes(default=1e-300)
builder.attach_as("ratio", values / 1e-300)
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.get_node_attr(0, "ratio") == 1.0
```
