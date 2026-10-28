# Builder optimization and statement order

`build()` defaults to optimization. Optimizations must preserve the ordered input
program, including operand positions and property read-after-write behavior.
Ordinary algorithm and sampler builds keep the original program when an
output-free effect precedes a later computation. A terminal output suffix can
still follow an optimized value program. `build(atomic=True)` continues to retain
all original steps and opts into one checked property publication.

```python
from groggy import Graph, Pipeline
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_node(score=7)
builder = AlgorithmBuilder("read after write")
builder.attach_as("score", builder.init_nodes(default=2))
builder.attach_as("copy", builder.load_attr("score"))
# The child uses its default local mode; the outer run stages both changes.
head = graph.current_state_id
Pipeline([builder.build()])(graph.view(), atomic=True)
assert graph.get_node_attr(0, "copy") == 2
assert graph.current_state_id == head + 1
```

Common-expression reuse preserves input positions even for commutative operators.
Subtraction, division, powers, comparisons and conditional branches cannot be
identified by sorting their operands. Reuse is limited to known value operations;
graph/property reads and opaque operations need richer source/effect identity.
Execution/control blocks are observable effects for dead-code elimination.

```python
builder = AlgorithmBuilder("operand order")
a, b = builder.init_nodes(default=10), builder.init_nodes(default=2)
builder.attach_as("result", (a - b) + (b - a))
graph.view().apply(builder.build(), persist=True)
assert graph.get_node_attr(0, "result") == 0
```

These repairs do not make ordinary runs atomic. Without explicit atomic staging,
earlier accepted writes can remain after a later runtime failure. `persist=False`
retains owned-output semantics: property loads still read the original graph, not
transient attachments. No new timestamps, deltas or history arise from compiling.

The conservative ordered-program fallback can forgo fusion for interleaved effect
programs. Complete typed effects, alias/redefinition identity, all lowering paths,
custom plans and compound topology acceptance remain existing AR7.8/30.1/09.4b.2
work. This guide does not certify every legacy IR optimization or builder method.
