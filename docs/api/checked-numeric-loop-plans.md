# Checked numeric loop plans

The serialized `iter.loop.batch_plan` compatibility frontend can now lower a
fully supported numeric program to the same checked kernels as ordinary native
steps. It works in persistent atomic memory execution and in `run_durable`.
This does not make every loop body, IR frontend, or custom step portable.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

initial = GraphBatch(effective_at=0)
initial.add_node()
graph = Graph()
graph.writer().apply_batch(initial)
with TemporaryDirectory() as folder:
    path = Path(folder) / "numeric.journal"
    store = DurableGraph.create_from_graph(path, graph)
    retained = store.read_view()
    head = store.current_state_id
    steps = [
        {"id": "core.init_nodes", "params": {"target": "x", "value": 1e-20}},
        {"id": "core.attach_node_attr", "params": {
            "source": "x", "attr": "x", "effective_at": 10}},
        {"id": "iter.loop", "params": {
            "iterations": 2, "body": [], "batch_plan": {
                "slot_count": 2, "carried_slots": [], "instructions": [
                    {"type": "load_node_attr", "dst": 0, "attr_name": "x", "default": 0.0},
                    {"type": "load_scalar", "dst": 1, "value": 1e-20},
                    {"type": "div", "dst": 0, "lhs": 0, "rhs": 1},
                    {"type": "store_node_prop", "src": 0, "var_name": "answer"},
                ]}}},
        {"id": "core.attach_node_attr", "params": {
            "source": "answer", "attr": "answer", "effective_at": 20}},
    ]
    pipeline = Pipeline([step_pipeline("checked numeric loop", steps, atomic=True)])
    result = pipeline.run_durable(store)
    assert result.write_state_id == head + 1
    assert result.node_attr(0, "answer") == 1.0
    assert retained.node_attr(0, "x") is None
    store.close()
    recovered = DurableGraph.open(path)
    assert recovered.read_view_at(15).node_attr(0, "answer") is None
    assert recovered.read_view_at(20).node_attr(0, "answer") == 1.0
    recovered.close()
```

## Supported numeric instruction inventory

- `load_node_prop`, `store_node_prop`, `load_scalar`, `init_nodes`,
  `load_node_attr`, `graph_node_count`, `graph_edge_count`, `broadcast_scalar`.
- `add`, `sub`, `mul`, `div`, `min`, `max`, `pow`.
- `abs`, `sqrt`, `exp`, `log`, `clip`, `recip`.
- `compare`, `where`, `reduce_scalar`, `mode_list`.
- `normalize` with `sum` or `max`.

These 26 instruction forms use native operand contracts. For example,
`add/sub/mul/div` require a map operand; a scalar broadcast remains explicit.
`mode_list` requires list-valued node inputs. Supplying a supported opcode does
not make invalid input types valid. Legacy literal fields are already `f64`;
this API cannot recover integer bits lost when a frontend creates such a literal.
Loaded integer properties and variable values retain their types, including
integers above 2^53. `store_node_prop` writes a caller variable; an explicit
attribute attachment is what stages a graph property change.

Registers live in a private frame. A caller variable called `slot/0` cannot
collide with register zero. Destinations may alias their input registers.
Between iterations, all carried values are read before any carried destination
is assigned. Duplicate carried destinations reject. Registers initially contain
zero-valued node maps, preserving the legacy initialization convention.

Malformed supplied plans reject, including at zero iterations; they never
silently select a different body. Negative iterations, out-of-range slots,
nonfinite controls, negative epsilon and reversed clip bounds reject. A valid
zero-iteration plan executes no instructions. Tiny nonzero divisors remain
nonzero; true division by zero reports the native error. Numeric comparison uses
exact ordering, including mixed integer/float inputs; equality does not apply an
implicit tolerance. Streaming comparison uses that same comparator.

## Remaining boundary and cost

`load_edge_attr`, `node_degree`, neighbor collection/aggregation/mode, in-place
updates, fused operations, and `normalize(min/mean)` remain outside this lowering.
A plan containing one of these stays uncertified for atomic/durable execution;
its existing non-atomic compatibility route remains available. The disabled JIT
is unchanged. Ordinary nested step-body loops still need a separate portable
composition migration. There is no new pipeline naming, cache or run-history
policy.

Execution uses native map intermediates and a shared pending-read cache.
Register bindings and frame clones share immutable values internally. Mutable
node-map or column access copies the value when it has another reader; mutation
of a uniquely held value stays in place. Exact-membership numeric loads/stores
and carried values share their backing value, while subsets still project
explicitly and column outputs retain their representation. Initial registers
share one zero map; a zero-slot plan allocates no unused zero map.

Copy on write can copy an entire large map or column on its first aliased
mutation. New arithmetic results still allocate their own maps, and membership/
numeric validation still scans input rows. See the
[measured binding follow-up](../../notes/planning/shared-numeric-bindings-2026-11-24.md)
for the benchmark and remaining costs. Acceptance retains the
[durable publication cost](durable-pipelines.md).

Evidence: [native parity tests](../../tests/checked_batch_lowering.rs),
[Python regressions](../../tests/test_checked_batch_lowering.py), and
[completion record](../../notes/planning/checked-batch-lowering-2026-11-24.md).
