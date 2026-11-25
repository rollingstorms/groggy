# Checked numeric and topology loop plans

The serialized `iter.loop.batch_plan` compatibility frontend can now lower a
fully supported numeric/topology program to the same checked kernels as ordinary native
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

## Supported instruction inventory

- `load_node_prop`, `store_node_prop`, `load_scalar`, `init_nodes`,
  `load_node_attr`, `graph_node_count`, `graph_edge_count`, `broadcast_scalar`.
- `add`, `sub`, `mul`, `div`, `min`, `max`, `pow`.
- `abs`, `sqrt`, `exp`, `log`, `clip`, `recip`.
- `compare`, `where`, `reduce_scalar`, `mode_list`.
- `normalize` with `sum` or `max`.
- `node_degree`, `collect_neighbor_values`, `neighbor_aggregate`, `neighbor_mode`.
- `update_in_place`, `neighbor_mode_update`.
- `fused_neighbor_mul_agg`, `fused_madd`, `fused_axpy` (the historical serialized
  spellings `fused_m_a_d_d` and `fused_a_x_p_y` remain accepted).


These 35 instruction forms use native operand contracts. For example,
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

`load_edge_attr` and `normalize(min/mean)` remain outside this lowering.
The edge-attribute instruction needs an edge-aware register result contract;
ordinary `core.load_edge_attr` already works in durable programs. A plan containing
an unlowered instruction stays uncertified for atomic/durable execution; its
existing non-atomic compatibility route remains available. The disabled JIT is
unchanged. Ordinary nested step-body loops still need a separate portable
composition migration. There is no new pipeline naming, cache or run-history
policy.

The Python compiler keeps graph-default neighbor expressions on their ordinary
native expression path: that direction depends on graph kind and must not be
rewritten as unconditional incoming traversal. Unsupported expression operations
fall back as a whole; the compiler never omits them from a partial plan. Supported
arithmetic expressions now bind their allocated output register as well as their
named variable. Per-node constants remain node maps, including across iterations.


### Topology and update semantics

Explicit `in`, `out` and `undirected` directions operate on stored endpoints, on
either graph kind. Collection without a direction and asynchronous label updates
use outgoing neighbors for directed graphs and both directions for undirected
graphs. Neighbor rows are sorted by node ID. Parallel edges contribute separate
incidences; a self-loop contributes once. `node_degree` counts outgoing incidences
on directed graphs and both endpoints on undirected graphs, counting a self-loop
once. These are the existing native step conventions.

`neighbor_mode` is synchronous: it reads one input map. `neighbor_mode_update`
is asynchronous: earlier updates are visible to later nodes in stable ID order.
Both values of its legacy `ordered` flag currently use that deterministic order;
false does not promise random traversal. `include_self` adds the node's value
in addition to any stored self-loop. `first` chooses the first modal candidate in
that ordered list; `lowest` and `highest` compare integer labels exactly, including
adjacent integers above 2^53. Incomparable modal ties reject for those two policies.
Nonfinite numeric list entries reject rather than being dropped or made null.
Numerically equal labels with different representations use their JSON spelling
as a deterministic secondary key for lowest/highest ties.

Empty-neighborhood sum/mean return zero; min/max and list mode return `None`.
The native steps can explicitly attach that value as null. Numeric
`store_node_prop` still requires numeric rows, so storing a null, text or list
result through that numeric instruction rejects; it does not silently use zero.
Register stores update variables, while explicit attachments stage graph changes.

Fused kernels retain their native **f64** arithmetic. They do not promise exact
integer arithmetic or compensated accumulation. Invalid nonfinite inputs and
nonfinite results reject before installing a result. Sum uses the existing
specialized neighbor multiplication kernel; mean/min/max compose checked
multiplication and neighbor reduction. The latter are supported semantics, with
an intermediate allocation rather than a promised fused speedup.

Pending endpoint revisions and removals feed the same topology reads in memory
and durable runs. Successful mixed programs publish once; failed computations
and cancellation publish no prefix. Retained views and effective-time recovery
keep their existing meaning.

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
for the benchmark and remaining costs. Selected topology construction is still
linear in selected nodes/edges; the checked neighbor path builds sorted adjacency
per invocation. Migrated collection therefore no longer uses its separate
unordered CSR cache. There is no claim of improved neighbor throughput; topology
cache sharing and measurement remain in the existing performance scope. Acceptance retains the
[durable publication cost](durable-pipelines.md).

Evidence: [native parity tests](../../tests/checked_batch_lowering.rs),
[Python regressions](../../tests/test_checked_batch_lowering.py), and
[numeric completion record](../../notes/planning/checked-batch-lowering-2026-11-24.md), and
[topology completion record](../../notes/planning/checked-topology-plans-2026-11-24.md).
