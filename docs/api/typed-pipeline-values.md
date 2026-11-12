# Typed graph and table pipeline values

Registered native stages can carry immutable graph views and tables between
steps and return them through `profile["outputs"]`. They use the existing step
interpreter, captured graph readers and columnar table operations.

`groggy.builder.step_pipeline(name, steps, atomic=False)` wraps native StepSpecs
in the existing `BuiltAlgorithm`. Each row has `id` and `params`; optional
`inputs`/`outputs` metadata is retained. The factory copies the supplied program,
preserves statement order and native aliases, and delegates validation to Rust.
It bypasses legacy DSL encoding and optimization. Native callers can use the
same registry, `StepValue::GraphView`, `StepValue::Table` and typed
`StepVariables::graph_view`/`table` accessors.

## Retained state → selection → table → outputs

```python
from groggy import Graph
from groggy.builder import step_pipeline

typed_graph = Graph(directed=True)
typed_graph.add_nodes([{"x": 11}, {}, {"x": None}, {"x": 44}])
typed_graph.add_edges([(0, 2), (0, 2), (2, 3)])
saved_state = typed_graph.current_state_id
typed_graph.set_node_attr(0, "x", 99)
typed_graph.remove_node(2)
head = typed_graph.current_state_id

typed_program = [
    {"id": "temporal.snapshot_at", "params": {"state_id": saved_state, "output": "saved"}},
    {"id": "graph.from_snapshot", "params": {"source": "saved", "output": "historical"}},
    {"id": "graph.select_nodes", "params": {"source": "historical", "ids": [2, 0], "output": "selected"}},
    {"id": "graph.nodes_table", "params": {"source": "selected", "output": "nodes"}},
    {"id": "table.select_columns", "params": {"source": "nodes", "columns": ["node_id", "x"], "output": "projected"}},
    {"id": "graph.emit", "params": {"source": "selected", "name": "graph"}},
    {"id": "table.emit", "params": {"source": "projected", "name": "table"}},
]
typed_handle = step_pipeline("historical projection", typed_program, atomic=True)
_, typed_profile = typed_graph.apply(typed_handle, persist=False, return_profile=True)
typed_output = typed_profile["outputs"]
assert sorted(typed_output["graph"].node_ids) == [0, 2]
assert sorted(typed_output["graph"].edge_ids) == [0, 1]
assert typed_output["table"].column("x").to_list() == [11, None]
assert typed_output["table"].source_entity_ids == [0, 2]
assert typed_graph.current_state_id == head
```

The explicit snapshot supplies the complete retained state, independently of
the current execution selection or in-flight attachments. `graph.from_snapshot`
shares its immutable native source; it creates no temporary writable graph,
checkpoint or delta. It does not copy the full graph on each alias/selection.
The temporal snapshot stage still pays its existing retained-state reconstruction
cost. Graph outputs retain topology, property values, original IDs and retained
group payloads after the owner changes or is dropped. They are read objects;
this conversion grants no current-owner write provenance.

Tables are owned columns with captured row identity/provenance. Editing a Python
table result changes neither its graph source nor another emitted table.
Explicit table writeback retains the existing source/identity checks. Missing
properties are aligned null cells; source readers still distinguish absence.

## Fixed stage inventory

All stages require a `source` variable. Intermediate-producing stages require
`output`; emit stages require the explicit result `name`.

| Stage | Input → output | Additional parameters |
|---|---|---|
| `graph.from_snapshot` | TemporalSnapshot → GraphView | None |
| `graph.select_nodes` | GraphView → GraphView | `ids`: original node IDs; keeps internal selected edges |
| `graph.select_edges` | GraphView → GraphView | `ids`: exact original edge IDs; keeps their endpoints |
| `graph.nodes_table` | GraphView → BaseTable | None; includes structural node IDs |
| `graph.edges_table` | GraphView → BaseTable | None; includes IDs and endpoints |
| `table.select_columns` | BaseTable → BaseTable | `columns`: ordered column names |
| `table.select_rows` | BaseTable → BaseTable | `rows`: positions; caller order/duplicates retained |
| `graph.emit` | GraphView → named result | `name` |
| `table.emit` | BaseTable → named result | `name` |

Empty selections, row lists and column lists are real values. Node/edge IDs must
belong to the source selection; row positions must be in range. Booleans,
fractional/negative positions and wrong value domains reject explicitly, even
on empty input. Native integer/text list parameters and JSON arrays share the
same validation. JSON builder numeric lists must fit its existing signed integer
representation; floating lists are never coerced into IDs. Unsupported controls
(for example an implicit live-output flag) reject at instantiation.

`alias` now copies any existing StepValue by its actual type, including snapshots,
graph views, shared tables and temporal indexes. It does not recapture the owner.
Explicit target/result names retain existing replacement semantics. This does
not decide the workbook's optional generated-stage-name or cache policies.

## Combining reads and atomic property effects

```python
mixed_typed_handle = step_pipeline("properties and historical output", [
    {"id": "core.init_nodes", "params": {"target": "values", "value": 7}},
    {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "latest_score"}},
    *typed_program,
], atomic=True)
head = typed_graph.current_state_id
_, mixed_typed_profile = typed_graph.apply(mixed_typed_handle, return_profile=True)
assert typed_graph.current_state_id == head + 1
assert typed_graph.get_node_attr(0, "latest_score") == 7
assert mixed_typed_profile["outputs"]["graph"].node_attr(0, "x") == 11
```

These read stages are certified for the existing property-only atomic scope.
Local atomic builders and outer atomic pipelines discard newly produced outputs
and staged properties on execution, cancellation or publication failure, restoring
pre-existing Context outputs. Successful outputs merge by explicit name. Shared
native output-scope completion moves retained buffers instead of cloning them.
Non-atomic execution keeps its existing per-stage property behavior.

Cancellation is checked before and after each typed operation; this does not add
interruptible inner kernels or streaming. The existing Python pipeline coordinator
still owns its execution thread; this slice makes no new GIL-release guarantee.

Remaining item 30.1/AR7.8 work includes current staged graph captures, topology
mutation intermediates/read-after-write, remaining DSL/IR lowering, automatic
naming/cache policy and streaming. This fixed nine-stage read inventory is
implemented; it is not a claim that every graph/table operation is a pipeline
primitive or that the broad pipeline parent is complete.


## Selected pending working-state inputs

In addition to retained snapshots, a persistent local atomic builder can produce
`graph.capture_staged(output=...)`. The resulting graph/table values include
pending properties and removals. These captures have no accepted write pin and
materialize selected data explicitly. See [staged captures](staged-graph-captures.md)
for effect ordering, ownership, cost and the remaining unsupported scope.
