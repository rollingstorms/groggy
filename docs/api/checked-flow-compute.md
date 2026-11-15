# Checked flow-map computation

`core.flow_update` and `core.residual_capacity` operate on explicit typed edge
maps. They now compose after pending topology, inside certified compute loops and
across outer atomic builders. Loading or initializing a map uses the pending
selection; an already-supplied map keeps its original IDs and values.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes(2)
graph.add_edge(0, 1)
head = graph.current_state_id
graph.apply(step_pipeline("flow and residual", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("graph.add_edges", edges=[[1, {"added": 0}]], output="new_edge"),
    stage("core.init_edges", value=3, target="flow"),
    stage("core.init_scalar", value=2, target="delta"),
    stage("core.flow_update", flow="flow", delta="delta", target="updated"),
    stage("core.init_edges", value=9, target="capacity"),
    stage("core.residual_capacity", capacity="capacity", flow="updated", target="residual"),
    stage("core.attach_edge_attr", source="updated", attr="flow"),
    stage("core.attach_edge_attr", source="residual", attr="residual"),
], atomic=True))
assert graph.current_state_id == head + 1
assert [graph.get_edge_attr(e, "flow") for e in [0, 1]] == [5, 5]
assert [graph.get_edge_attr(e, "residual") for e in [0, 1]] == [4, 4]
```

## Numerical and sparse-map rules

- Integer addition/subtraction stays exact within signed 64-bit range. Overflow
  returns an error; it cannot panic in debug builds or wrap in release builds.
- Mixed integer/float arithmetic uses f64, retaining ordinary floating-point
  precision limits. Nonfinite operands and nonfinite computed results reject.
  Boolean, text and null values are not flow numbers.
- Map deltas update the union of flow/delta keys; a missing flow is zero and a
  missing delta leaves the original value. Scalar deltas apply only to supplied
  flow keys. Passthrough rows are validated too. A scalar is validated even when
  the flow map is empty.
- Residuals include only positive `capacity - flow`, with absent flow treated as
  zero. Positive flow-only keys retain their supplied IDs as backward residual
  amounts. Zero and negative residuals are absent; positive subnormal floats are
  retained. This primitive does not invent or traverse reverse edges.

These are arithmetic building blocks, not a flow-conservation or feasibility
certificate. Caller algorithms determine those constraints. Existing valid
numeric results and sparse-key behavior are preserved; inputs that previously
wrapped, panicked or passed invalid values now fail explicitly.

## Ownership and failure

The operations reuse checked native expression arithmetic and install their
result map only after success. They do not read owner topology or expand an input
map to newly added edges. Final attachment checks graph selection and entity
identity. An old map containing removed edge IDs therefore cannot silently write
onto a different pending selection.

Per-row cancellation and final installation checks are retained within the shared
atomic execution boundary. Errors discard staged topology/properties and restore
prior outputs. No new FFI computation or storage format is introduced.
