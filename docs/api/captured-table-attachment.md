# Attach captured table results in an atomic program

`table.attach` writes explicitly selected property columns from a row-bound table
created from `graph.capture_staged` in the same atomic run. It supports node and
edge tables, retained row selections/projections, degree tables and the existing
captured algorithm tables. The table and its immutable source stay unchanged.

```python
from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
initial = GraphBatch(effective_at=0)
initial.add_node()
graph.writer().apply_batch(initial)
head = graph.current_state_id
_, profile = graph.apply(step_pipeline("compute and attach", [
    stage("graph.add_nodes", count=1, output="new", effective_at=10),
    stage("graph.add_edges", edges=[[0, {"added": 0}]], output="edges", effective_at=20),
    stage("graph.capture_staged", output="pending"),
    stage("graph.algorithm_table", source="pending", output="metrics", algorithm="bfs", start=0),
    stage("table.attach", source="metrics", columns={"value": "distance"}, effective_at=30),
    stage("table.emit", source="metrics", name="metrics"),
], atomic=True), return_profile=True)
assert graph.current_state_id == head + 1
assert graph.get_node_attr(1, "distance") == 1
assert graph.read_view_at(29).node_attr(1, "distance") is None
assert graph.read_view_at(30).node_attr(1, "distance") == 1
metrics = profile["outputs"]["metrics"]
assert metrics.source_views()[0].write_state_id is None
assert metrics.source_views()[0].node_attr(1, "distance") is None
```

## Parameters

- `source`: explicit table variable.
- `columns`: a list of property names, or a mapping from source column name to
  destination property name. No implicit selection of all columns. An empty
  selection validates identity and stages no assignments.
- `effective_at`: optional signed Unix microseconds for these assignments. Omit
  it for existing canonical batch/default clock rules.

A local `atomic=True` builder and `persist=True` are required, including inside
an outer atomic pipeline. The source table variable remains local to its builder.
Use `table.emit` for owned read results without attachment. Certified
[nested compute bodies](nested-pending-execution.md) can also capture and attach,
with the same persistence and source-authority rules.

## Identity and validation

Authority comes from a logical capture issued by the current native mutation
batch, checked through weak capture records. Numeric IDs alone are insufficient.
Imported, unrelated or historical tables require the existing explicit mapped
writeback APIs; they do not gain authority merely because IDs match this graph.
The weak registry does not retain old graph payloads and is discarded with the run.

Canonical table source-writeback validation checks retained row IDs, duplicate
rows, column lengths, edited structural fields and reserved columns. Aliases
cannot target structural/system names. Property kinds and all pending targets
validate before any selected column is staged. Added IDs can receive properties;
removed/cascaded IDs reject. Joins/derived rows without unambiguous row binding
require explicit identity resolution through the existing writeback boundary.

An earlier capture can supply values for rows that still belong to the pending
selection. Values are those of the supplied immutable snapshot; the engine does
not recompute them automatically. Selecting surviving rows explicitly can retain
those prior computations after topology changes. A table containing removed rows
fails rather than being silently filtered.

Null cells assign explicit Null. Omitted property columns remain unchanged.
Source structural columns cannot be copied as properties through this operation,
including aliases. Use existing explicit ID/value primitives where appropriate.

## Publication and timestamps

All attachments and topology effects share one final canonical acceptance.
Invalid clocks, schema failures, source/row errors, cancellation and final
validation errors discard staged properties/topology and restore outputs.
Rejection consumes no IDs or history. Successful attachments retain their
per-property effective times through archive recovery and time reconstruction.

A batch accepts one final assignment per entity/property key. A later staged
assignment replaces that pending value and its timestamp choice; an unstamped
replacement restores default clock rules. Intermediate pending values do not
become separate accepted deltas. Distinct accepted mutations retain their usual
history and same-effective-time tie behavior.

Canonical lifetime rules also remain unchanged: a property assignment outside an
entity's effective active lifetime is retained in accepted history but does not
carry into a later activation in effective-time reconstruction. Accepted-state
reads and effective-time reads are distinct. See [effective-time reads](effective-time-reads.md).

The emitted table keeps provisional provenance after success. Ordinary
`update_source_graph` still requires accepted-state provenance; this operation
does not relax that API. Refresh an owner capture for subsequent checked writes.
Current selected-read materialization costs and cancellation boundaries remain
as documented in [pending properties](pending-topology-properties.md).
