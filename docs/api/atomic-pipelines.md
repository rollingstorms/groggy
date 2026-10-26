# One property publication for a whole pipeline

`Pipeline.run(..., atomic=True)`, calling the pipeline with `atomic=True`,
`groggy.apply(..., atomic=True)`, `Graph.apply(..., atomic=True)` and
`Subgraph.apply(..., atomic=True)` stage a whole certified property pipeline.
Later algorithms read earlier algorithms' staged columns. Success accepts one
mixed node/edge batch; failure or cancellation discards all staged properties.
Ordinary `atomic=False` execution keeps its existing per-algorithm behavior.

```python
from groggy import Graph, Pipeline
from groggy.builder import AlgorithmBuilder

g = Graph()
g.add_node(score=1)
head = g.current_state_id
old = g.read_view()
first = AlgorithmBuilder("first")
first.attach_as("score", first.init_nodes(default=1.0000000001))
second = AlgorithmBuilder("second")
second.attach_as("copied", second.load_attr("score"))
pipe = Pipeline([first.build(), second.build(atomic=True)])
_, profile = pipe(g.view(), atomic=True, return_profile=True)
assert g.current_state_id == head + 1
assert g.get_node_attr(0, "copied") == 1.0000000001
assert old.node_attr(0, "score") == 1
assert profile["atomic"]
```

## Certification and shared staging

The core Algorithm trait defaults to uncertified. A builder algorithm certifies
itself only when every descendant step supports the existing property overlay.
Every outer algorithm is checked before any runs. Uncertified classic wrappers,
streaming/topology effects and unsupported builder steps reject explicitly.
Supported builder children can have either local atomic flag: an outer run owns
the batch, and children borrow it without accepting intermediate versions. Execution
uses the supplied compiled step order. Build mutation-sensitive Python children
with `atomic=True` (or disable optimization), which preserves their original step
order; an outer run cannot recover steps reordered before compilation. Effect-aware
IR migration remains AR7.8.

The outer run requires a clean graph head for persistent updates, preserves the
selected node/edge scope and rejects a change of graph owner. Validation at final
acceptance, runtime errors and cancellation—including after the last algorithm—
discard staging. A failed run clears its pending batch, preserves pre-existing
Context outputs and discards partial outputs; telemetry remains available. Success
merges produced outputs into the Context. Native Contexts and compiled pipelines
can be reused after an error. Native API: `Pipeline::run_atomic_properties`.

```python
failure = AlgorithmBuilder("invalid managed property")
failure.attach_as("entity_type", failure.init_nodes(default=4))
head = g.current_state_id
bad = Pipeline([first.build(), failure.build()])
try:
    bad(g.view(), atomic=True)
except RuntimeError:
    pass
else:
    raise AssertionError("Managed identity writes must reject")
assert g.current_state_id == head
assert g.get_node_attr(0, "score") == 1.0000000001
assert not g.has_uncommitted_changes()
```

`persist=False` returns owned transient outputs with no publication, using the
existing original-graph read semantics. It does not turn transient attachments
into a working property overlay. Read-only/empty persistent runs create no version.
A successful persistent pipeline receives one engine acceptance/effective-time
default through GraphBatch; this entry point does not add per-algorithm timestamps.

## Materialized manual bases

A native `Graph::from_snapshot` and lightweight temporal `as_subgraph` now stage
the complete selected topology/properties as a pending manual import. Checked
updates reject until that base is accepted, so a first property update cannot
silently become a history root with missing topology or untouched values.
`commit` accepts a manual import; native `into_publishing` imports the entire
selected current graph once, with new acceptance clocks, then enables ordinary
publication. Existing property indices are reused, and malformed endpoint/value
mappings reject. An explicit owned temporal capture's `to_graph()` already has
retained accepted history and needs no such import.

This completes the certified property-only outer pipeline and selected import
boundaries under 09.4b.2. Native infallible/default constructor migration, mutable
raw pool/space escape auditing and uncertified/topology algorithms retain that
item's remaining scope. It does not introduce arbitrary topology transactions or
a durable pipeline executor. Owner-thread graph assembly still holds the GIL.
