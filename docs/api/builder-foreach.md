# Nested sampler for-each execution

`sample.for_each` maps a body over an ordered selector array. Each item gets fresh
variables and the input selector as its scope. Body-local variables do not capture
outer variables. Nested bodies share the enclosing property staging context.
The step is certified for property-atomic algorithms/loops exactly when **every**
descendant step is certified, including when there are no input items.

## Child selectors: flatten, empty, fallback

A child may set `__sample_output__` to a selector array. Every element is retained,
in child order, then input-item order. Duplicates remain distinct outputs. An
explicit empty array emits zero results. If the variable is absent, retain the
input selector once. A present variable of another type is an error.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph(directed=True)
graph.add_nodes([{"score": 0} for _ in range(4)])
for node in range(3):
    graph.add_edge(node, node + 1)
builder = AlgorithmBuilder("flat_each")
mapped = builder.each(builder.iterate_edges(), lambda child: child.emit_subgraphs(child.iterate_nodes(), induced=False))
builder.emit_subgraphs(mapped, induced=False)
samples = graph.view().sample(builder.build_sampler())
assert [sorted(sample.node_ids) for sample in samples] == [[0], [1], [1], [2], [2], [3]]
```

All input selectors validate before any body runs. Inputs must share the owner
and stay within the invocation selection with valid endpoint closure. Each child
output must stay within that child's selector, not merely the larger outer
selection. Revalidate child liveness/topology after the body before installation.
These remain legacy owner-connected selectors, not pinned value snapshots.

```python
builder = AlgorithmBuilder("empty_each")
mapped = builder.each(builder.iterate_edges(), lambda child: child.emit_subgraphs(child.sample_nodes(count=0), mode="per_item", induced=False))
builder.emit_subgraphs(mapped, induced=False)
assert len(graph.view().sample(builder.build_sampler())) == 0
```

Note that unified emission from an empty map deliberately creates one empty
selector; use an explicit empty array (for example per-item emission from an empty
mask) when zero outputs are intended.

## Property-atomic execution

Use `build(atomic=True)` on the algorithm executor for a checked outer property
batch. Earlier item assignments are visible through later property loads, including
when inputs overlap. Every item/descendant shares that one batch; acceptance occurs
once after the entire algorithm succeeds. Failures/cancellation discard the batch.
No per-item commit/checkpoint is created.

```python
builder = AlgorithmBuilder("increment_each")
def increment(child):
    values = child.load_attr("score")
    child.attach_as("score", child.map_nodes("x + 1", inputs={"x": values}))
builder.each(builder.iterate_edges(), increment)
before = graph.current_state_id
graph.view().apply(builder.build(atomic=True), persist=True)
assert [graph.get_node_attr(node, "score") for node in range(4)] == [1, 2, 2, 1]
assert graph.current_state_id == before + 1
```

Standalone sampler pipelines continue to reject atomic=True. Ordinary non-atomic
persistent execution retains its incremental write behavior: a later failure can
leave earlier writes accepted. Certification of this step does not make an outer
list of algorithms, arbitrary native storage escapes or standalone sampler runs
transactional.

## Owned transient outputs

Each child step contributes to a private output frame. Node/edge attribute columns
merge by entity ID across body steps and items, then with prior columns at the
same key. Later accepted assignments win; final rows are sorted by ID. Node/edge
identity domains remain separate. Untouched earlier rows and unrelated outputs
remain retained. Columns accumulate once per frame rather than repeatedly
concatenating/sorting the entire growing result for every item.

Transient property reads still see owner values. These output frames do not
simulate a graph or feed prior transient assignments into subsequent loads.

```python
before = graph.current_state_id
_, profile = graph.view().apply(builder.build(atomic=True), persist=False, return_profile=True)
assert profile["outputs"]["node_attributes.score"] == [(0, 2), (1, 3), (2, 3), (3, 2)]
assert [graph.get_node_attr(node, "score") for node in range(4)] == [1, 2, 2, 1]
assert graph.current_state_id == before
```

Non-column payloads, including histograms, use
`for_each.<target>.<item_index>.<original_key>`. For example, an item producing
`histograms.hist` under target mapped has key
`for_each.mapped.0.histograms.hist`. Nested frames add their enclosing paths.
Repeated writes to the same target/item/output path replace its previous payload;
this is result ownership, not history of every loop iteration. Entity columns
retain their original keys, such as node_attributes.score.

On body failure, cancellation, invalid selectors or output-type errors, discard
frame-local owned payloads and retain prior context outputs and the prior target
array. Do not catch a failed child step and continue the same staged transaction;
the enclosing atomic algorithm owns discarding its property batch.

## Checks, costs and remaining work

Recursive child-effect checks include nested loops/for-each and run before atomic
execution even on empty inputs. Unknown/malformed bodies reject during native
construction; unsupported temporal descendants remain explicit errors.
Cancellation checks cover inputs, items, child boundaries and final preparation;
individual sorts/merges/large kernels do not promise bounded interruption latency.
Python validation recognizes attachments nested in bodies, avoiding a false
missing-property-output warning.

Working storage includes prepared selected topology, per-item fresh variables,
flattened selector memberships, deduplicated touched output rows and per-item
non-column payloads. Overlapping selections still duplicate selector memberships.
Frame isolation moves prior outputs aside; it does not clone the owner/history.
Retaining all child outputs can use more memory than the old truncating executor.
Bounded native release receipts are not matched speedups, RSS/FFI/allocation or
production-scale guarantees. Pinned view/source provenance, optimizer effects,
standalone sampler transaction migration and remaining step families stay open.
