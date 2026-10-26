# Immutable graph reads at effective time

`graph.read_view_at(T, recorded_by=U)` reconstructs the graph at signed Unix
microsecond effective time `T`. `recorded_by` is optional: omitting it uses every
accepted publication in the selected branch's pinned ancestry. Supplying `U`
excludes publications whose engine-recorded acceptance clock is later than `U`.
This works on `Graph` and `DurableGraph`, including after durable reopen.

```python
from groggy import Graph, GraphBatch

graph = Graph(directed=True)
batch = GraphBatch(effective_at=10)
a = batch.add_node(score=1)
b = batch.add_node()
batch.add_edge(a, b)
first = graph.apply_batch(batch, 0, "initial", "example")["state_id"]
recorded = graph.publication_metadata(first)["recorded_at"]

batch = GraphBatch(effective_at=20)
batch.set_node_attr(0, "score", 2)
graph.apply_batch(batch, first, "correction", "example")
view = graph.read_view_at(15)
assert view.node_attr(0, "score") == 1
assert graph.read_view_at(25).node_attr(0, "score") == 2
assert view.bfs_distances(0) == [(0, 0), (1, 1)]
assert graph.read_view_at(25).filter_nodes("score == 2").node_ids == [0]
assert view.read_time["effective_at"] == 15
assert view.select_nodes([0]).read_time == view.read_time
assert view.write_state_id is None
```

## Ordering, ownership and lifecycle

Changes replay by effective time, then accepted version order. For the same key
and effective timestamp, the later accepted write wins. One batch's changes at a
single effective instant are considered together for endpoint validity. Creation
starts a fresh entity lifetime; properties written outside an active lifetime do
not carry into a later activation. Explicit null remains a value; a never assigned
property remains absent. Property deletion is not yet represented by this API.

The read pins the selected published head and excludes sibling branches and later
publications. Captures remain readable after later writes, branch changes, graph
destruction or durable close. Derived selections and table source captures retain
the effective clocks and head. `view.read_time` reports branch, published_head,
effective_at and recorded_by; ordinary current-state views report None. The head
identifies ancestry, not an assertion that the reconstructed values equal that
version's working graph. Temporal views have no implicit writeback authority.

Reading does not create a checkpoint, publish a delta, alter current properties
or rewind the working graph. A checkpoint remains an explicit label of accepted
state. The Python adapter captures immutable native history inputs under the GIL,
then releases it for replay. Current graph owners remain on their owning thread.

## Current limits

Precise publication metadata is required throughout the selected ancestry.
Legacy manually grouped versions, pending manual edits, merge ancestry and pooled
subgraph-reference values reject explicitly. Typed ordinary properties, topology,
loops, parallel edges and graph direction are reconstructed. Arbitrary pooled
meta/group history remains in checklist 12.2/16.2; property unset remains 11.2.

Checked topology publication now validates each affected edge's complete effective
lifetime against both endpoints **before** ID allocation, working-state mutation
or durable journal append. Backdated node removals inspect retained edges even if
they are no longer active in the working graph. The same-time acceptance and
lifecycle ordering is shared with the reader. Invalid future intervals reject too;
validation is not limited to the current wall clock.

The history-owned index stores lifecycle events and incident edge IDs, without
property payloads. Accepted changes update it incrementally. Property-only writes
skip topology validation. Switching to another ancestry rebuilds the selected
branch's index; explicit purge releases the cache. Validation sorts the affected
entities' lifecycle events, so cost depends on affected degree/lifetime history,
not unrelated property history. Current bounded measurements are in
`notes/benchmarks/effective-topology-publication-2026-10-25.json`.

Legacy manually grouped ancestry has unknown effective lifetimes: explicit timed
topology writes reject. Untimed legacy compatibility writes remain available,
without a historical temporal guarantee; precise effective-time reads still reject
that ancestry. Merge ancestry also remains unsupported. Journal recovery validates
using the original recorded clocks. Older journals containing invalid timed
topology now reject on open rather than recreating an invalid timeline.

The reader retains its independent boundary check for raw/imported history that
bypasses checked publication. An invalid earlier interval cannot disappear behind
a valid final state. Future invalid intervals after T do not change an earlier
read of such raw history.

This first implementation walks the pinned lineage, sorts relevant logical
changes, and builds a frozen graph plus adjacency. Preparation is O(history
depth). Replay visits all A logical changes in retained ancestry, sorts the C
changes admitted by the clocks, and constructs the frozen graph: O(A + C log C +
N + E) time. Memory is O(C + L + N + E + property payload), where L is the largest
publication validated for unique logical keys. It has no effective-time index or replay checkpoints yet.
Table source pins preserve clocks; table cells retain the existing rectangular
null-padding convention. Use the view's property-presence query to distinguish
explicit null from absence.

Benchmark: `cargo bench --bench effective_time_reads`. The fixture checks current
reconstruction costs through 100k nodes; it does not claim indexed performance,
production capacity or a speedup over a prior implementation.
