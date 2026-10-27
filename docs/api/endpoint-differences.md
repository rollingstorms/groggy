# Endpoint differences and raw restoration safety

Native `GraphSnapshot.diff_with` and `HistoryForest.diff_commits` return
`edge_endpoint_changes` separately from edge additions/removals and property
changes. The canonical `EdgeEndpointChange` contains `edge_id`, `old_endpoints`
and `new_endpoints`. Records are sorted by edge ID. IDs present in only one state
remain additions/removals; a retained edge with different endpoints is counted
once in `StateDiff.summary().edges_changed` and makes `is_empty()` false.

Endpoints are the ordered stored pair, even for undirected graphs. Reversing that
pair is a storage difference; traversal semantics still follow graph direction.
Parallel edge instances remain separate. Readonly diffs do not publish deltas or
alter either state. This adds one linear endpoint comparison pass to snapshot
diffing and allocates records only for actual endpoint differences.

Python `Graph.diff_commits` includes an `edge_endpoint_changes` list whose entries
have `edge_id`, `old` and `new` keys; endpoint pairs are tuples. The list is empty
for ordinary add/remove/property-only history, which retains its existing output.

```python
from groggy import Graph

g = Graph()
g.add_nodes(2)
first = g.current_state_id
edge = g.add_edge(0, 1)
diff = g.diff_commits(first, g.current_state_id)
assert diff["edges_added"] == [(edge, 0, 1)]
assert diff["edge_endpoint_changes"] == []
```

## Current mutation boundary

Coordinated stable-ID mutation is now implemented through the
[endpoint revision API](endpoint-revisions.md). Raw reconstruction remains guarded;
a caller cannot overwrite retained topology through `GraphPool.add_edge_with_id`.
Recorded installation restores the selected typed revision, and immutable views
retain their own endpoint pairs. Legacy membership-only groups explicitly reject
rewiring until their source ownership is resolved; frozen sources are supported.
General revert/rewind and three-way merge planning remain separate work.

## Detached logical reconstruction

Native `Graph.prepare_commit_diff(from, to)` / `HistoryForest.prepare_commit_diff`
return a `CommitDiffPlan`. `HistoryForest.prepare_recorded_read(state)` returns a
`RecordedReadPlan`. Preparation uses the shared recorded/effective lineage cache and
shares immutable commit records; execution reconstructs logical state and typed
properties independently of the owner. Plans are Send + Sync and survive later
mutation, history pruning, purge and owner destruction. Preparation rejects
unknown states, missing parents and cycles rather than inventing partial states.

`Graph.diff_commits` and ordinary native recorded reconstruction use these same
kernels. Python prepares the plan while holding the owner borrow, releases that
borrow, and releases the GIL during reconstruction/comparison. Python output
marshalling remains under the GIL. Cold preparation is O(ancestry), runs on the owner thread and retains shared
record handles. A matching cached head shares its lineage in O(1); empty-root
reads do not evict that head. The existing cache retains only its most recent
head, so alternating different heads can still need ancestry walks. This is not
a universal latency guarantee. Cloned plans share their lineage arrays. Comparing
a state with itself validates its lineage, then returns an empty diff without
reconstructing either graph.

These logical plans carry property values and group-reference IDs, not graph
mutation authority or a frozen group-source registry. Use immutable GraphViews
or explicit owned snapshots when source payload/graph restoration ownership is
required. Other historical Python entry points retain their existing GIL behavior;
this change specifically migrates recorded commit diffs.
