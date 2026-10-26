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

This is the typed comparison foundation, not a completed endpoint mutation API.
`GraphPool.add_edge_with_id` is a reconstruction helper: it accepts the same
retained endpoints or a previously unmaterialized ID and rejects changing an
existing edge's endpoints. Graph state restoration preflights existing endpoint
identity before replacing working membership, so a conflicting snapshot cannot
partially change the working state. Dense and explicitly purged sparse pools
share this rule.

Coordinated rewiring/reversal, before/after recorded deltas, effective-time indexes,
clock validation, retained endpoint revisions, archive/journal replay, batch/bulk
interfaces and tracked/destructive revert remain under existing 04.2/13.3. Raw
storage overwrite is not a substitute for that history path. No storage format
version changes in this comparison/safety slice.
