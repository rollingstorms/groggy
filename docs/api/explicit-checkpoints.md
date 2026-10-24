# Explicit checkpoints and published versions

A checked write publishes a mutation delta and advances the graph state ID.
An explicit checkpoint labels an already-published state and preserves that ID.
It neither creates an empty mutation delta nor changes its effective/recorded
clocks. Multiple checkpoint labels may refer to the same state, including the
empty state with ID zero.

```python
from groggy import Graph

graph = Graph()
writer = graph.writer(author="analysis")
nodes = writer.add_nodes(2, effective_at=100)
writer.set_node_attr(nodes[0], "score", 3, effective_at=150)
state = graph.current_state_id
clocks = graph.publication_metadata(state)
assert writer.checkpoint("ready for analysis") == state
assert graph.commit("named snapshot", "analysis") == state
assert graph.current_state_id == state
assert graph.publication_metadata(state) == clocks
assert [row["message"] for row in graph.checkpoints()] == [
    "ready for analysis", "named snapshot"
]
assert len(graph.commit_history()) == 2  # Two mutation versions.
```

`graph.checkpoint(message, author, expected_state=...)` requires a clean published
state and rejects an optimistic-head mismatch. `writer.checkpoint(message)` uses
the session's selected state and schema policy, so stale sessions reject without
adding metadata. `graph.checkpoints()` returns owned records in acceptance order:
`state_id`, `branch`, `message`, `author`, and engine `recorded_at` microseconds.
This checkpoint time describes the label; it does not introduce another
mutation clock or alter history reconstruction.

## Compatibility while ordinary mutations migrate

Python `Graph()` now publishes ordinary mutations automatically. Explicit
`Graph(manual=True)` and legacy native compatibility APIs accumulate unpublished
changes. Explicit
`checkpoint()` rejects that dirty state. For compatibility, `commit()` publishes
pending legacy changes once and records a checkpoint of the resulting state.
When no changes are pending, `commit()` checkpoints the current state and succeeds
instead of raising `NoChangesToCommit`. Checked writer mutations need no commit
for visibility/history.

The existing Rust infallible `add_node`/`add_nodes` signatures remain compatibility
APIs. New `try_add_node`/`try_add_nodes` publish through the checked coordinator
and return failures before allocation. The count constructor preserves one bulk
publication; count zero is a no-op. Writers remain the interface for properties,
explicit effective timestamps and schema policies. The Python ordinary defaults now use checked publication. Existing native
infallible compatibility and compound pooled/meta mutation audits remain open.

## Durable checkpoints

`DurableGraph.checkpoint()` and its Python `commit()` alias sync one journal
record before exposing checkpoint metadata in memory. Reopening or backing up
the journal retains the original labels and times without consuming a mutation
state ID. The existing journal envelope remains unchanged; the new checkpoint
record is an explicit extension that older readers reject if unsupported.

Checkpoint labels do not add permanent retention pins. They are also not
compressed recovery images or automatic storage checkpoints. Materialized
recovery checkpoints, schedules, compaction and pruning remain separate storage
lifecycle work. Active graph JSON export does not contain the retained history
or checkpoint annotations; use the journal for that recovery scope.
