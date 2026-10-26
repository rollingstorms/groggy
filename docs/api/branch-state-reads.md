# Branch lifecycle and selected state reads

A branch names a retained published state. Switching branches restores active
membership, topology and typed properties while preserving lifetime ID/name
reservations, backing values and frozen source revisions. It creates no graph
mutation version. Existing immutable views and read plans keep their captured data.

Recorded-timestamp and effective-time selection now both use the selected branch's
first-parent ancestry. Explicit state-ID inspection can still read any retained
state, including another branch.

```python
from groggy import Graph

g = Graph()
n = g.add_node(value=None)
first = g.current_state_id
g.create_branch("other")
g.writer().set_node_attr(n, "value", 2)
main = g.current_state_id
g.checkout_branch("other")
assert g.snapshot_at_timestamp(2**64 - 1).commit_id == first
g.writer().unset_node_attr(n, "value")
side = g.current_state_id
g.checkout_branch("main")
assert g.snapshot_at_timestamp(2**64 - 1).commit_id == main
assert g.snapshot_at_commit(side).node_attr(n, "value") is None
assert g.get_node_attr(n, "value") == 2
```

## Timestamp units and scope

`snapshot_at_timestamp(seconds, owned=False)` selects the closest retained commit
at or before unsigned Unix seconds on the current first-parent lineage. Equal
second-resolution timestamps select the higher accepted state ID. A branch with
no recorded versions, or a bound earlier than its first version, rejects instead
of selecting another branch. A checkpoint label does not manufacture a new version.
An owned copy selected at an older state also excludes later retained descendants
from timestamp selection on its retargeted branch.

`read_view_at(effective_microseconds, recorded_by=...)` reconstructs changes at
signed-microsecond effective time with an optional explicit acceptance cutoff.
It can return an empty graph before the first admitted event. It is a different
operation from selecting a recorded version by second-resolution time.

Native callers with a bare HistoryForest can use
`commit_at_or_before_on_lineage(head, seconds)` or
`TemporalSnapshot::at_timestamp_on_lineage`. The older bare-history global
`commit_at_or_before` / `TemporalSnapshot::at_timestamp` intentionally inspect the
whole retained history because no graph branch is supplied. Graph facade methods
use the explicit selected-head selector and share its cached ancestry.

## Manual edits and metadata

Manual `create_branch` points at the last published state; pending edits are not
accepted by creating a reference. Dirty branch switches reject. Explicit
`Graph.reset_hard()` discards pending edits and restores the current published
state, retaining allocated IDs and backing data. A later allocation does not reuse
IDs from discarded manual work.

```python
manual = Graph(manual=True)
node = manual.add_node(value=1)
manual.commit("initial", "example")
manual.set_node_attr(node, "pending", 3)
manual.create_branch("published-only")
manual.reset_hard()
manual.checkout_branch("published-only")
assert not manual.has_node_attribute(node, "pending")
assert manual.get_node_attr(node, "value") == 1
```

## Durable branch operations

DurableGraph branch creation/switching require the expected current state. Duplicate,
unknown, stale and installed-schema-invalid operations reject before journal I/O.
Successful references and selected branches recover through reopen/full backup;
checks include ID high-water marks and retained group sources whose live entities
were deactivated on another branch. DurableGraph also exposes
`snapshot_at_timestamp(seconds, owned=False)` with the same selected-lineage rules.

Injected journal failures before header, after header, after payload and after
sync roll back bytes before branch/reference exposure. Retrying succeeds and
reopen reproduces that accepted selection. Indeterminate rollback failure retains
the existing reopen-required policy.

This completes verification of the existing branch lifecycle inventory (14.1).
Merge planning remains 14.2; explicit destructive branch pruning, pool/identity
reclamation and storage compaction retain 15/10.4 scope.
