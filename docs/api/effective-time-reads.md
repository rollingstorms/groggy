# Immutable graph reads at effective time

`graph.read_view_at(T, recorded_by=U)` reconstructs the graph at signed Unix
microsecond effective time `T`. Omitting `recorded_by` uses every accepted
publication in the selected branch's pinned first-parent ancestry. Supplying `U`
excludes publications whose engine-recorded acceptance clock is later than `U`.
Both `Graph` and `DurableGraph` support this, including after durable reopen.

```python
from groggy import Graph, GraphBatch

graph = Graph(directed=True)
batch = GraphBatch(effective_at=10)
a = batch.add_node(score=1)
b = batch.add_node()
batch.add_edge(a, b)
first = graph.apply_batch(batch, 0, "initial", "example")["state_id"]

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

## Ordering, lifecycle and ownership

Changes replay by effective time, then accepted version order. For the same key
and effective timestamp, the later accepted write wins. One instant's changes are
considered together for endpoint validity. Creation starts a fresh lifetime;
properties written outside an active lifetime do not carry into a later
activation. Explicit null remains a value; a never-assigned property is absent.
Property deletion/unset is separately tracked in checklist 11.2.

The read pins the selected published head and excludes sibling branches and later
publications. For an already resolved merge, reconstruction follows the first
parent and applies its resolution delta, matching ordinary state reconstruction.
Conflict detection and caller merge policies remain 14.2; a temporal read does not
choose a resolution or replay secondary-parent mutations a second time.

Captures survive later writes, branch changes, owner destruction and durable
close. Selections and table source views retain effective clocks and the selected
head. `view.read_time` reports branch, published_head, effective_at and recorded_by;
ordinary current views report None. Read-local legacy clock/source bindings also
appear when supplied. A temporal view has no implicit current-head write authority.
Reads do not label a checkpoint, publish a delta, rewind or change the working graph.

## Frozen group references

`retain_group(source, kind="group")` retains an explicitly chosen immutable source
selection. Membership, typed properties, topology, direction and nested group
references belong to that source revision. Changed source state or values produce
a new reference. Old references remain readable. A group stores selected rows and
direct dependency handles, without retaining the full parent graph or a mutable
owner. Retaining a group does not itself publish a mutation state.

`view.meta_subgraph(node)` returns the referenced frozen GraphView. Its source
state/time remains its explicit source pin, independent of the parent query time.
Root meta properties and subnode properties remain separate changes. Creating or
updating `contains_subgraph` publishes the chosen reference at its effective time.

```python
from groggy import AttrValue

source = graph.read_view_at(15).select_nodes([0, 1])
group = graph.retain_group(source, kind="source")
meta = graph.create_meta_node(group, effective_at=30)
old = graph.read_view_at(35)
assert old.meta_subgraph(meta).node_attr(0, "score") == 1

graph.writer().set_node_attr(0, "score", 9, effective_at=40)
new_group = graph.retain_group(graph.read_view().select_nodes([0, 1]), kind="source")
assert new_group != group
graph.writer().set_node_attr(meta, "contains_subgraph",
                             AttrValue.subgraph_ref(new_group), effective_at=50)
assert graph.read_view_at(49).meta_subgraph(meta).node_attr(0, "score") == 1
assert graph.read_view_at(50).meta_subgraph(meta).node_attr(0, "score") == 9
assert old.meta_subgraph(meta).node_attr(0, "score") == 1
```

Durable retention persists typed group documents in dependency order before a
batch references them. Reopen and backup retain their exact source pins, including
nested groups and explicitly resolved legacy aliases. Unknown group references
reject before mutation/journal append. Duplicate retention is idempotent and does
not append another record. The source may be another graph: entity IDs inside the
group belong to its frozen source selection and are not inserted into the working
graph's entity space.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as folder:
    path = Path(folder) / "temporal-groups.ggj"
    with DurableGraph.create(path) as durable:
        group = durable.retain_group(old.select_nodes([meta]), kind="nested")
        outer = durable.create_meta_node(group, effective_at=100)
    with DurableGraph.open(path) as durable:
        inner = durable.read_view_at(100).meta_subgraph(outer).meta_subgraph(meta)
        assert inner.node_attr(0, "score") == 1
```

This canonical frozen read API is separate from connected legacy NodeView/Subgraph
adapters, whose broader migration remains 26.1. Active JSON export still rejects
pooled references; durable retention carries their payloads explicitly.

## Explicit inputs for legacy information

Older manual versions have no precise per-change clocks. Supply
`legacy_times={state_id: (effective_at, recorded_at)}` to assign clocks for that
read. Missing clocks reject; existing precise publication clocks cannot be
overridden. The supplied clocks are never written back or guessed from the old
second-resolution timestamp. Native `EffectiveReadOptions.legacy_publications`
also accepts individual ChangeKey overrides within a legacy version.

```python
legacy = Graph(manual=True)
legacy.add_node(v=1)
one = legacy.commit("old", "example")
legacy.set_node_attr(0, "v", None)
two = legacy.commit("old2", "example")
clocks = {one: (10, 100), two: (20, 200)}
assert legacy.read_view_at(15, legacy_times=clocks).node_attr(0, "v") == 1
assert legacy.read_view_at(25, recorded_by=150, legacy_times=clocks).node_attr(0, "v") == 1
assert legacy.publication_metadata(one) is None
```

Membership-only legacy groups similarly lack a frozen source. Supply
`group_sources={legacy_group_id: explicit_source_view}`. The source must have
exactly the stored node/edge membership. Frozen references reject overrides.
ReadTime records the source revision binding and the resulting view retains it.
Native `pin_legacy_group(id, source)` creates a new stored revision when migrating
a working reference; it does not replace the old immutable definition. Frozen
views with explicit legacy bindings can themselves be retained durably.

Pending manual changes must be published before a temporal read. Neither explicit
legacy inputs nor reconstruction convert active exports into complete raw pool
recovery; storage migration and full configuration recovery retain their scopes.

## Endpoint validity and cost

Checked publication validates affected edges' complete effective lifetimes before
allocation, working mutation or durable append. Backdated removals inspect
retained edges already absent from current state. Invalid future intervals reject.
Journal replay uses original clocks. Untimed legacy writes retain compatibility;
explicit timed topology requires known canonical lifetimes. The reader independently
checks every relevant interval through T for raw history bypassing checked writes.

The selected head caches its immutable ancestry and a lazy sorted event index.
Index construction runs outside the Python GIL. Repeated reads binary-select the
effective prefix and reconstruct only admitted changes; optional recorded cutoffs
filter acceptance separately. New heads get a new index, branch changes select the
correct ancestry, and history GC clears the cache. Old plans/views remain owned.
Read-local legacy clocks build their own index rather than altering the shared one.

Cold preparation visits history depth; the first build visits/sorts all logical
changes A: O(A log A). Warm ordinary preparation is O(1), prefix selection is
O(log A), and reconstruction visits admitted prefix C and builds the result:
O(C + N + E). Memory includes the O(A) index and O(C + N + E + property payload)
reconstruction. These are event indexes, not materialized recovery/read checkpoints.
Legacy source normalization and retention additionally copy selected payloads.

`cargo bench --bench effective_time_reads` measures cold and warm behavior. The
100k-node fixture measured approximately 155 ms cold index plus reconstruction,
105 ms warm reconstruction and 0.4 µs for a warm empty prefix. These are local
bounded costs, not constant-time complete graph reconstruction, RSS, FFI or
production-capacity guarantees. Read checkpoint optimization remains performance
work; storage checkpoint scheduling stays in 10.4/A02.
