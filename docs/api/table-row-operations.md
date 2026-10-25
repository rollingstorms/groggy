# Grouping and table row operations

Joins, grouping, deduplication and set operations use the same typed row keys.
Equality follows [property identity](table-joins.md): compact/compressed text and
small integers match their ordinary representations; numeric property types
remain distinct; signed zero and NaN payloads are canonical within a floating
type; null matches null. Composite keys are tuples, without text delimiters.

## Stable groups and first occurrences

Both `group_by(columns)` and `groupby(by)` return groups in first-key occurrence
order. Rows within each group retain input order. Their existing distinct Python
collection wrappers remain; this change shares the native grouping mechanism.
Missing or repeated key names reject. The first encountered key value supplies
the representative group key, including its storage representation.

`drop_duplicates(subset=None)` retains the first row for each key. Without a
subset it uses every column; an explicit empty subset treats all rows as equal.
Groups and deduplicated rows retain exact frozen source row identity.

```python
from groggy import BaseTable
source = BaseTable.from_dict({"kind": [2, 1, 2, None], "value": [10, 11, 12, 13]})
groups = source.groupby("kind")
assert len(groups) == 3
assert groups[0]["value"].to_list() == [10, 12]
assert source.drop_duplicates(["kind"])["value"].to_list() == [10, 11, 13]
summary = source.group_by_agg(["kind"], {"value": "sum"})
assert summary["kind"].to_list() == [2, 1, None]
assert summary["value"].to_list() == [22, 11, 13]
```

Grouped aggregates use the same stable groups. Output columns are the requested
group keys followed by aggregation columns sorted by name (the input specs are
a map). Group/aggregation name overlap rejects rather than overwriting keys.
Missing aggregation columns and unsupported operations reject even on empty
input. Numeric reductions retain their existing checked numeric kernels.
Ungrouped `aggregate` also sorts output columns by name.

Existing empty-key behavior remains explicit: `group_by([])` returns a single
unchanged group, including for an empty input. `groupby([])` returns one group
for a nonempty input and no groups for an empty input. `group_by_agg([], specs)`
delegates to the ungrouped aggregate. Nonempty keys on empty inputs produce no
groups. Zero-column tables keep their row counts through these operations.

## Sets and explicit combinations

| Method | Contract |
| --- | --- |
| `union(other)` | Distinct rows, first left occurrences followed by unseen right occurrences. |
| `intersect(other)` | Distinct rows present in both inputs, in first-left-occurrence order. |
| `stack(other)` | Every left row followed by every right row; no deduplication. |
| `concat(other)` | Combine columns by row position, padding the shorter input with nulls. |

Union, intersection and stack require identical **ordered column names**. They
do not require homogeneous cell types. No implicit column alignment or graph
identity merge occurs. Concat preserves left column order then right column
order; a right name also present in the left receives `_y`. Any collision with
an existing or generated name rejects before output construction.

```python
left = BaseTable.from_dict({"key": [2, 1, 2]})
right = BaseTable.from_dict({"key": [1, 3]})
assert left.union(right)["key"].to_list() == [2, 1, 3]
assert left.stack(right)["key"].to_list() == [2, 1, 2, 1, 3]
combined = left.concat(BaseTable.from_dict({"extra": [7, 8]}))
assert combined["extra"].to_list() == [7, 8, None]
```

## Source snapshots and writeback

Group and dedup selections preserve direct entity IDs, even if visible ID columns
were dropped. Updating a selected group uses the normal checked source API.
Every existing view and table remains unchanged after that write.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"kind": 2}, {"kind": 1}, {"kind": 2}])
table = owner.read_view().nodes_table().base_table()
group = table.groupby("kind")[0]
assert group.source_entity_ids == [0, 2]
group["kind"] = [9, 9]
group.update_source_graph(owner, ["kind"])
assert owner.get_node_attr(2, "kind") == 9
assert table.source_views()[0].node_attr(2, "kind") == 2
combined = table.union(table)
assert combined.source_entity_ids is None
assert len(combined.source_views()) == 1
```

Sets, stack, concat and aggregate results keep immutable source contexts but have
no automatic row writeback identity. Explicit mapping/materialization remains
available; these operations never mutate a graph implicitly. Python releases
the GIL for grouping, deduplication and combination calls. Native stack/concat now have thin Python
adapters with the same semantics.

All outputs are eager owned columns; retained sources keep their frozen payloads
alive. Key hashing includes cell content. Group selection, set indexes and row
plans allocate memory proportional to their inputs/results; no memory budget,
streaming or billion-row guarantee is established. Broader graph/table conversion adapters remain open under 27/AR7.10.

[Collection reductions](table-collection-reductions.md) now share native numeric
kernels, checked group metadata and lossless supplied-table flattening.

[Specialized graph-table merge](graph-table-merge.md) now defines the six existing
identity/conflict policies, checked reference maps and retained source provenance.

[Collection ownership](table-collection-ownership.md) retains source captures and
known schema across supported empty collection transformations.
