# Table joins

`BaseTable.join(other, on=..., how=...)` uses the Rust table engine. Python
releases the GIL while that engine plans and constructs the result. It supports
`inner`, `left`, `right`, `outer` and `cross`. Single keys, ordered lists of keys,
and explicit left/right key pairs share the same implementation.

```python
from groggy import BaseTable
left = BaseTable.from_dict({"kind": [1, 1, 2], "tag": ["a", "b", "b"], "value": [10, 11, 12]})
right = BaseTable.from_dict({"category": [1, 2], "label": ["b", "a"], "extra": [20, 21]})
result = left.join(right, on={"left": ["kind", "tag"], "right": ["category", "label"]})
assert result["value"] == 11
assert result["extra"] == 20
```

## Keys and ordering

Keys use Groggy's property identity, without converting values to debug strings.
Text and its compact/compressed storage variants match; Int and SmallInt match.
f32 and f64 remain distinct property types, as do integers, floats, booleans,
JSON, text and bytes. Signed zeros match within a floating type; NaN encodings
match within that type. Vector and temporal keys use their property identity.
**Null matches null**, retaining Groggy's existing behavior. Sparse projection
holes represented as Null follow that rule. This is not SQL null comparison.

Duplicate keys produce every matching pair. Inner/left joins preserve left row
order and right match order. Right joins preserve right row order and left match
order. Outer joins emit left matches/unmatched rows first, then unmatched right
rows in right input order. Cross joins use left-major order.

```python
left = BaseTable.from_dict({"key": [2, 1], "a": [10, 11]})
right = BaseTable.from_dict({"key": [2, 3], "b": [20, 21]})
result = left.join(right, on="key", how="outer")
assert result["key"].to_list() == [2, 1, 3]
assert result["a"].to_list() == [10, 11, None]
assert result["b"].to_list() == [20, None, 21]
```

## Output columns and validation

Existing single-key naming policies are preserved:

| Family | Column policy |
| --- | --- |
| Inner/left | Keep left key columns; omit right key columns. Conflicting non-key columns receive `left_` / `right_` prefixes. |
| Right | Uses the swapped-left schema, with original right input first. |
| Outer | Keep left columns; omit a right key paired with the identically named left key. Conflicting retained right columns receive `_right`. Shared key columns use the right key for right-only rows. |
| Cross | Keep left columns; conflicting right columns receive `_right`. |

Composite keys apply these policies to each paired key. Output column order
follows the two input column orders. Generated names must be unique: a conflict
with an existing `left_name`, `right_name` or `name_right` rejects and asks for
explicit input renaming. No column is silently overwritten. Empty, repeated,
missing, or unequal key lists reject. Cross joins do not inspect keys.

## Source identity and costs

Frozen graph tables retain all distinct [source captures](table-source-pins.md)
through every join, including empty results and composite joins. Joins never
infer row writeback ownership. Explicit identity resolution is required before
writing joined data back; creating an independent graph remains explicit.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"kind": 1, "tag": "a"}, {"kind": 1, "tag": "b"}])
source = owner.read_view().nodes_table().base_table()
joined = source.join(source, on=["kind", "tag"])
assert len(joined.source_views()) == 1
assert joined.source_entity_ids is None
owner.set_node_attr(0, "kind", 99)
assert joined.source_views()[0].node_attr(0, "kind") == 1
```

Native `BaseTable::join_on` accepts ordered key slices and the same five family
names. Existing single-key methods delegate to it. Native `TableArray::join`
delegates each corresponding pair to those methods; it supports all five families
and keeps its existing single-key, positional table-pairing contract.

These joins eagerly build a right-side hash index, a row-pair plan and owned
result columns. Expected keyed work is proportional to input key content plus
output size; duplicate fanout and Cartesian products can be large. Cross row
multiplication and row-plan reservations are checked, but this is not a memory
budget or streaming join. Source captures remain alive with the result. There is
no row-identity guarantee for joined output and no inferred graph mutation.

Broader graph/table conversion adapters remain open under 27/AR7.10.

[Grouping and row operations](table-row-operations.md) now share these typed keys
and define stable group/set ordering and explicit combination contracts.

[Collection reductions](table-collection-reductions.md) now share native numeric
kernels, checked group metadata and lossless supplied-table flattening.

[Specialized graph-table merge](graph-table-merge.md) now defines the six existing
identity/conflict policies, checked reference maps and retained source provenance.

[Collection ownership](table-collection-ownership.md) retains source captures and
known schema across supported empty collection transformations.
