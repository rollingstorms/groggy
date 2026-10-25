# Collection ownership and empty results

Core table collections retain a shared context containing an empty schema,
known numeric/boolean column roles, declared group names and immutable source
captures. Removing all tables does not turn the result into an unrelated object.
The context contains no invented data rows and does not change graph history.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"kind": 2, "score": 10, "text": "a"}, {"kind": 1, "score": 11, "text": "b"}])
source = owner.read_view().nodes_table().base_table()
empty = source.groupby("kind").take(0)
assert len(empty) == 0
assert empty.sum()["kind"].to_list() == []
assert empty.sum()["score"].to_list() == []
assert len(empty.concat().source_views()) == 1
owner.set_node_attr(0, "score", 99)
assert empty.concat().source_views()[0].node_attr(0, "score") == 10
```

## Supported transformations

| Operation | Ownership and schema behavior |
| --- | --- |
| take / skip / filter / native select_tables | Retain the input schema, group names and source captures, even when zero tables remain. Selected nonempty tables retain their direct row identity. |
| head / tail / sample | Retain collection context while transforming real tables. Zero rows within real groups retain group count and keys. |
| select / drop_columns | Transform physical schema and roles even on empty collections; declared group keys may stay as metadata after physical columns are projected away. |
| rename | Rename physical columns and group metadata together; reject physical/key name collisions. |
| sort_by | Check the known schema even when there are no tables, then sort only real tables. |
| map / apply | Run once per real table, never on a synthetic prototype. Empty inputs preserve their known input schema; unknown callback output schemas are not guessed. Source ancestry is retained even if a callback returns imported tables. |
| native pairwise join | Retain both input contexts, use the actual BaseTable column plan for output schema/roles/key aliases, and reject ambiguous key-name collisions. |

Native filter predicates run exactly once per real table. The Python core
collection filter preserves callback and truth-conversion exceptions instead of
silently treating errors as false. Failed callbacks leave the input collection
unchanged. User callbacks retain the GIL; native selection remains in Rust.

```python
projected = empty.select(["score"]).rename({"score": "value", "kind": "category"})
assert projected.sum()["category"].to_list() == []
assert projected.sum()["value"].to_list() == []
assert len(projected.concat().source_views()) == 1
calls = []
result = empty.map(lambda table: calls.append(table))
assert calls == []
assert len(result.concat().source_views()) == 1
```

## Grouping and adapters

Grouping an empty sourced table retains its declared keys, physical schema and
source capture in both `group_by` and `groupby`, including node and edge group
adapters. An untouched empty collection has no inferred data schema; explicitly
declared keys and captured schemas remain available. Explicit aggregation specs
against a captured schema reject missing fields even on empty results.

```python
for method in ["group_by", "groupby"]:
    groups = getattr(source.head(0), method)("kind")
    summary = groups.agg({"score": "sum"})
    assert summary["kind"].to_list() == []
    assert summary["score"].to_list() == []
    assert len(summary.source_views()) == 1
```

Known text columns do not become numeric solely because row selection removed
all observations. A table created empty without prior observations has no value
type to infer; its columns follow the existing all-null/unknown role policy.
Empty numeric reductions have zero rows for zero-table collections; real empty
groups keep one result row per group. Aggregate/flatten outputs remain read-only
source results, without automatic graph writeback identity.

Invalid group metadata cannot be hidden by taking/filtering everything away:
fallible operations reject it, and existing infallible selections retain the
validation failure for subsequent checked operations. Public native table/key
fields are still available, but direct replacement/clearing of fields is not a
substitute for supported transformations with retained ownership.

## Native construction compatibility

`TableArray` now has private context. Native callers constructing struct literals
must use `new`, `from_tables`, `from_tables_with_keys` or `from_group_columns`.
The `tables` and `group_keys` fields remain public. Python call shapes are unchanged.
The native `with_empty_source` adapter accepts an explicitly empty BaseTable
schema/capture; it rejects property rows rather than treating them as metadata.

## Costs and scope

Schema-only context holds empty columns and shared frozen source payloads. Role
classification and group validation may scan existing cells when preparing a
context; table selections can still clone selected owned columns. Derived empty
results retain source payloads until their final owner drops them. No O(1), total
memory-budget, matched speedup or FFI latency guarantee is claimed.

This completes the supported collection selection/projection/rename/map/join
ownership family under 27/AR7.10. Schema-changing callback behavior on empty input
is explicitly unknown and not executed speculatively. Broader graph/table
conversion adapters, legacy connected reads and portable pooled storage remain
in their existing checklist entries.
