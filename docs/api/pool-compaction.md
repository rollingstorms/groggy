# Explicit pool compaction

`Graph.compact_pool()` removes duplicate retained property slots and returns
before/after node and edge slot counts. `Graph.optimize()` performs the same
maintenance and returns `None`. Both require an accepted, clean working state.

```python
import groggy as gr

g = gr.Graph()
g.add_node(value=1)
for _ in range(5):
    g.set_node_attr(0, "value", 1)
state = g.current_state_id
frozen = g.read_view()
report = g.compact_pool()
assert report["current_node_values"] < report["previous_node_values"]
assert g.current_state_id == state
assert frozen.node_attr(0, "value") == 1
assert g.optimize() is None
```

This is exact portable-value deduplication within each column: signed zero, NaN
payloads, scalar/vector widths, null, text and JSON spelling retain their value
identity. Internal small-value/compressed representations may share a slot when
they represent the same portable value. The first matching representation wins;
hashes only select candidates and collisions are checked for exact equality.

Every distinct raw value remains, including values without a current active
mapping. Branches, versions, inactive entity IDs, group references, external ID
reservations and all historical deltas remain. This does not delete a branch,
purge an entity, release an external name or discard unreachable distinct data.
Immutable graph/table read pins own their frozen values and remain readable.

Raw column offsets can change. Callers using low-level pool offsets must acquire
them again after maintenance; node/edge IDs and logical graph versions do not
change. All working mappings are remapped, including inactive mappings.

The explicit operation scans raw values and temporarily copies retained column
payloads. The native prepare/execute/finish API permits detached computation;
Python releases the GIL during execution. Capture and installation run on the
owner. A stale or foreign detached result rejects before changing storage.
Before/after counts measure slots, not bytes, unique history values or peak RSS.
Future restoration/writes can grow the columns again.
