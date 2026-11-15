# Explicit compatible schema migration

Durable stores persist an installed policy and its revision. Policy installation
validates the current graph before appending a record. Data conversion is a
separate checked batch: choose an explicit compatible intermediate policy, convert
all affected values in one batch, then install the tighter policy.

```python
import tempfile
from pathlib import Path
from groggy import DurableGraph, GraphBatch

def policy(kind):
    return {"node_labels": {"Entity": {"code": {"type": kind}}}}

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "migration.ggj"
    with DurableGraph.create(path, schema=policy("int")) as store:
        node = store.writer().add_node(labels=["Entity"], code=10)
        original = store.current_state_id
        old = store.snapshot_at_commit(original)
        assert store.set_schema(policy("any"), original, 0) == 1
        batch = GraphBatch()
        batch.set_node_attr(node, "code", "10")
        converted = store.apply_batch(batch, original, "convert code", "example")["state_id"]
        assert store.set_schema(policy("text"), converted, 1) == 2
        assert old.node_attr(node, "code").value == 10
    with DurableGraph.open(path) as store:
        assert store.schema_revision == 2
        assert store.get_node_attr(node, "code") == "10"
        assert store.snapshot_at_commit(original).node_attr(node, "code").value == 10
```

`any` keeps required/nonnullable rules; it does not make missing or Null values
valid. An intermediate policy broadens the types accepted by writes during that
stage. Use the single-writer boundary to control those writes. Each policy change
and each conversion batch is separately recorded; the entire sequence is not one
transaction. On interruption, reopen, inspect the revision and current values,
and explicitly resume the remaining stages. No automatic conversion or coercion
occurs.

Rejected policy installations or batches preserve the journal and graph state.
Retained immutable reads keep their original values. The current installed policy
is global to the durable store: switching to an incompatible historical branch
rejects until the caller explicitly installs a compatible policy. Historical
reads remain possible. Branch-specific policy history, automatic migration
planning and arbitrary storage-format upgrades remain separate open work.

The regression workflow also covers a rejected mixed conversion, incompatible
branch checkout, full backup and reopening both source and backup.
