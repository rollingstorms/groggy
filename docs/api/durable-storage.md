# Durable checked graphs

`DurableGraph` adds a single-writer, binary transaction journal to the checked publication API. It starts a new empty graph; it does not turn an existing legacy `Graph` or its active JSON export into a complete backup.

```python
from groggy import DurableGraph

with DurableGraph.create("example.ggj", directed=True) as graph:
    writer = graph.writer(author="example")
    nodes = writer.upsert_nodes([
        {"external_id": "ada", "name": "Ada"},
        {"external_id": "lin", "name": "Lin"},
    ], effective_at=0)
    writer.add_edge(*nodes, relationship_type="KNOWS", effective_at=100)
    saved_state = writer.state_id

with DurableGraph.open("example.ggj") as graph:
    assert graph.current_state_id == saved_state
    assert graph.node_by_external_id("ada") == nodes[0]
    assert len(graph.commit_history()) == 2
```

Both `create` and `open` take path-like objects. Creation never overwrites a file. One OS file lock excludes a second writer/open handle; close the store or leave its context manager to release it. This is an embedded single-writer API. Multi-client coordination remains future work.

## Acceptance and failure contract

1. Parse and validate the complete proposed mutation, expected head, managed identities, installed schema and any additional session schema.
2. Capture one engine-recorded clock, resolve per-change effective times, and validate complete affected edge/endpoint lifetimes.
3. Encode a bounded journal record, append its header/payload, and sync the file.
4. Apply the prepared mutation and publish one in-memory state. Return success only after both the sync and publication succeed.

A successful call is recoverable after reopening, subject to the filesystem/device honoring sync. Opening verifies the format version, contiguous sequence numbers, length redundancy and payload checksum. Replay preserves data versions, typed deltas, authors/messages, original recorded/effective clocks, content hashes, graph configuration, allocation counters, external-ID reservations, branch heads and installed schema revisions. Derived caches are reconstructed.

A partial final record is discarded and its byte count is available in `recovered_tail_bytes`. A complete corrupt record causes an error without truncating the file. The checksum detects accidental corruption; it is not an authentication mechanism. Failed writes attempt to truncate and sync back to the previous record boundary. If rollback fails, the result is indeterminate and the store rejects graph reads/writes until reopened. A crash after sync but before acknowledgement may leave an accepted transaction that replay recovers; callers must reconcile using identities/state rather than assume a failed or missing acknowledgement means nothing was stored. Identical managed upserts do not append another record.

## Full-journal backup

`graph.backup(destination)` writes a separately recoverable copy of the accepted journal, including all retained versions, branches, schema revisions and identity reservations. It copies to a sibling temporary file, syncs and verifies that copy, publishes it without overwriting an existing destination, and syncs the directory. Later source writes do not change the backup. Open the result with `DurableGraph.open`; no special restore loader is needed.

Backup requires a healthy open store and an existing destination directory. The source writer stays exclusively borrowed while the copy is made. A copy/verification failure leaves no destination and removes the temporary file. A directory-sync failure after publication can leave a complete destination; inspect/reopen it before retrying. Never replace a live journal by copying a backup over it. Backup copies the full log and verifies it through replay, so time and disk cost grow with history. It is not a compact checkpoint. It includes persisted frozen group revisions; it does not import unrecorded raw pool contents.

## Installed policy

Pass `schema=` to `create` to install a policy. It applies to every durable publication and branch switch. A writer's optional schema adds validation; it cannot weaken the installed policy. `schema()` returns a copy in Python. `set_schema(schema, expected_state, expected_revision)` validates before installation/removal and persists an independently incremented `schema_revision`. Policy revision is global to the store, not a branch ontology migration.

`create_branch(name, expected_state)` and `checkout_branch(name, expected_state)` journal branch actions. A switch validates the target against the current installed policy before appending. Retained allocation counters and identity reservations survive switches and restart.

## Rust

`groggy::DurableGraph::create(path, config, schema)` / `open(path)` provide `apply_batch`, `apply_batch_with_schema`, managed upserts, policy changes and branch actions. `graph()?` borrows the graph immutably for reads; no mutable graph handle is exposed. Core mutation through `Graph::pool_mut` now requires `&mut Graph`, closing the interior-mutation escape through this read handle. Advanced callers of that accessor must hold an exclusive borrow.

## Current limits

The journal is format version 1 and uses CBOR records. Each record is limited to 256 MiB; larger transactions reject before append or mutation. All supported inline logical properties round-trip, including raw f32 bits and structured JSON values. Frozen group revisions are persisted through `retain_group`; `create_meta_node` publishes a reference to one. Nested dependencies, source clocks and typed properties survive reopen and backup. Unknown or membership-only references without explicit frozen source data reject. General graph import, periodic checkpoints, compaction/pruning, format migration tooling, backup orchestration and broader multi-client support remain under storage/history/operations roadmap items. Opening replays the journal, so recovery time grows with retained operations; this is not yet a bounded-recovery storage engine.

The Python wrapper releases the GIL for detached effective reconstruction and frozen group preparation. It holds the GIL during journal I/O/replay because the underlying graph is thread-confined. It does not promise concurrent Python reader execution. Ordinary `Graph` publication remains in-memory, with explicit manual compatibility available. Explicit checkpoint annotations now sync/recover through the journal without creating duplicate mutation states; they are separate from materialized recovery checkpoints. See [explicit checkpoints](explicit-checkpoints.md). Immutable effective-time reconstruction is available through `read_view_at`; see [effective-time reads](effective-time-reads.md).

Verification includes subprocess exit without close, replay after sync before in-memory publication, every partial-header boundary, partial payloads, checksum/length corruption, injected write/sync/rollback failures, schema rejection, stable history hashes, branches, reservations and typed property values. These tests exercise the implementation contract; they do not emulate every physical device failure.

Frozen group records are an additive journal record type with a versioned typed
payload. Existing records remain readable; old binaries cannot read the new group
records. Ordinary batches omit the new optional meta-node field when unused.
Active JSON export and arbitrary raw pool transfer remain separate contracts.
