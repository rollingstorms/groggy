# Complete explicit retained-storage purge — 2026-10-26

Existing scope: **06.3, 15.1, 15.2**, following history reachability collection,
full retained archives, raw-property compaction and verified physical replacement.
This closes those three core entries under the accepted embedded-library policy;
it does not close broader 10.4/AR6 production, platform, scale or scheduling work.
No checklist rows or owner decisions added.

## Accepted policy and implementation

A09: purge is explicit, preserves shared retained history and open immutable
payloads, adds no permanent audit/checkpoint pins, grace period or mandatory
backup. History pruning/collection is a separate explicit action. Inactive IDs
remain reserved until purge can prove no owner-domain retained root needs them.

`Graph.preview_purge` and `purge_unreferenced` share one complete candidate path.
Native prepare/execute/finish captures portable bytes plus history-owner identity,
checks retained integrity, compacts orphan properties, traces roots, collects and
verifies, then rejects stale/foreign input before installation. The pool is
replaced in the original Rc and history identity stays stable, preserving current
immutable-view write authority. No accepted version/clock/checkpoint is invented.
Candidate pool ownership moves into the outcome without an extra full pool clone.

Root closure includes current membership/mappings, all retained delta targets,
added-edge endpoints, before/after property values and group references. Nonhead
and otherwise unreferenced retained versions protect data until explicit gc.
Legacy groups protect owner members/endpoints. Frozen source revisions and nested
sources own separate ID domains/payloads; owner registrations can be reclaimed
without deleting their nested immutable content. NodeArray/EdgeArray remain
numeric values rather than hidden strong pins, matching property-values.md.

External reservations survive while current or retained historical identity
properties reference the same binding. Purge also releases obsolete names of
still-retained entities when the naming version is explicitly collected. Global
name reuse is permitted afterward; monotonic internal node/edge IDs never reuse.
Content-derived group references may be explicitly registered again from the same
source. Full archives, owned copies, journal replay and backups retain purge state.

## Reclaimed physical representation

Sorted half-open permanent purge intervals avoid a per-ID tombstone allocation.
A trillion-node/two-trillion-edge reserved extent is represented by two intervals;
subsequent node and edge creation and archive roundtrip work at those high IDs.

An actual implementation issue found in this slice was dense interior endpoint
retention and huge empty prefixes on later insertion. Explicit edge purge now
moves surviving raw endpoints into a sparse HashMap and releases all dense arrays.
Ordinary graphs retain their dense fast path. Bulk insert/lookup/incident readers,
state restoration and archives support both representations. Sparse incident IDs
are sorted to preserve prior order. Active GraphSpace traversal retains its CSR.

Legacy nonpurged archives keep document v1 shape; v2 supports dense ledgers and
v3 supports sorted sparse endpoint records. Readers reject unsupported versions,
layout downgrades, mixed storage, duplicate/unsorted/out-of-bound sparse records,
malformed intervals and retained targets/endpoints/memberships into purged IDs.
Version migration tests also keep legacy imports readable.

## Durable protocol

`DurableGraph.purge_unreferenced(expected_state)` uses the existing locked Unix
checkpoint replacement protocol, patching a complete archive with the exact purge
candidate. It syncs and verifies the replacement before rename and installs the
same candidate into memory. Pre-replacement failures leave original memory and
bytes unchanged; post-replacement errors poison the writer until reopen. Both
inodes remain locked across replacement. Schema revisions, publication clocks,
checkpoint labels and retained versions remain unchanged. Continued writes, name
reuse, reopen and backup are verified.

Five injected failure boundaries plus six real SIGKILL boundaries test purge.
The existing serial process matrix now has **18 cases**: six ordinary checkpoint,
six orphan-value compaction and six full purge cases. These include interrupted
large baseline fragments, exact old/new allocation/name state, lock release and
continued monotonic allocation after reopen. Actual machine power-loss and
non-Unix replacement remain existing 10.4/31.4 work.

## Measured receipt and trade-offs

`cargo bench --offline --locked --bench retained_purge` with
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`;
receipt: `../benchmarks/retained-purge-2026-10-26.json`.
Three local release samples per fixture; setup/history pruning excluded.

| Orphan nodes and edges | Capture median | Candidate median | Validate/install median | Durable total median | File before → after |
| --- | --- | --- | --- | --- | --- |
| 1k each | 1.93 ms | 10.69 ms | 2.20 ms | 28.20 ms | 169,258 → 1,627 B |
| 100k each | 239.53 ms | 1.333 s | 260.97 ms | 2.035 s | 8,769,890 → 1,635 B |

Both fixtures retain one live node/version and release 1k names plus an orphan
group registration. The extremely small final file follows from that fixture;
retaining actual history/data continues to cost storage.

A separate 50k-retained/50k-purged raw-endpoint fixture measures 500k lookups over
five samples: dense median **2.99 ns/lookup**, sparse median **14.78 ns/lookup**.
This records the locality/hash cost rather than implying purge has no runtime
trade-off. It is not an end-to-end graph traversal or FFI throughput result.

Capture/final byte comparison run on the owner and encode/copy retained data;
Python releases the GIL for candidate work and durable I/O. Full temporary copies
can cost substantial memory. These are explicit maintenance paths, not ordinary
mutation hooks. No isolated peak RSS, FFI budget, 1B scale or universal latency
claim. Automatic scheduling stays deferred per A02.

## Verification

- Full native offline/locked suite: **1,308 passed / 52 ignored**.
- Dedicated serial SIGKILL gate: **1 passed / 18 cases**; still one ignored-by-default
  parent invoked separately. No additional ignored tests.
- Latest archive downgrade regression separately passes after its final extension.
- Root Python contract suite: **2,017 passed / 1 skipped**; owner benchmark test
  and legacy nested survey exclusions retained.
- **167 executable documentation examples**, including three purge examples.
- Strict offline all-target Clippy; rebuilt/staged in-tree extension, three known
  unrelated FFI warnings. No external installs or push.
- Scoped Rust formatting, Python/stub syntax and diff checks.

Tests cover preview identity, stale/foreign/dirty outcomes, still-retained versions,
legacy/frozen/nested groups, retained/obsolete identity names, independent owned
captures, current pin write authority, numeric ID arrays, interior endpoint arrays,
high-counter insertion, malformed archives, schema/clocks, continued writes,
verified physical bytes, backup and failure recovery.

## Checklist and next work

**201 checked / 67 open → 204 checked / 64 open**, same 268 rows. This is an
inventory including overlapping parents/children, not a completion percentage.
Next core priority is typed endpoint changes/reversals (04.2/13.3), then three-way
merge planning/caller resolution (14.2). Remaining execution/default boundaries,
view/index/query closure and numerical/scale/release parents retain their scope.
Graphiti/retrieval/service and automatic maintenance schedules remain deferred.
