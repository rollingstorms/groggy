# Explicit retained-graph integrity and observability — 33.2 component

No new checklist rows; **201 checked / 67 open**. The broad 33.2 operational scope
remains partial; this completes its retained-memory diagnostic component.

Graph.check_integrity and DurableGraph.check_integrity now return a structured
native/Python report after verifying portable recoverability and every retained
recorded state, including unreferenced/nonhead versions. Invalid data raises a
typed error without repair/publication/ID consumption. Reports include selected
state/branch, durable schema revision, active/allocated entities, raw column/slot
counts, retained/legacy/verified versions, branch/group/identity references and
historical maximum active cardinalities. Frozen-source payload values are not
included in raw slot counts; aliases are counted as references, not unique groups.

The existing recovery validator remains authoritative for config/types/raw indices,
allocation/references/groups/identities, hashes, publication keys, branches and
selected-state agreement. The new first-parent walk validates before-values,
entity lifecycle, allocated edge identity, endpoint closure and active property
ownership at every retained version. It reuses linear snapshots and keeps only
first-parent snapshots needed by descendants. Merges retain their existing accepted
first-parent resolution; this does not implement three-way conflict resolution.
Errors identify state/entity/property, and invalid raw encoding now includes its
column and slot even for an unused retained value.

Native prepare_integrity_check freezes owned portable input; execute can run on a
separate thread and still reports the old capture after source changes. Python
releases the GIL during verification, while capture remains owner-bound and copies
portable data. Pending manual changes reject; clean untimed legacy history is
counted and checked without inventing effective timestamps. Durable checks enforce
the captured current schema, not historical retroactive policy.

## Verification

Native tests cover detached ownership, groups/branches/reservations, legacy and
pending state, durable schema/file/lock preservation, transient create/remove and
manual edit/remove histories, hidden bad before-values, inactive nonhead properties
and invalid unused raw JSON diagnostics. Four public Python tests and two executable
docs examples exercise the exposed methods. API: `docs/api/retained-integrity.md`.

Release fixture: 1k nodes / 1k property updates / 1,002 retained versions, three
samples. Median capture/execute: **3.48/39.33 ms** (one mutation branch),
**2.97/42.55 ms** (eight interleaved mutation branches). Setup excluded; no SLA.
Receipt: `notes/benchmarks/retained-integrity-2026-10-26.json`.

Final gate: **1,270 native passed / 52 ignored**, plus the dedicated serial
checkpoint SIGKILL matrix **1 passed** (six boundaries); **2,004 root Python
passed / 1 skipped**, **160 docs examples**, strict all-target Clippy, rebuilt
in-tree extension. Owner benchmark and nested legacy survey suites remain excluded.
Three pre-existing FFI warnings remain; no packages installed.

## Remaining existing scope

This is an explicit expensive audit: state checking scans topology/mappings at
versions, and branch frontiers can clone full snapshots. Decoded history/source
memory still scales with retained data. It is not run automatically per mutation.
Durable checks do not reread an already open journal file; reopen verifies physical
framing/checksums/recovery under existing rules. Derived secondary indexes/caches,
in-place physical file comparison, migration/platform/service diagnostics and their
metrics remain under the original 18.3/10.4/33.2/33.3 scopes. No raw purge or implicit
maintenance policy was introduced.
