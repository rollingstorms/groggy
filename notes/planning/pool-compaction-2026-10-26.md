# Explicit retained property compaction — 2026-10-26

Existing scope: 15.2 / 10.4. This closes the no-op `Graph.optimize` component
with real exact-value column deduplication. It does **not** close broad raw
entity/group/reference-closure collection or destructive identity release.

## Contract and implementation

- Native and Python `Graph.compact_pool` return node/edge slot counts;
  `Graph.optimize` uses the same implementation and discards the report.
- Distinct portable values remain, including inactive/unmapped values. No history,
  group, entity allocation, external identity reservation or branch is removed.
- Every working column mapping is rewritten, including inactive mappings. Logical
  IDs, accepted graph version, publication clocks and frozen read pins remain.
- Hashes find candidates; exact portable comparison decides equality. Signed zero,
  NaN payloads and JSON spelling retain exact identity. Portable equivalence can
  coalesce internal compressed/small representations without changing values.
- Detached capture copies raw payloads. Execute owns its buffers and releases the
  Python GIL; installation checks owner identity, state, original mappings and raw
  lengths. Foreign/stale/dirty results reject before changing columns or mappings.
- Explicit compaction invalidates low-level raw offsets. This is an expensive
  maintenance operation; it does not add work to ordinary mutations. Future writes
  and restoration can grow columns again.

Implementation: `src/storage/pool/compaction.rs`,
`src/api/graph/compaction.rs`, and the existing GraphSpace column mappings.
The graph coordinator owns maintenance policy; FFI only detaches and marshals.

## Evidence and remaining scope

Native tests cover selected node/edge remapping, branch/history/inactive values,
exact floating/JSON/null representations, immutable pins, retained groups,
archive round trips, stable allocation, idempotence and stale/foreign/dirty plans.
Python tests cover public behavior and round trips. Local release benchmark:
`../benchmarks/pool-compaction-2026-10-26.json` (prepared fixtures, estimated retained
column memory; not peak RSS or total graph memory).

Destructive deletion of distinct values/entities/groups, identity release and full
retained-reference closure remain 15.1/15.2/06.3. No automatic maintenance schedule
or purge policy is introduced. No checklist rows are added.

## Local release result

10k nodes / five bulk text revisions create 50k text slots. Three-run medians:

| Distinct text values | Text slots after | Capture | Detached execute | Install | Estimated text-column bytes before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 10 | 1.24 ms | 74.19 ms | 1.69 ms | 2,400,296 → 736 |
| 10,000 | 10,000 | 0.97 ms | 71.78 ms | 1.70 ms | 2,662,736 → 492,298 |

These estimates exclude retained history and detached capture peaks; repeated text
can already use compact internal representations. They are not physical RSS savings
or guarantees for unique/uncompressible values. No ordinary-write speed claim.

Gates: **1,277 native passed / 52 ignored**, **2,007 root Python passed / 1
skipped**, **161 executable documentation examples**, strict all-target Clippy,
rebuilt in-tree extension and scoped formatting/diff checks. Default ignores include
the separately required process-death matrix, which is rerun for the following
durable slice. Owner benchmark and nested legacy surveys remain excluded. Three
pre-existing FFI warnings remain; no packages installed.

## Durable physical installation

`DurableGraph.compact_storage` uses the existing checkpoint plan/outcome protocol:
prepare a raw-column capture plus full portable baseline, deduplicate off-owner,
remap the baseline, write/sync/verify a sibling file, replace it under continuous
inode locks, sync the directory, then install the same column result in memory.
An ordinary `checkpoint_storage` still preserves raw offsets. Its Python result
shape is unchanged; the explicit compactor adds nested `pool_compaction` counts.

Pre-replacement failure leaves original memory and named bytes untouched.
Post-replacement failure preserves original inaccessible memory, poisons the owner
and requires reopening the complete replacement. No partial in-memory remapping
occurs. Detached maintenance blocks owner reads/writes until completion.

Five injected failure phases and twelve actual SIGKILL/reopen cases (six ordinary
checkpoint plus six compacting checkpoint phases) verify old/new raw layouts,
complete clocks/history/schema/checkpoints and locks. Large interrupted-fragment
cases remain included. This verifies process death, not sudden power loss.
Native/Python tests also cover physical shrinkage in a repetitive fixture,
idempotence, branches, group pins, external reservations, stale rejection,
append/reopen and backup. No storage format change or package install.

Durable local release medians (three warm-cache fixtures, 1k nodes / ten bulk
text revisions): file **8,583,515 → 5,800,469 bytes**, reopen **34.37 → 22.31 ms**;
capture **7.74 ms**, detached deduplication/write/sync/verification **69.17 ms**,
installation **0.46 ms**. All eleven versions remain. Receipt:
`../benchmarks/storage-compaction-2026-10-26.json`. Not isolated RSS, unique-value
performance, physical power loss or a production latency guarantee.

Durable-slice gates: **1,279 native passed / 52 ignored**, dedicated serial
process-death matrix **1 passed / twelve cases**, **2,009 root Python passed / 1
skipped**, **162 executable docs examples**, strict all-target Clippy and rebuilt
in-tree extension. Owner benchmark/nested legacy survey exclusions and the three
pre-existing FFI warnings remain unchanged. No packages installed or push.

## Explicit orphan raw-value reclamation

The opt-in `reclaim_unreferenced=True` policy completes raw property-slot garbage
collection under 15.2: working mappings and **all retained delta before/after
values** are roots, including nonhead/unreferenced versions. Matching uses exact
portable value identity within the node/edge column. Frozen source payloads have
independent storage and remain untouched. Entity IDs, stored group registries,
external reservations and empty-column declarations are retained.

The graph coordinator supplies immutable history roots; the pool layer only
computes exact storage reachability/deduplication. The detached capture shares
immutable commit payloads via Arc and builds a unique borrowed root lookup. It
rejects installation if history allocation/count changed, even with unchanged
selected state and property mappings. Default compact/optimize semantics do not
change. Explicit manual reset plus reclamation can remove abandoned raw writes;
pruning history plus reclamation can remove side-branch-only values. No implicit
history pruning, grace period, backup or scheduling policy is introduced.

Tests cover inactive historical properties, before/after value protection,
nonhead/unreferenced retained versions, shared source domains and owned snapshots,
manual discarded values, stale history roots, exact signed zero, reservations,
archive and durable recovery/backup. Five injected failure phases run with both
default and reclaiming policies; the twelve-case process matrix now covers ordinary
checkpoints plus reclaiming compaction, with real orphan values and large fragments.

Remaining original 15.1/15.2/06.3 scope is raw entity topology/tombstone reclamation,
complete group/reference closure and external identity release. Property-value
reclamation and real verified physical compaction are implemented; this does not
claim the broad retention parent is finished.

Local orphan-reclamation release medians (three samples, 1k nodes / twenty pruned
bulk versions): **20,000 distinct orphan raw slots reclaimed**, total raw node
slots **22,000 → 1,001**, estimated retained `value` column bytes **1,966,712 →
57,696**. Capture **0.454 ms**, detached execution **21.02 ms**, install **0.457
ms**; durable total **48.24 ms**, file **~6.35 MB → ~0.73 MB**. Both retained
versions and an open discarded-state view survive. Receipt:
`../benchmarks/pool-value-reclamation-2026-10-26.json`. Column estimates exclude
history, immutable pins, temporary lookup/capture memory and allocator/RSS effects.

Reclamation final gates: **1,285 native passed / 52 ignored**, separately executed
serial process-death matrix **1 passed / twelve cases**, **2,011 root Python
passed / 1 skipped**, **163 executable docs examples**, strict all-target Clippy,
rebuilt in-tree FFI and scoped formatting/diff checks. Two internal corruption cases
verify a malformed unused value cannot be silently discarded and an invalid edge
mapping cannot partially install already compacted node columns. The existing owner
benchmark/nested legacy survey exclusions and three FFI warnings remain unchanged.
