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
