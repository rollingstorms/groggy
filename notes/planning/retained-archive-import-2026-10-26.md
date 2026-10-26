# Complete retained archives and existing-graph journal import (10.3)

The next owner-authorized priority window began 10:33 UTC on 2026-10-26. This
slice closes the existing 10.3 inventory, without adding checklist rows.

## Delivered

- Explicit `Graph.save_archive/load_archive` and native detached `PreparedArchive`.
  Versioned CBOR envelope, byte count/complement/checksum and strict document
  version; semantic restore rejects malformed history, parents, hashes, counters,
  selected topology/column indices, group dependencies and identity reservations.
- All retained raw column positions and logical typed values, inactive allocation
  slots, sparse allocation high-water marks, branches/tags/merge ancestry,
  publication/effective clocks, checkpoint labels, configuration and publication
  policy survive independent restoration. Float bits remain exact. Storage
  optimization representations and caches are rebuilt.
- Frozen nested sources preserve their own identity domains; raw legacy memberships
  and frozen source aliases are distinct. Typed entity-ID array values are retained
  verbatim, consistent with the existing property API.
- Dirty sources reject. Capturing does not accept pending edits. An archive is a
  full copy, not an ordinary mutation or a new checkpoint/acceptance timestamp.
- Files validate/encode before creating a synced sibling temporary; publication
  cannot overwrite an existing destination. Directory sync can report an error
  after publication, leaving a complete file to reconcile. Python archive file
  I/O/checksumming releases the GIL; owner capture/restore stays on its thread.
- `DurableGraph.create_from_graph` writes a complete source as one initialization
  frame, then uses existing checked writes, schema policy, replay, tail handling,
  locking and backup. Original source ownership is independent; no IDs or clocks
  are reissued. Frame-size/schema/dirty validation precedes destination creation.

This is additive to active JSON interchange, which keeps its existing smaller
contract. Old journals remain readable; old binaries cannot read new initialized
archive journals. Import's baseline uses the existing 256 MiB frame limit. A failed
new journal initialization may leave an incomplete new file, as with existing
journal creation; it never replaces an existing graph. Standalone archives do not
have the journal frame-size limit.

## Measured scope

`cargo bench --bench graph_archive` (offline/locked release) verifies restored
values, branch history and unchanged raw column counts at 1k/10k/100k nodes, two
accepted versions and two retained score values per node. Local 100k receipt:
**37,425,515 bytes**, **2.448 s save including validation/sync**, **1.491 s load**.
Save includes a validation restore; it deliberately pays full retained-copy cost.
These single local timings are not mutation throughput, peak RSS, or a storage
compression claim. Receipt: `../benchmarks/graph-archive-2026-10-26.json`.

## Open work retains existing scope

10.4 still owns materialized recovery checkpoints, physical journal compaction,
format migrations and crash-safe replacement/recovery-cost policy. Full retained
archive loading parses retained history and validates branch tips: it does not
promise constant or bounded opening time. 15 still owns raw value/entity/group/
identity reclamation and the no-op optimize implementation. Native mutation escape
migration remains 09.4b.2; archives reject inconsistent selected state rather than
hide it. No owner decision changed.

Executable API description: `../../docs/api/graph-archives.md`. Native regressions
cover complete round trips, exact bits/absence, clean legacy clocks, independent
source scopes, sparse counters, independent prepared bytes, semantic corruption,
no-clobber, schema, durable continuation/reopen/backup. Python regressions cover
the same public workflow. Verification totals are recorded in the priority-window
summary after the final gates.

## Slice verification

- Full native gate: **1,230 passed / 51 ignored**, plus the final five targeted
  archive unit cases after adding resolved merge coverage.
- Root Python contract: **1,969 passed / 1 skipped**, and **154 executable docs
  examples**. Owner benchmark/legacy nested survey suites remain excluded.
- Strict all-target Clippy, scoped rustfmt, Python/stub syntax and diff checks.
- Extension rebuilt/staged in tree; the three pre-existing FFI warnings remain.
  Black/isort remain unavailable; no packages were installed.
