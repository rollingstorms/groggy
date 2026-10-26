# Endpoint differences and reconstruction safety — 2026-10-26

Existing scope: **04.2 / 13.3**, after retention closure. This is a completed
comparison/safety component; those broad rows remain open. **204 checked / 64 open**,
no new checklist rows. The next implementation remains coordinated rewiring and
reversal rather than unrelated facade cleanup.

## Demonstrated failures repaired

1. `GraphSnapshot.diff_with` compared edge keys but ignored different endpoint
   tuples under the same ID, allowing an endpoint-only difference to look empty.
2. `GraphPool.add_edge_with_id` could overwrite a retained edge's only raw endpoint
   pair, bypassing typed history; working-state restoration could also replace
   membership before discovering the conflicting reconstruction.

`state::change::EdgeEndpointChange` is the canonical readonly before/after record.
Snapshot/commit diffs carry sorted endpoint records, preserve edge instances,
report true additions/removals separately, and include endpoint-only changes in
emptiness and edge-change counts. Reverse diffs swap before/after tuples.
Undirected graphs still compare the ordered stored tuple; this does not change
traversal semantics. Snapshot diffing adds one linear comparison pass and only
allocates endpoint records for changes.

Raw restoration now rejects attempts to change an already materialized edge's
endpoints, while identical endpoints and previously unmaterialized IDs still work.
Graph restoration preflights conflicts before replacing any working membership.
Dense and explicitly purged sparse pools follow the same rule. This prevents a
storage escape from impersonating a coordinated endpoint change.

Python `Graph.diff_commits` marshals the additive `edge_endpoint_changes` list as
`{edge_id, old, new}` dictionaries. It contains no new business logic. Native
StateDiff/CommitDiff expose the canonical records; existing compatibility imports
share that type. No mutation or archive/journal format is added in this slice.

## Verification

- **1,310 native passed / 52 ignored**; direct snapshot comparisons cover directed
  and undirected graphs, reversal, parallel instances, loops, stable ID ordering,
  endpoint-only changes, mixed addition/removal, inverse diff and identity diff.
- Internal boundary regression checks dense/sparse pool reconstruction rejection,
  unchanged complete archive bytes/membership/allocation, valid identical restore
  and full retained integrity.
- **2,017 root Python / 1 skipped**; existing owner/legacy exclusions retained.
- **168 executable docs examples**: prior 167 plus the new endpoint example.
- Strict all-target Clippy and rebuilt in-tree extension, with the existing three
  unrelated FFI warnings. Scoped formatting, syntax and diff checks.

## Fixed remaining endpoint/reversal implementation inventory

These are dependencies within the existing requirements, not additional checklist
rows or new owner questions:

1. Typed recorded endpoint delta with before/after validation, canonical hashing
   and compatibility for older deltas that do not contain endpoint changes.
2. Prepared scalar/bulk/mixed batch writes and one acceptance; stable edge ID,
   explicit effective times or one engine clock, duplicate/result validation.
3. Version-aware endpoint storage/selected-state restoration and cache invalidation;
   preserve historical endpoint tuples and independent immutable sources.
4. Effective topology events, old/new endpoint lifetimes, backdating/tie semantics
   and rejection of contradictions before mutation or journal I/O.
5. Archive/journal codecs and semantic verification, durable replay/backup/failure
   recovery and purge closure for every before/after endpoint dependency.
6. Public Rust/Python setters and explicit reversal/revert policy with GIL release
   for long operations. Preserve properties and external edge identity.
7. Legacy membership-only group behavior must preserve its documented scope or
   reject unsupported rewiring; frozen revisions already own independent payloads.
8. Recorded/effective/owned views, branch switching, parallel/loop cases, atomic
   batches and recovery examples must verify the complete mutation contract.

Three-way merge planning/caller resolution (14.2) follows this endpoint model.
This slice does not claim a supported rewiring API exists yet.
