# Captured table result attachment — 2026-11-15

## Completed implementation

`table.attach` selects/aliases explicit node or edge property columns from a
row-bound table produced by a capture issued in the current atomic run. Existing
graph projections, selected rows/columns, degree tables and captured algorithm
tables keep their retained identity. Added entity IDs can receive computed values
before the owner publishes; all effects accept once through the canonical batch.

Table preparation reuses the source-writeback validator: duplicate rows, edited
structural fields, malformed columns and reserved source columns reject. Aliases
cannot target structural/system fields or duplicate a destination name. Complete
pending membership and property-kind checks precede staging.

Weak logical-capture authority prevents unrelated or historical tables with equal
numeric IDs from being mistaken for current-run results. It retains no graph
payloads and lives only for the mutation run. Existing accepted-source writeback
and explicit mapped writeback retain their own provenance requirements.

Optional signed-microsecond effective_at becomes per-property overrides in the
same canonical delta. Replacing a pending assignment replaces its timestamp
choice; an unstamped replacement clears the previous override for that key.
Pending intermediate assignments remain unaccepted intermediates, not separate
history events. Earlier captures can supply explicitly selected surviving rows;
values are never silently refreshed or targets silently filtered. Existing
canonical behavior for assignments outside effective entity lifetimes is
preserved: accepted history retains them; effective reconstruction ignores them
rather than carrying their values into a later activation.

The native StagedProperties effect makes persistence/local-atomic requirements
explicit before prefix execution. No second mutation engine, FFI business logic,
implicit checkpoint or permanent capture pin is added.

## Verification

Full locked/offline Rust suite: **1,587 passed / 54 ignored / zero failed**,
223 suite results including doctests. Strict all-target Clippy, formatting and
whitespace checks passed. The rebuilt in-tree release extension passed **2,862
maintained Python tests / one skipped / 14 established warnings**, plus **274
executable docs examples**. The three established binding compiler warnings are
unchanged. No wheel was installed. Native gates used the in-tree target with
incremental compilation disabled; Python/docs used the configured Python 3.9
interpreter and `notes/scripts/verify_immutable_view_migration.py`.
Black/isort/pre-commit remain unavailable locally. Unrelated owner work is untouched.

Native release benchmark, five observations per size: **6.366 ms** median at 1k
initial nodes/edges and **67.911 ms** at 10k. The ring workflow stages one node/two
edges, captures pending state, computes a degree table, aliases/attaches two
columns and publishes once. Source authority, current pending materialization,
validation and result projection are timed; fixture, initial capture, compilation
and assertions are excluded. No before/after speedup, memory/FFI or scale claim.
Raw receipt: `notes/benchmarks/captured-table-attachment-2026-11-15.json`.
Reproduce via the atomic_builder_properties bench with `--captured-attachment`
and `GROGGY_ATOMIC_BUILDER_PROFILE` selecting the output path.
Coverage includes all 13 captured kernels, node/edge projections, reordered and
projected row identity, provisional entities, Null/f64 values, source immutability,
explicit/default replacement clocks, archive/time reconstruction, weak lifetime,
duplicate/tampered rows, invalid kinds/aliases/clocks and late rollback.

## Progress and boundaries

This advances existing **09.4b.2, 30.1 and AR7.8** without adding checklist rows.
Their broader requirements remain open: **215 checked / 53 unchecked**, with
**44 unchecked release rows**. Remaining sampling/community/nested effects,
public facade lowering, general topology loops, journal-owner dispatch and
naming/history policy keep their original exit requirements.

No automatic index persistence, external adapter integration or table row identity
inference is introduced. Captured outputs remain immutable and provisional even
after the accepted mutation; ordinary update_source_graph is not relaxed.

Contract/example: [captured table attachment](../../docs/api/captured-table-attachment.md).
