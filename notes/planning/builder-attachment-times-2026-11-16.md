# Direct builder attachment clocks — 2026-11-16

## Repaired gap

The direct node/edge attachment factories previously ignored `effective_at` and
other unknown keys. Both now accept a strict optional signed-microsecond clock,
reject other keys/types, and stage explicitly timed node maps, node columns and
edge maps through PropertyBatch::stage_at. A shared StepScope helper retains exact
child-frame validation; Context retains canonical whole-input validation. Existing
untimed constructors and their fast/default/transient paths are unchanged.

Timed attachments classify as staged persistent effects: an atomic persistent
builder/outer run is required, including zero-iteration bodies. There is no second
publication path. Explicit times replace prior pending clock choices; an unstamped
later assignment restores the batch clock. One final assignment per key is retained,
not an implicit history of intermediate computations. Distinct accepted writes
at the same effective timestamp retain later-accepted-write precedence.

Canonical temporal rules are unchanged. Properties outside an entity's effective
active lifetime remain in accepted history but do not carry into a later activation
in effective-time reconstruction. Accepted-state reads remain distinct. Initial
new test assumptions about rejection/carry-forward were corrected against the
existing documented table-attachment contract; no new lifetime policy was invented.

## Verification

- Two new native regressions exercise all three input representations, provisional
  rows, exact before/at reconstruction, one publication and strict parameter parsing.
- 55 new Python cases cover local/outer staging, source row types, explicit metadata,
  archive reconstruction, later clock replacement/defaulting, exact accepted ties,
  malformed clocks/unknown keys even in zero loops, late rollback, fixed child
  rejection and success, persistence guards and outside-lifetime backfills.
- Focused direct/table attachment, root-loop and child-confinement suite:
  **141 passed**.
- Full locked/offline native: **1,610 passed / 54 ignored / zero failed**, 225
  suite results. Strict all-target Clippy, Rust formatting and whitespace pass.
- Rebuilt in-tree release extension: **3,273 maintained Python passed / one
  skipped / 14 established warnings**, **287 executable documentation examples**.
  CPU-sensitive gate ran after native builds/tests; artifact qualification is separate.

No FFI business logic, storage schema, implicit checkpoint or automatic index change.
No new allocation/throughput guarantee. Extends existing 09.4b.2, 07.4, temporal
clock conformance and 19.2; fixes the emerging issue recorded in the preceding root
loop slice. Broad checklist inventory remains 215 checked / 53 unchecked, with
44 open release rows and nine deferred. Unrelated owner files are untouched.
