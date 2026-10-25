# 12.3 exit review — explicit checkpoints

The original 12.3 requirement is complete. Its earlier Partial label was left
over from the checkpoint implementation before the ordinary Python publication
defaults migrated. This is a status correction supported by existing code and
passing conformance tests, not a newly added capability or checklist row.

## Requirement and evidence

| Required behavior | Authoritative implementation / evidence |
| --- | --- |
| Ordinary checked writes publish their own deltas before any checkpoint. | Checked publication and ordinary Python default migration; `tests/test_automatic_publication.py` and the current root suite. |
| Clean commit/checkpoint labels the current published state. | `Graph::commit` in `src/api/graph.rs`; `src/api/graph/checkpoints.rs`; native GraphWriter checkpoint. |
| Labels consume no state ID/delta and preserve publication clocks. | `tests/explicit_checkpoints.rs`; `tests/test_explicit_checkpoints.py`; repeated and empty labels included. |
| Explicit dirty-state handling is compatible and visible. | `checkpoint` rejects dirty state; legacy/manual `commit` publishes once then labels the result. Both native and Python compatibility cases pass. |
| Stale head/schema failures do not add labels. | Existing native/Python checkpoint tests; writer preconditions and schema validation. |
| Durable annotations survive reopening and backup with original timestamps. | DurableGraph checkpoint coordinator, journal replay, branch/reopen/backup integration tests. |
| Journal failures cannot expose unaccepted labels or consume mutation IDs. | `src/storage/journal.rs` checkpoint fault-injection, after-sync recovery and invalid-record tests. |
| Retained read selections remain independent of later writes and labels. | Existing checkpoint integration tests verify frozen views and retained-state reads. |

The current full gate covers these existing suites: **1,150 native passed /
51 ignored; 1,881 root Python passed / 1 skipped; 133 executable docs passed;
strict all-target Clippy passed**. No additional tests mirror the same
implementation for this documentation-only status change.

## Scope that remains in its original items

- 09.4b.2: native default/infallible compatibility and remaining mutation escape
  paths. Explicit legacy/manual publication compatibility is documented here;
  closing 12.3 does not close that wider migration.
- 10.4 / 15: compressed recovery images, checkpoint scheduling, format upgrades,
  compaction, pruning and reclamation. A named checkpoint is an annotation of an
  accepted state, not a materialized recovery image or permanent retention pin.
- 12.2: complete portable snapshots including pooled/history contents.
- 16.2–16.3: arbitrary effective-time reconstruction and optional recorded cutoff.

12.3a remains the narrower implementation evidence row. Marking its parent
complete changes the overlapping inventory from **192 checked / 76 unchecked**
to **193 checked / 75 unchecked**; no rows were created or removed.
