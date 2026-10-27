# Tracked restoration exit — 13.3


## Explicit restore modes completed — 2026-10-27

Owner approved one restore_state API with required mode="tracked" or mode="switch".
Tracked restoration records one exact current-parent delta with explicit/default
effective clocks; switch retargets the branch without publication. Graph,
DurableGraph and native/Python writers share the contract. Exact delta preparation
and checked snapshot publication are shared with merges.

Existing 13.3 is complete: native/Python compatibility, stale/dirty/schema/clock
guards, frozen sources, history/identity preservation, durable replay/rollback,
backup, explicit source-target pruning and compaction, plus five real SIGKILL
cases. Full verification: 1,376 native / 56 ignored; 2,040 root Python / 1 skipped;
182 executable docs; strict all-target Clippy; rebuilt in-tree FFI. The final
focused rebuilt restore/merge suite also passes (18 tests). Three existing FFI
warnings remain. No new checklist rows: **207 checked / 61 open**.

Local release medians for 100k-node chains with all properties changed: switch
reconstruction 30.9 ms + validation/install 132.5 ms; tracked replay/comparison
240.2 ms + validation/publication/install 206.8 ms. This measures full-state
restoration, not ordinary mutation overhead. Receipt:
`../benchmarks/state-restore-modes-2026-10-27.json`. Full replay/assembly costs,
RSS/production targets and broader lifetime/GIL work remain explicit limits.

Next priority is existing canonical state/change ownership (AR7.9), followed by
compound topology acceptance and remaining view/index/query boundaries.

## Contract and compatibility

Mode is required in Rust capture/direct APIs, Python Graph/DurableGraph and writer
methods. Existing shipped callers/tests/examples explicitly add switch. There is
no separate revert_to_state API. Switch rejects change timestamps. Tracked uses
GraphPublicationOptions (GraphMergeOptions remains an exact type alias),
per-change overrides and one engine clock otherwise. Native writer author/schema/
time defaults apply. A data-identical explicit tracked action records a version.

Private captured provenance binds owner, branch, current head, retained target and
mode; off-owner reconstruction computes whole exact typed before/after changes.
Shared snapshot publication checks current identity/reference/schema and complete
effective topology before mutation. The new state's only parent is the old head,
not the source target. No implicit permanent source pin, prune or checkpoint is
created. Other branches, retained versions, monotonic IDs and immutable views survive.

Durable records reference the preceding retained target with original metadata.
Replay reconstructs and validates before installation; sync precedes memory
acceptance. Four injected append boundaries preserve bytes, state and allocation.
Six semantic replay cases cover unknown/stale inputs, foreign branch, missing
clocks and unrelated overrides. Five actual SIGKILL phases cover preappend,
partial header/payload, synced-before-memory and installed publication.
A sibling target can later be explicitly collected; the accepted delta owns the
restored payload, including frozen source references, through baseline replacement,
purge, compaction, backup and reopen.

Verification logs reside under /tmp/groggy-tracked-* for this work session.
Native tests cover exact signed zero, Null, topology/parallel edges/loops, stale/
foreign/collected targets and timestamp atomicity. Python tests cover required
mode and invalid input, writer retargeting/author, manual dirty work, effective
reads and durable recovery. No production-scale, RSS or power-loss claim is made.
