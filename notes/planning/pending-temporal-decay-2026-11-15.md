# Selected pending temporal decay — 2026-11-15

## Completed slice

The existing temporal.decay stage now certifies pending selected reads. Node maps,
ordered node columns and edge maps retain explicit IDs, values and entity kind;
new entities and revised endpoints are checked against the selected source rather
than old owner liveness. One common read validation precedes processing, and edge
closure uses the selected endpoint reader. The decay formula, finite/control
rules, subnormal handling, private installation and cancellation remain intact.

In-place variables across nested iterations and outer atomic pipelines retain one
acceptance. Retained immutable native inputs can compute after owner deactivation
without borrowing that owner. Public write application still validates its source.
Seven other retained-history temporal steps retain their pending-topology guards;
unaccepted new entities are not assigned fabricated retained creation/history.

## Verification

Final locked/offline native retry: **1,593 passed / 54 ignored / zero failed**,
223 suite results including doctests. Strict all-target Clippy, Rust formatting
and whitespace checks passed. Rebuilt in-tree release extension with the three
established binding warnings; no wheel installation. **2,952 maintained Python
tests passed / one skipped / 14 established warnings**, plus **278 executable
documentation examples**. Black/isort/pre-commit remain unavailable.

The focused Python suite passed **106 tests**. Eleven new Python cases cover the
three layouts, local/outer/nested execution, in-place repeated decay, explicit old
maps with removed IDs, invalid domains and complete prefix rollback. One new native
test holds an exclusive owner borrow while decaying retained inactive node/edge
values, proving the immutable source is sufficient. A test-only missing import was
corrected before the final native retry.

Native release benchmark, five observations per size: directed ring plus one
staged node/two edges, one loop with node-ID load, decay, property attachment and
one publication. Medians **1,000 initial nodes: 3.240 ms; 10,000: 26.953 ms**.
Includes pending reads, checks and acceptance; excludes fixture, initial capture,
compilation, algorithm construction and assertions. No before/after speedup, FFI,
memory or scale guarantee. Receipt:
`notes/benchmarks/pending-temporal-decay-2026-11-15.json`. Reproduce using the
atomic_builder_properties bench's `--pending-decay` mode and
GROGGY_ATOMIC_BUILDER_PROFILE for the output path. The existing shared pending
fixture runner now uses a typed workload selector for its three compute cases.

## Checklist and next execution boundary

Advances existing 09.4b.2, 30.1 and AR7.8; **215 checked / 53 unchecked**, including
**44 unchecked release rows**. No rows added or broad parents closed.

The next subgraph/for-each slice must keep the original execution root separately
from each fixed child selection. Materialize the shared pending root, then derive
an exact child source from the same pin. Property reloads must see sibling writes
without expanding a child to every provisional node or rejecting siblings' valid
staged targets as foreign. Emitted child subgraphs must preserve their source pin.
Qualify these frames before permitting topology-mutating descendants. Existing
owner, selection closure, deterministic ordering, rollback and custom-effect guards
remain required; a capability annotation alone cannot implement this boundary.

Follow-up maintenance now reclaims expired weak capture-authority metadata with
amortized work, preserving earlier live table authority. See
`capture-authority-churn-2026-11-15.md`; no new checklist row was added.
