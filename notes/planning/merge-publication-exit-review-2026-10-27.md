# Merge publication exit — 2026-10-27

Closes existing checklist 14.2; inventory 206 checked / 62 open, fixed 268 rows.

## Contract and implementation

Native owner-bound capture reconstructs three retained inputs outside the graph
borrow. Explicit typed resolution retains exact scalar/vector payloads, absent
versus Null, endpoint order and stable parallel edge IDs. Concrete topology
repair is required; no silent cascades. The outcome carries an exact delta
relative to its first parent. Acceptance rechecks owner/head/branch/base, allocated
unpurged IDs, frozen group sources, external identities and full target schema.
One engine recorded clock and explicit/default effective times feed the shared
lifetime validator. One accepted version has two ordered parents, including
intentional data-identical ancestry joins. Manual policy remains unchanged; dirty
manual writes reject. Other branches, allocation counters and immutable pins survive.

Native and Python writer sessions retarget only after success. Python exposes
MergePlan/MergeOutcome and copied conflict dictionaries with native side selection.
Resolution/reconstruction release the GIL; owner validation/installation remain
thread-confined. Both installed and additional durable schemas are enforced.

Durable records store exact logical deltas, not whole snapshots or pool offsets.
Append/sync precedes memory installation; replay rebuilds and compares the exact
declared delta and metadata. Semantic corruption rejects without file repair.
An unexpected post-sync installation failure poisons the owner until reopen.

## Verification

- Full native: 1,369 passed / 54 ignored.
- Rebuilt root Python: 2,034 passed / 1 skipped (owner benchmark excluded).
- Executable documentation: 181 examples, including five merge examples.
- Strict all-target Clippy passed; FFI build succeeded with three existing warnings.
- Merge append-failure matrix: four stages; semantic replay matrix: seven cases.
- Serial real SIGKILL: five merge scenarios and eighteen existing baseline,
  compaction and purge scenarios passed (23 total).
- Native/Python regressions cover clocks, identity, callbacks/stale plans, exact
  signed-zero history and v5 archives, topology repairs, parallel edges, loops,
  schema atomicity, backup/reopen/maintenance and continued allocation.

One full Python run emitted an intermittent unsendable PyGraph finalizer warning
from worker-thread garbage collection during a later frozen A* test. No test
failed; a focused owner-thread callback-finalizer control did not reproduce it.
Thread-confined owner lifecycle/GIL work remains under existing 19.2; no unsafe
destructor change or warning suppression was introduced.

## Release benchmark

Receipt: ../benchmarks/merge-publication-2026-10-27.json; three local release
samples each, setup excluded, old immutable pin retained.

| 100k-node workload | Planning/reconstruction | Native publish | Durable publish | Journal delta | Reopen |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50k independent properties | 818 ms | 224 ms | 261 ms | 3,619,012 bytes | 2.452 s |
| One changed property | 542 ms | 109 ms | 138 ms | 427 bytes | 1.281 s |

Small deltas have bounded journal growth. Full replay, snapshot assembly and
installation remain proportional to selected graph size. No FFI/RSS, production
latency or billion-entity guarantee follows. Journal frame limit remains 256 MiB;
chunked storage/migration/platform and scheduling work remain original open scope.

## Next accepted scope

Owner now chose explicit restore_state modes: tracked publishes a new delta on
the current branch; switch selects a retained state without publishing. Mode
selection must be explicit. Effective timestamps are caller-supplied or one engine
clock default. This is existing 13.3; implementation and compatibility updates
follow this merge closure.
