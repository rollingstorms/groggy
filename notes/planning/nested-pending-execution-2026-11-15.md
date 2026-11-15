# Nested pending computation and one selected execution boundary — 2026-11-15

## Completed implementation

Top-level builder steps and interpreted loop children now share native
StepExecutor. It selects the appropriate source, refreshes after staged changes,
checks cancellation before/after a child and retains original execution
membership for final acceptance. It does not publish or create a second mutation
coordinator. Nested compute loops can load/write/reload properties against the
same shared batch and include provisional entity IDs.

Recursive effect composition permits certified pending property/reader bodies,
explicit immutable inputs and persistent staged capture/attachment bodies.
Topology mutations, alternate batch plans, streaming and opaque callbacks remain
uncertified. A legacy owner-reading descendant does not gain pending-topology
capabilities merely by being wrapped in a loop. Checks apply at zero iterations.

Typed loop bindings clone the complete existing StepValue rather than a manually
maintained subset; graph views, tables, snapshots and temporal indexes preserve
source identity. Capture authority still requires issuance by this mutation run.
Explicit immutable inputs retain their supplied values, while scope property
loads refresh.

Message-passing blocks and neighbor aggregation now share checked directional
selected CSR preparation. Pool endpoint reads are removed from message blocks.
Ordered Gauss–Seidel and synchronous Jacobi modes retain their visibility rules,
selected IDs, loop/parallel conventions and private result installation.

Weak capture authority records also carry a pending revision for read reuse.
Child frames can borrow an already-current parent pin; writes invalidate that
reuse. No graph payload is retained by authority metadata or cached permanently.

## Verification

Full locked/offline native suite: **1,588 passed / 54 ignored / zero failed**,
223 suite results including doctests. Strict all-target Clippy, formatting and
whitespace checks passed. Rebuilt in-tree release extension: **2,883 maintained
Python tests passed / one skipped / 14 established warnings**, plus **275
executable docs examples**. The focused suite passed 328 tests. The three
established binding compiler warnings are unchanged. No wheel was installed.
Native commands used locked/offline dependencies, the in-tree target and disabled
incremental compilation; Python/docs used the configured Python 3.9 interpreter
and `notes/scripts/verify_immutable_view_migration.py`. Black/isort/pre-commit remain
unavailable. Unrelated owner review/benchmark work is untouched.

The native release benchmark uses five observations per shape, with one staged
new node and three identical property load/increment/attach operations:

| Initial nodes | Flat | Loop | Nested loops |
| --- | --- | --- | --- |
| 1,000 | 5.022 ms | 3.573 ms | 3.368 ms |
| 10,000 | 41.718 ms | 36.709 ms | 31.362 ms |

These are medians from one process, not a statistical speedup claim. Fixture,
initial capture, algorithm construction, compilation and assertions are excluded;
read refresh, frames, staging and one acceptance are included. No FFI, memory or
general overhead guarantee is made. Raw observations:
`notes/benchmarks/nested-pending-execution-2026-11-15.json`. Reproduce with the
atomic_builder_properties bench's `--nested-pending` mode and
`GROGGY_ATOMIC_BUILDER_PROFILE` selecting its output path.
Coverage includes nested read-after-write, provisional IDs, endpoint revisions,
cascades, typed aliases, captured-table attachment/effective times, both message
visibility modes, explicit historical inputs, zero-iteration guards and late
nested failure rollback. Existing exhaustive kernel references remain maintained.

## Checklist and remaining scope

This advances existing **09.4b.2, 30.1 and AR7.8** without adding rows or closing
broad parents prematurely. Inventory stays **215 checked / 53 unchecked**, with
**44 unchecked release rows**.

General topology-mutating loops, unadapted sampling/community families, alternate
execution plans/JIT/streaming, remaining public facade/DSL lowering, journal-owner
dispatch and naming/history policy remain open under their existing requirements.
Runtime capture reuse is an execution-frame optimization, not a new global
result-cache or pipeline-history policy.

Contract/example: [nested pending execution](../../docs/api/nested-pending-execution.md).
