# Reclaim expired pending-capture authority — 2026-11-15

## Completed bounded maintenance

Pending capture authority already used weak graph/group/capture references, so it
retained no snapshot payload. Expired entries could nevertheless accumulate once
per read revision in a long atomic run. PropertyBatch now sweeps expired records
at an amortized threshold and shrinks an oversized registry after reclamation.
The next threshold grows with live authority rather than rescanning every live
capture on every step. Short runs keep the small existing fast path.

Earlier live captures remain authorized, including tables whose standalone view
was dropped. This is metadata reclamation for dead captures, not a budget or
retention policy for caller-owned immutable objects. Read revisions, identity,
property clocks and one-batch publication are unchanged. No public API or storage
format changes.

Native stress tests create thousands of captures, assert expired-only churn stays
bounded, grow/drop a set of live captures to exercise capacity reclamation, and
then attach an earlier live table through one successful acceptance. Matching
numeric IDs remain insufficient authority.

## Verification and performance

Final locked/offline native suite: **1,594 passed / 54 ignored / zero failed**,
223 suite results including doctests. The focused ten-test PropertyBatch suite
also passes with the high-water capacity case. Strict all-target Clippy, Rust
formatting and whitespace checks pass. Rebuilt the in-tree release extension
without wheel installation; three established binding warnings remain.
**2,952 maintained Python tests pass / one skipped / 14 established warnings**,
plus **278 executable documentation examples**. Black/isort/pre-commit remain
unavailable.

| Initial nodes | Flat | Loop | Nested loops |
| --- | --- | --- | --- |
| 10 | 10.649 ms | 8.413 ms | 7.772 ms |
| 100 | 71.715 ms | 64.981 ms | 59.433 ms |

These are medians of the current native release observations.

The benchmark reuses the existing flat/loop/nested fixture runner with 1,000
identical load/increment/attach iterations, five observations for each shape and
size, one staged node and one final publication. It excludes fixture/initial
capture, compilation, algorithm construction and assertions. Raw receipt:
`notes/benchmarks/capture-authority-churn-2026-11-15.json`; reproduce through
atomic_builder_properties `--capture-churn` with GROGGY_ATOMIC_BUILDER_PROFILE.
Timings are current-workload observations; there is no before/after speedup, FFI,
RSS or general memory guarantee. The native resource tests prove the registry
reclamation behavior; timings alone do not measure it.

## Existing checklist and next priority

Supports 09.4b.2, 30.1, 32.3 and AR7.8. No rows added or broad parents closed:
**215 checked / 53 unchecked**, including **44 unchecked release rows**.

The next implementation chunk remains exact root/child selection refresh for
subgraph-producing sampling and for-each. Its required boundary is recorded in
`pending-temporal-decay-2026-11-15.md`; general topology loops and unadapted effects
remain guarded. Remaining history/DSL/service/release requirements retain their
original scope.

## Completed priority pass

| Commit | Qualified implementation slice |
| --- | --- |
| 51242690 | Selected structural, neighbor, expression and path readers after pending topology. |
| 029d2035 | Explicit source-bound table result attachment with aliases and effective times. |
| 0ad713ba | Shared nested step execution, typed loop bindings and selected message passing. |
| e15aeb76 | Seventeen filter/order, numerical, fused/community and sampling-mask step types. |
| d2ab0425 | Checked flow arithmetic and pending explicit-map execution. |
| 4b35e014 | Selected pending temporal decay for maps and ordered columns. |
| This slice | Amortized expired authority metadata/capacity reclamation. |

Each implementation slice has native/Python qualification, documentation and a
bounded release benchmark receipt. No expanded-release completion is claimed.
Unrelated owner review/benchmark work is untouched.
