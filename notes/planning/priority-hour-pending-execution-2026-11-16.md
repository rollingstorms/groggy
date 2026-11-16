# Pending execution priority hour — 2026-11-16

Requested continuation began 2026-11-15 23:55 UTC, with a one-hour window through
2026-11-16 00:55 UTC. Branch: groggy-roadmap-2026-10-20. Production changes are
committed and pushed in four coherent slices; artifact/measurement closeout follows.

| Commit | Completed scope |
| --- | --- |
| f4e588c8 | PageRank, closeness and betweenness consume pending selected geometry and values through the common frame. |
| ffae633e | All six classic community wrappers share that boundary, preserving component modes, labels, direction restrictions and edge-instance semantics. |
| 797e6db4 | Root/nested-root topology loops use the existing executor/batch with refreshed selection, global allocation positions and cancellation/rollback. |
| be6512d1 | Direct node-map/node-column/edge attachments honor explicit effective clocks and reject unknown parameters. |

The fixed thirteen-classic-wrapper inventory is complete. This extends existing
parent scopes rather than adding checkbox rows. Ordinary source suites progress
from 1,604 to **1,610 native tests**, and from 3,089 to **3,273 maintained Python
tests** (186 new cases, two obsolete root-loop rejection cases retired). The
curated documentation gate grows from 283 to **288 examples**. Counts remain
**268 rows: 215 checked / 53 unchecked**, with **44 open release rows / nine deferred**.
The broad parents retain additional requirements; this is not a completion percentage.

## Final local qualification

Native implementation be6512d1, Rust 1.96.1: locked/offline native tests pass with
54 ignored and zero failures (225 suite results); strict all-target Clippy, Rust
formatting and whitespace checks pass. The separate serial killed-checkpoint
process recovery test passes on macOS. No external package or wheel was installed.
Three established binding warnings and 14 established Python warnings remain;
Black/isort/pre-commit are unavailable in the existing environment.

Fresh macOS arm64 CPython 3.9 wheel:

- SHA256 f6963a21814f79ed0900b3433e731b12eaf0b704aa24641fda68697eb23cb1c5.
- Isolated -I -S smoke, 105 declared exports/version agreement, archive and durable
  journal application workflows pass.
- Against the unpacked artifact: **3,273 maintained Python passed / one skipped /
  14 established warnings**, **288 executable documentation examples**.
- Receipt: `../benchmarks/priority-hour-final-wheel-qualification-2026-11-16.json`.
  The earlier same-wheel receipt records 287 examples before the final capability
  guide; its hash is identical. Tests ran after CPU builds/benchmarks stopped.

The capability guide `docs/api/atomic-pipeline-capabilities.md` includes all thirteen
classic wrappers sequentially in one pending-topology application and publication.
It records input ownership, time/retained-history distinctions and remaining guards.
After be6512d1 only benchmark/docs/evidence change; production sources remain pinned
by the artifact and CI source. No package/tag publication or PR was created.

## Matched root-loop execution receipt

Three fresh native release processes, five observations each, alternating loop
and unrolled order. The same program grows one isolated node per iteration and
increments all currently selected counters. Every final row, ID/count, unchanged
prior immutable input and one publication are asserted in both modes.

Combined fifteen-observation medians, milliseconds:

| Initial nodes | Iterations | Unrolled | Root loop |
| ---: | ---: | ---: | ---: |
| 100 | 10 | 1.115 | 1.117 |
| 100 | 100 | 11.900 | 11.637 |
| 1,000 | 10 | 9.885 | 9.835 |
| 1,000 | 100 | 86.754 | 87.165 |

Observed differences range from about -2.2% to +0.5%; no general speed guarantee
or statistical significance claim. Includes execution/preflight/capture/staging/
publication; excludes fixtures, initial capture, definition/algorithm construction,
assertions and compilation. No allocation/RSS/FFI measurement. Raw receipts:
`../benchmarks/root-topology-loops-2026-11-16-{1,2,3}.json`. Reproduce with the
atomic_builder_properties executable's `--root-topology-loops` and
GROGGY_ATOMIC_BUILDER_PROFILE selecting a new receipt path. Post-harness strict
all-target Clippy and fmt/whitespace checks pass.

The preceding centrality/community timing corpus has its own differing-graph
limits in `pending-classic-communities-2026-11-16.md` and the raw 36-case receipt.
Do not compare that chain/ring corpus as a matched performance experiment.

## Cross-platform qualification

Manual artifact qualification started at 00:28 UTC, pinned to be6512d1:
https://github.com/rollingstorms/groggy/actions/runs/43780585321

The configured matrix builds 19 wheels: Python 3.8–3.12 on Linux/Windows/macOS
Intel and 3.9–3.12 on macOS arm64. It also runs native/serial-recovery/Clippy and
extracted-source rebuild gates. Its publish job is excluded for manual runs.
Status/results are updated separately when completion is confirmed; workflow
configuration or partial success is not proof that all jobs passed.

## Open work retained from the checklist

1. Remaining execution/coordination scope: fixed child/provisional topology,
   alternate batch plans and facade lowering, streaming/unsupported custom effects,
   and durable owner dispatch. Root sequential loops and all thirteen classic
   wrappers are complete within documented canonical policies.
2. Provisional nodes in retained lifecycle/window queries: owner question remains
   pending; rejection is preserved. No accepted history is invented.
3. Wide pending snapshots still copy selected logical properties; broader payload,
   history-depth, allocation/RSS, numerical and analytics scale budgets remain open.
4. Existing provenance, concurrency/service, schema/history/storage and remaining
   conformance/retirement requirements retain their expanded release scope.

Emerging direct-attachment clock handling was found by a concrete loop regression
and fixed in be6512d1. Outside-lifetime property behavior follows the pre-existing
canonical contract; no new policy was introduced. Owner development/benchmark and
untracked review/notebook files remain untouched.
