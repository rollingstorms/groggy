# Priority execution block — 2026-11-25

Follow-up: [the 2026-11-27 block](priority-execution-block-2026-11-27.md)
completes this block's edge-transform, collection/foreach and min/mean priorities,
plus community/decay/path read kernels. The priorities below retain their dated
historical scope; use the follow-up for current remaining work.

This block extends the existing engine's shared execution boundary and repairs
one demonstrated numerical discrepancy. It does not redefine the engine contract
or introduce a new roadmap. Seven finite slices are complete, with 42 component
checks recorded in their notes and 19 additional step implementations sharing
memory/durable execution.

## Completed slices

| Slice | Finished scope | Code commit |
| --- | --- | --- |
| [Aggregation](portable-aggregation-steps-2026-11-25.md) | Seven additional map-statistic kernels; all eight module steps portable | `a305dac5` |
| [Message blocks](portable-message-blocks-2026-11-25.md) | Supported message-pass blocks, selected pending reads and public builder route | `23399898` |
| [Flow and aliases](portable-flow-and-aliases-2026-11-25.md) | Three flow/residual/alias kernels; immutable sharing with copy-on-write mutation | `46fe0212` |
| [Expressions](portable-declarative-expressions-2026-11-25.md) | One evaluator through private read inputs; public Rust expression API preserved | `05e71ca4` |
| [Filtering/order](portable-filtering-steps-2026-11-25.md) | Four predicate/rank kernels; exact comparisons, stable ties and null/domain rules | `8c29687d` |
| [Sampling masks](portable-sampling-masks-2026-11-25.md) | Three selected seeded node/edge/reservoir mask kernels | `bd1c42ff` |
| [Normalization safety](normalization-safety-2026-11-25.md) | Invalid epsilon, NaN/reversed clip bounds and nonfinite arithmetic reject atomically | `6ec32ed1` |

These reuse existing algorithms and definitions. Immutable outputs stay unchanged
when working properties change; read computation alone does not retain a graph
version. Explicit attachments publish accepted computed changes. Workers receive
selected immutable reads and sealed kernels rather than mutable graph owners or
registry callbacks. Sparse selections, pending IDs, directedness, parallel edges,
self-loops, exact numeric domains, timestamps and complete failure rollback are
covered by the component evidence.

## Qualification

Latest local runtime `6ec32ed1959543036f638abedf0dbe23df21a6f7` passes **1,695 native
tests** (54 ignored), **3,597 packaged Python tests** (one skipped, 23 expected
warnings) and **302 documentation examples**. Strict all-target Clippy, formatting,
whitespace, Python 3.8 syntax and packaged Python/stub comparisons pass. All 500
production source files match the recorded runtime. See the
[local receipt](../benchmarks/normalization-safety-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/normalization-safety-wheel-qualification-2026-11-25.json).

The earlier [flow candidate CI](https://github.com/rollingstorms/groggy/actions/runs/45646459570)
passed all 21 required jobs, including 19 wheel combinations and source rebuilding.
Only Linux Python 3.12 runs the full maintained Python/docs suite; other wheel
jobs run smoke/application/ownership checks. That evidence is pinned to its
candidate and is not assigned to newer code. The [newer candidate run](https://github.com/rollingstorms/groggy/actions/runs/45652492406)
also passed all 21 required jobs on `ac59fcf1`, whose 500 production source files
match the local numerical safety runtime. Its [CI receipt](../benchmarks/normalization-safety-ci-qualification-2026-11-27.json)
was collected on November 27 with the reusable validation tool. Manual workflow dispatch cannot publish a package.

Matched repeated-alias observations improved 81–83% at 100 aliases for their
recorded source and fixtures; controls also varied. See the
[bounded alias receipt](../benchmarks/owned-alias-performance-2026-11-25.json).
The [matched normalization measurements](../benchmarks/normalization-performance-2026-11-25.json)
cover 24 valid-input public workloads with builds/tests stopped: at 10k rows and
ten invocations, four normalizers show +0.9–4.5% median changes; clipping and
controls also vary. These describe only their pinned source/artifacts. No allocator/RSS, arbitrary workload or production SLA
claim follows from either transient benchmark.

## Checklist interpretation and next work

The stable checklist remains **217 checked / 51 open** (42 core-target rows and
nine deferred rows). No broad row is closed merely because one of its components
is complete; no new stable row was added for this block. Remaining execution
work already belongs to 09.4b.2, 30.1 and AR7.8:

1. Complete remaining owner-only read primitives, starting with the edge-weight
   transform, with explicit numeric/error domains and selected pending reads.
2. Give subgraph-array producers and foreach composition a compatible immutable
   collection boundary; do not fabricate worker graph owners.
3. Prepare retained-history readers from explicit accepted history without
   inventing retained history for provisional IDs.
4. Finish edge-aware registers, normalize(min/mean), remaining frontend/alternate
   plans and fixed-child/provisional topology domains, with unsupported paths
   continuing to reject explicitly.

A14 naming/cache/run-history choices and A15 production budgets remain owner
questions. Numerical overflow requiring scaled normalization intermediates is
an explicit limitation within the existing numerical audit, not a newly added
roadmap row. Service concurrency, broader certification and representative scale
requirements remain in their existing parents.
