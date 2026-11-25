# Ordinary loop portability — 2026-11-24

Finite component of existing 09.4b.2, 30.1, 30.3 and AR7.8. The stable flat list
remains 268 rows: 217 checked and 51 open. This closes the ordinary `iter.loop`
composition limitation within the certified child inventory; broader execution
parents remain partial.

## Completed implementation

- [x] Prepare ordinary bodies recursively on the owner, capturing sealed child
  kernels and effect/identity metadata before worker execution.
- [x] Execute nested ordinary bodies and nested checked register plans through
  the shared selected-read executor. Preserve pending-property refresh and one
  outer staged mutation, including repeated creation and explicit timestamps.
- [x] Keep earlier graph/table captures immutable, fixed input selections scoped,
  and provisional allocation positions global to the outer batch.
- [x] Validate every child before durable execution, even when an enclosing loop
  has zero iterations. Effect declarations alone do not certify a kernel.
- [x] Preserve whole-operation rollback, cancellation, missing binding errors,
  sequential aliases, zero-iteration bindings and empty/read-only no-op history.
- [x] Preserve the owner-only memory fallback for unported ordinary bodies;
  supplied malformed or unported register plans never silently select a body.

## Architecture

Leaf steps retain borrowed native kernel references. A container can prepare an
owned sealed control plan borrowing its already-instantiated child kernels.
This avoids a self-referential cache and leaves existing public Step signatures
intact; the additional hidden preparation hook defaults to no capability.
Workers receive neither registry dispatch nor mutable graph owners. Nested
execution calls only prepared sealed kernels and uses the existing shared
executor's cancellation, effect and pending-read rules.

Loop binding initialization shares immutable variable payloads. Mutable access
continues through existing copy-on-write behavior. No new value kind or graph
versioning policy is introduced. Stats retain their existing invocation/iteration
count semantics.

Repeated writes to one key in an atomic builder retain the final staged value
and effective timestamp. Internal iterations do not each become retained
versions. Immutable captures can retain intermediate computed values. This
follows the existing batch contract.

## Evidence and boundaries

Native regressions: `tests/durable_loop_execution.rs` verifies actual worker
execution, owner-only factory and Step-hook guards, nested topology/property
refresh, selected-outside isolation, immutable earlier captures, exact reopen,
inner failure rollback, zero-iteration rejection/bindings, missing bindings,
cancellation and nesting a checked register plan inside an ordinary body.
Public Python regressions in `tests/test_durable_pipeline_execution.py` exercise
nested creation/property updates, output lifetime, recovery and unsupported
nested preflight. The executable example is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Final local qualification: **1,670 native tests pass** (54 ignored), **3,450
packaged Python tests pass** (one skipped, 14 established warnings), and **294
documentation examples pass**. Strict all-target Clippy, formatting, whitespace,
changed Python 3.8 syntax and all 62 packaged Python/stub byte comparisons pass.
All 500 production source files match runtime `1ac65a15fed0ca0919554e35e310f3d09202b46e`.
See [local receipt](../benchmarks/ordinary-loop-local-qualification-2026-11-24.json)
and [wheel receipt](../benchmarks/ordinary-loop-wheel-qualification-2026-11-24.json). Earlier platform receipts remain historical; this slice does
not claim new platform certification.


## Matched performance observations

[Raw samples and summary](../benchmarks/ordinary-loop-performance-2026-11-24.json)
compare the preceding qualified runtime (`fc70c592`) with this runtime. Three
alternating sequential processes per runtime supply nine observations per case.
Both use the unchanged `benches/checked_neighbors.py` independent expected-mode
oracle. Builds, tests and source-archive checks were finished before timing.
Setup, compilation and warm-up are excluded; execution and output marshalling
are included. This measures memory/transient execution, not durable sync costs.

| Route | Nodes | Iterations | Before ms | After ms | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1,000 | 10 | 2.794 | 2.738 | -2.0% |
| native | 10,000 | 10 | 26.875 | 28.369 | +5.6% |
| native | 1,000 | 100 | 23.395 | 24.032 | +2.7% |
| native | 10,000 | 100 | 259.013 | 262.049 | +1.2% |
| register | 1,000 | 10 | 3.033 | 3.028 | -0.2% |
| register | 10,000 | 10 | 35.360 | 32.539 | -8.0% |
| register | 1,000 | 100 | 26.051 | 26.144 | +0.4% |
| register | 10,000 | 100 | 285.879 | 283.469 | -0.8% |

Ordinary-body medians range from 2% faster to 6% slower, with overlapping sample
ranges; the register control also varies. No general speedup or performance
budget is claimed. Preparation adds a small owned control plan per loop
invocation, rather than a child metadata/registry lookup per iteration. The
existing checked-register adjacency/row-validation costs remain; this slice
does not resolve that separate performance work or establish allocator/RSS limits.

## Remaining work in the existing parents

Unported child kernels/history producers, other composition containers, fixed
child/provisional topology and alternate public IR/streaming are still separate
migration work. Register `LoadEdgeAttr` still needs an edge-aware result frame;
normalize(min/mean) does not map to the existing native normalization contract.
Selected adjacency construction and checked register overhead remain in the
performance scope. A14 naming/cache/history and A15 production budgets remain
unanswered. No new owner decision was needed for this finite component.
