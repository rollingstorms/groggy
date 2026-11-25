# Checked topology plans — 2026-11-24

Continuation of existing execution rows 09.4b.2, 30.1 and AR7.8/AR7.11. This
slice migrates existing implementations onto read authority and shared checked
execution; it does not create a second graph engine or new roadmap rows.

## Finite component inventory

- [x] One read-only topology facade supplies graph kind, selected membership,
  stable IDs, selected endpoints and sorted adjacency to memory and detached
  workers. The prior owner adapter delegates; no writable owner is exposed.
- [x] Move the existing degree, neighbor aggregation/collection, asynchronous
  neighbor mode and in-place map update implementations once. Expression and
  community callers reuse the same graph-default neighbor construction.
- [x] Move the three existing fused kernels once, preserving native f64 arithmetic;
  reject nonfinite inputs/results. Accept the Python frontend's fused spellings
  while preserving the older serialized spellings.
- [x] Lower nine further instruction forms through known sealed native kernels:
  degree, collection, aggregate, mode, update-in-place, neighbor-mode-update,
  fused neighbor multiplication/reduction, MADD and AXPY. Total: **35/36 forms**.
  Directional mode composes collection/list mode; non-sum fused neighbor reduction
  composes multiplication/aggregation. No new math interpreter is introduced.
- [x] Preserve exact integer modal ordering, deterministic first ties,
  parallel-edge/self-loop incidence and asynchronous update visibility.
  Reject incomparable ordered modal ties and nonfinite lists before installation.
- [x] Add independent native incidence oracles for **112 combinations** across
  graph kinds, all directions/reductions/tie policies, self inclusion, both update
  ordering flags, sparse IDs, aliased destinations and parallel edges/self-loops.
  Compare both backend plans, ordinary native steps where available, and recovery.
- [x] Add focused pending-endpoint/removal, retained-view/clock, large-integer mode,
  native empty-neighborhood and failed-overflow rollback checks. Keep unsupported
  preflight coverage by testing a genuinely unported weighted-degree step.
- [x] Repair the Python expression compiler's missing named-register binding and
  per-node constant representation. Never omit an unlowerable expression; preserve
  graph-default neighbor expressions through the ordinary native path instead of
  silently hardcoding incoming direction. The prior weak execution-only regression
  now checks the two-iteration numerical result. Add chained arithmetic, constant
  and identity expression regressions across two iterations.
- [x] Add a bounded integer-mode specialization to the same native kernel, with
  exact signed-limit labels, frequency/lowest/highest/first policies and comparisons
  against the general mixed-label path. Align private selected-ID indexing with
  the existing immutable algorithm reader; preserve all source/authority checks.
- [x] Qualify full native and rebuilt packaged Python surfaces; update the pinned
  capability receipt and commit/push this coherent component.

Implemented in `17d2f0eaae7828fb6e1e1f575953c754aec3a756`; the measured follow-up is
`fc70c592865e8fc6490b4fb9b72904580fd13d55`. All finite component checks are complete.

## Qualification

- **1,664 native tests passed**, 54 ignored; 227 suite results include ten passing
  doctests. The checked-plan integration suite has 13 passing cases, including
  the 112-combination incidence matrix and signed-limit fast/general mode parity.
- **3,447 packaged Python tests passed**, one skipped, fourteen established
  warnings; **293 documentation examples** pass. All 62 packaged Python/stub files
  match and the changed Python/benchmark/test files parse with Python 3.8 grammar.
- Strict all-target native Clippy, formatting and diff checks pass. The existing
  three binding-build warnings remain; this slice does not claim a new FFI lint gate.
- All 500 production source files match the final committed Git archive byte for
  byte. Source SHA-256:
  `54e573227670d204dc6bfb69958d87f7b29a6f0d24273ab6a16690f5310d29dd`.

[Final local qualification](../benchmarks/integer-mode-local-qualification-2026-11-24.json)
and [fresh-wheel qualification](../benchmarks/integer-mode-wheel-qualification-2026-11-24.json)
retain source/artifact identities and gate-log hashes. The
[initial migration qualification](../benchmarks/checked-topology-local-qualification-2026-11-24.json)
retains its earlier exact source identity. Earlier platform-wide qualification
remains historical; no fresh cross-platform release claim is made here.

## Contracts and costs

[Public loop contracts](../../docs/api/checked-numeric-loop-plans.md) distinguish
explicit stored-edge directions from graph-default outgoing/symmetric traversal.
Self-loops count once; explicit self inclusion is additive. Asynchronous updates
use stable node ID order for both legacy ordering-flag values. Read computations
create no history; explicit attachments stage one accepted outer batch.

Fused arithmetic still narrows integers to f64 and uses ordinary floating-point
accumulation. This slice adds failure checks, not an exact-integer fused contract.
Numeric register stores still reject null/text/list rows. Native empty-neighbor
min/max/mode remain null-capable intermediates with explicit native attachment.

The common selected CSR is rebuilt and sorted per checked neighbor invocation.
Collection previously used a separate unordered CSR cache. Sharing removes that
semantic split but can increase repeated collection cost. This change does not
claim general neighbor acceleration. Map results/async target copies and composed
non-sum fused operations still allocate. The measured follow-up below avoids
integer JSON counting keys and aligns the private ID index with `AlgorithmRead`.
Broader topology cache reuse and throughput work remain under existing 32.2/AR7.8;
no new row or production latency/memory promise is introduced.

## Matched neighbor performance

[Benchmark source](../../benches/checked_neighbors.py) runs a directed ring with
parallel edges/self-loops, index-valued labels, repeated collection/lowest mode
and transient attachment from a pinned selection. Every result matches an
independent integer incidence/frequency oracle; graph history stays unchanged.
Three separate processes provide three timed observations each: **nine samples
per case per runtime**. Setup, compilation and one warm-up are excluded; output
marshalling is included. Builds and test gates ended before sampling. CPU power,
OS cache and unrelated background activity were uncontrolled.

Median milliseconds:

| Path | Nodes | Iterations | Prior `0ff24177` | Migrated `17d2f0ea` | Final `fc70c592` | Final vs prior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 1,000 | 10 | 3.144 | 3.770 | 2.871 | -8.7% |
| register | 1,000 | 10 | 2.249 | 3.972 | 3.060 | +36.0% |
| native | 10,000 | 10 | 37.206 | 40.411 | 34.351 | -7.7% |
| register | 10,000 | 10 | 19.612 | 43.524 | 30.137 | +53.7% |
| native | 1,000 | 100 | 25.383 | 35.110 | 24.132 | -4.9% |
| register | 1,000 | 100 | 14.569 | 36.458 | 26.461 | +81.6% |
| native | 10,000 | 100 | 296.946 | 367.799 | 269.144 | -9.4% |
| register | 10,000 | 100 | 144.492 | 389.209 | 299.450 | +107.2% |

The follow-up reduces migration-only medians **15–31% for native workflows** and
**23–31% for register workflows**. Native workflow medians are **5–9% below the
prior baseline**. Checked register execution remains **36–107% slower than the
prior float-only interpreter** on this fixture. The earlier interpreter narrows
and discretizes labels and ignores explicit direction; answers match here for
small integer labels and outgoing traversal. This is not a claim of equivalence
for other types, directions or empty neighborhoods.

The registered workflow still pays for checked map bindings and rebuilding
sorted selected adjacency per invocation; the earlier register interpreter
cached graph-default neighbor vectors once. Those costs remain a concrete
performance priority in existing 32.2/AR7.8. No allocator/RSS, accepted-delta,
fsync or production-scale budget is measured by this benchmark.

[Aggregate and raw-file hashes](../benchmarks/checked-neighbors-2026-11-24-summary.json)
retain all selected samples, ranges and source identities. One intermediate run
overlapping the qualification source audit is explicitly excluded. A one-process
index-only pilot is retained separately and is not used as evidence of a speedup.

## Validation finding

The first full packaged run found one failure in a pre-existing structured
neighbor-expression test. Its generated plan stored an expression temporary in
one register but read an unassigned named-output register next. Iteration one
used initialized zeros; iteration two reused that register after it had become a
scalar. The old interpreter had masked the error with implicit scalar/vector
behavior. Inspection also found the expression compiler hardcoded incoming
neighbors where native expressions use graph-default outgoing/symmetric neighbors.

The frontend now binds named registers, represents per-node constants as maps,
and returns no compiled plan when it cannot preserve an expression. The ordinary
expression implementation remains authoritative in that case. The strengthened
undirected recurrence is **1 → 1.85 → 3.295**, rather than merely checking an
attribute exists. Explicit-direction register instructions remain supported.

## Remaining existing work

- `load_edge_attr` is the sole unlowered opcode form. Its legacy executor produces
  a dummy scalar register while writing an unrelated edge-map variable; the
  numeric result facade cannot honestly certify that as an edge-aware register.
  Ordinary native edge-attribute loading remains supported in durable programs.
- `normalize(min/mean)` options remain unlowered; native normalization has no
  corresponding semantics to reuse. These must not silently select another method.
- Ordinary step-body composition, other read/history kernels, alternate IR and
  streaming frontends remain in their existing execution rows.
- A14 naming/cache/run-history policy and A15 production budgets remain unanswered.

The broad inventory remains **217 checked / 51 open**. Nine previously unported
instruction families are completed within existing parents, without implying those
larger parents are complete. Earlier platform qualification remains historical.
