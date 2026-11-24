# Checked batch-plan lowering — 2026-11-24

Continuation of existing checklist rows 30.1, 09.4b.2, AR3 and AR7.8. This
component does not close those broader rows or create additional roadmap rows.

## Completion boundary for this slice

- Share the graph-independent arithmetic and normalization implementations across
  memory and detached durable execution, without writable owners in kernels.
- Lower the supported numeric batch instructions to those native kernels; preserve
  exact integer input values, isolate registers from caller variable names, and
  define carried-slot copying from the iteration-end state.
- Reject malformed supplied plans, including zero-iteration plans, rather than
  silently running a different body. Keep unported instruction families explicitly
  uncertified for atomic/durable execution.
- Verify ordinary and lowered execution, staged reads, aliasing, cancellation,
  publication rollback and journal reopening; measure the checked execution path.
- Retain the public builder facade. JIT and pipeline naming/cache policy are outside
  this slice. Remaining neighbor/in-place/fused instruction migrations stay under
  the existing parent rows.

## Completed implementation

- [x] One implementation of binary/unary arithmetic, reciprocal, comparison,
  conditional selection, scalar reduction/broadcast, list mode, and four
  normalization wrappers over the read-only native scope.
- [x] Lower 26 instruction forms to owned sealed kernels and already-captured
  metadata; reuse one nested `StepExecutor` and its pending-read cache.
- [x] Isolate registers from caller names, preserve loaded integer types, permit
  aliased destinations, and copy carried assignments simultaneously. Reject
  duplicate carried destinations.
- [x] Reject malformed supplied plans, invalid slots/controls, and negative loop
  counts, including the zero-iteration boundary. A malformed plan never silently
  falls back to a different body. Valid plan-only loops work.
- [x] Reuse the exact numeric comparator from filtering/ordering in ordinary,
  lowered and streaming comparison. Adjacent integers above 2^53 and tiny
  nonzero float differences remain distinct; equality has no implicit tolerance.
- [x] Preserve the direct public `BatchExecutor` facade and add explicit Context
  execution. Retain typed registers after execution; optional JIT pointer
  conversion cannot reject valid list intermediates or change exported values.
- [x] Verify node/edge counts, broadcast/where, defaults and pending creation,
  selected sparse IDs, parallel/self-loop fixtures, one-node execution, detached
  worker preparation, clocks, cancellation, rollback and journal reopening.

The native parity matrix compares 25 operator/control combinations against
ordinary native steps for both directed and undirected fixtures. Separate tests
exercise the public facade, list intermediates and slot carries. Python covers
the raw compatibility frontend and ordinary public `AlgorithmBuilder` arithmetic
with `atomic=True, optimize=False`.

## Remaining existing scope

Ten instruction families remain uncertified: edge-attribute register loads,
node degree, in-place update, neighbor mode update, neighbor collection,
neighbor aggregation/mode, and the three fused forms. `normalize(min/mean)` is
also outside this lowering. Their non-atomic compatibility interpreter remains;
no atomic/durable support is inferred from an effect flag. Ordinary native loop
bodies, public IR routes, custom/streaming effects, scoped/provisional history
and the unanswered A14/A15 policies remain under their original parent rows.
The fixed checklist remains 217 checked / 51 open, with no added roadmap rows.

## Qualification and performance

Runtime `4fa2be08991b61d4fdaf1b5f0082cf78af39d23d` is committed. All 500 production source files
match that Git archive byte for byte; the local source-tree SHA-256 is
`d1ce52d61d7ab6f8eeaaca168ebd952242fcab4e2355b99eb513afe316a91797`.

- Full native gate: **1,654 passed, 54 ignored**, 227 suite results including ten
  passing doctests. The seven numeric integration tests all pass.
- Strict all-target core Clippy, root formatting and diff checks pass. No new
  dependencies or FFI business logic were added.
- Fresh release wheel: **3,432 maintained Python tests passed, one skipped**, with
  fourteen established warnings; **293 executable documentation examples** pass.
  All 23 new numeric Python cases pass, including the public builder and lists.
- All 62 packaged Python/stub sources match; Python 3.8 syntax checks pass.
- [Local receipt](../benchmarks/checked-batch-local-qualification-2026-11-24.json)
  and [wheel receipt](../benchmarks/checked-batch-final-wheel-qualification-2026-11-24.json)
  pin the artifact, source and executed gate logs.

Earlier platform qualification of `b24e86e0` remains historical. This changed
runtime is qualified locally; no fresh nineteen-platform claim is made.

### Measured cost — existing row 32.2

`cargo bench --offline --locked --bench checked_batch` compares the same f64
recurrence through an ordinary checked step loop and the lowered frontend. Three
independent processes provide seven observations per case each, after warming
both lazy caches. Graph/journal setup and compilation are excluded; memory times
include final atomic publication. Durable preparation and synced acceptance are
timed separately. Samples were collected after local builds and test gates ended.
Every computed answer matches.

Median milliseconds across 21 observations per case:

| Nodes | Iterations | Native step loop | Lowered loop | Durable prepare | Durable accept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.018 | 0.024 | 0.023 | 4.040 |
| 1,000 | 10 | 2.135 | 3.998 | 3.651 | 6.749 |
| 10,000 | 10 | 20.737 | 37.782 | 35.431 | 23.394 |
| 1 | 100 | 0.069 | 0.115 | 0.122 | 4.858 |
| 1,000 | 100 | 7.914 | 24.627 | 24.664 | 6.763 |
| 10,000 | 100 | 85.200 | 250.611 | 248.524 | 24.954 |

The lowered frontend costs **1.30–3.11×** the equivalent checked native memory
loop in these cases. Register initialization and load/store map copying are
visible in the implementation; their individual contributions have not been
isolated. Profile and reduce those copies under the already-open performance row
32.2 before presenting this frontend as acceleration. This is a semantic
consolidation result with a measured cost, not a performance closeout.

[Aggregate and raw sample identities](../benchmarks/checked-batch-2026-11-24-summary.json)
retain all observations, ranges and hashes. OS cache, power state and unrelated
background activity were uncontrolled. This is not a comparison with the old
vector interpreter, a native-allocation/RSS measurement, or a production SLA.
No new budget or roadmap row is introduced.
