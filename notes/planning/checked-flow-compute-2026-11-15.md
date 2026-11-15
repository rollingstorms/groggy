# Checked flow computation — 2026-11-15

## Completed slice

The two existing native flow primitives now consume explicit edge maps after
pending topology, including nested compute frames and outer atomic builders.
Their effects are immutable-input computation; normal pending initialization and
loads determine membership. Old maps are not rebound to new IDs.

Unchecked i64 addition/subtraction could panic in debug builds or wrap in release
builds. Both primitives now reuse checked native expression arithmetic, preserving
exact in-range integer results and existing mixed f64 arithmetic. They reject
integer overflow, nonnumeric/nonfinite operands and nonfinite results. Untouched
union rows and flow-only residual rows are validated; scalar deltas are validated
even for empty maps. Positive subnormals remain valid. Per-row cancellation and
private result installation preserve prior outputs on failure.

Existing sparse semantics remain: flow updates use map-key union or scalar updates
to supplied flow keys; residuals retain positive capacity-minus-flow and positive
flow-only values. No reverse edges or feasibility constraints are invented. API
documentation now states that distinction explicitly.

## Verification

Locked/offline full native suite: **1,592 passed / 54 ignored / zero failed**,
223 suite results. Strict all-target Clippy passed. In-tree release extension
rebuilt with the three established binding warnings; no wheel installation.
Focused Python: **190 passed / 13 established warnings**. Three new native tests
cover overflow, nonfinite values, invalid passthrough/empty-input paths, exact large
integers, sparse-key preservation and cancellation without replacing prior output.
Twenty-three new Python cases cover local/outer/nested pending execution, failed
publication prefixes, exact integers beyond 2^53, float result overflow, positive
subnormals and unchanged explicit old-map identities.

Full maintained qualification: **2,941 Python tests passed / one skipped /
14 established warnings**, plus **277 executable documentation examples**.
Rust formatting and whitespace checks passed.

Benchmark: five native release observations each, directed ring with one staged
node/two edges, one compute loop with initialization, checked flow update,
residual computation, two edge attachments and one publication. Medians:
**1,000 initial nodes: 6.313 ms; 10,000: 52.578 ms**. Fixture, initial capture,
compilation, algorithm construction and assertions excluded; pending execution,
checks and acceptance included. Bounded current-workload evidence only; no
before/after speedup, FFI, memory or scale guarantee. Raw receipt:
`notes/benchmarks/checked-flow-compute-2026-11-15.json`. Reproduce with the
atomic_builder_properties bench's `--checked-flow` mode and
GROGGY_ATOMIC_BUILDER_PROFILE selecting its output path. It shares the existing
pending-core fixture runner rather than duplicating another benchmark engine.

## Existing checklist

Advances 09.4b.2, 28.15–28.16, 30.1 and AR7.8; the numerical failure evidence also
supports 29.3. No rows added or broad parents closed: **215 checked / 53 unchecked**,
including **44 unchecked release rows**. General subgraph frames/topology loops,
remaining step families and facade lowering retain their existing requirements.
Broader arithmetic APIs are not redefined by this fix. Unrelated owner work stays
untouched. Black/isort/pre-commit remain unavailable.
