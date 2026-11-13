# Captured structural degree tables — 2026-11-12

Existing scope: 30.1 / AR7.8 and read-after-topology work in 09.4b.2. Completed
one explicit captured structural computation family; broad parents remain open.

## Implemented

`graph.degrees_table(source=..., output=...)` produces the existing immutable
Table intermediate from an explicit GraphView. It uses the captured view's native
scalar degree methods, not the writable owner. Original sparse IDs sort into
`node_id`, followed by `degree`, `in_degree`, `out_degree`. Incident instances
include incoming directed edges, each loop once; directional counts use their
existing GraphView definitions. Parallel selected edges count separately.

No node properties are copied into the summary; a stored property named degree
is neither read nor overwritten. Counts and structural IDs convert to table
integers with overflow checks. Source IDs and graph provenance are retained.
Empty shape is (0,4); Python result edits leave source values unchanged.

The step shares the existing interpreter, immutable-input certification and
native table result wrapper. It can run after staged endpoint revisions/removals,
using distinct before/after captures, and can read history whose nodes are no
longer active in the owner. All eight local/outer atomic and persistence mode
combinations work with explicit retained inputs, without creating history.
Late type/projection failures reuse atomic staging/output rollback.

## Performance and ownership

Small inputs use sequential column construction. At 4,096 rows and above,
independent row counts fill three disjoint owned column buffers through the
existing Rayon pool. Only the immutable GraphView, cancellation token and owned
cells cross workers; Context, Subgraph and writable Graph stay on their thread.
There is no additional intermediate row-result buffer or permanent cache.
Cancellation is checked per result node and around the stage. One large
adjacency row is not interruptible, and the whole pipeline still holds its
existing owner-thread/GIL boundary.

Five-sample native release medians on directed paths with one additional parallel
edge and one loop (source capture, fixture, scope/variable setup, compilation,
assertions and result destruction excluded):

| Degree-table stage | 1,000 nodes | 100,000 nodes |
| --- | ---: | ---: |
| Initial sequential implementation | 0.331 ms | 114.056 ms |
| Initial parallel fill with all-size preinitialization | 0.511 ms | 15.325 ms |
| Final: sequential small buffers / parallel large buffers | 0.352 ms | 18.957 ms |

The final 100k case is about 6x faster than the initial implementation. The small
case varies slightly; this is not a universal parity/speedup guarantee. Raw
samples are preserved in `../benchmarks/captured-pipeline-degrees-2026-11-12.json`,
`captured-pipeline-degrees-parallel-2026-11-12.json` and
`captured-pipeline-degrees-final-2026-11-12.json` in that same benchmark directory.
This isolates optional degree-table work; no FFI, allocation/RSS, dense/narrow
view workload or production scale claim. Selected ID ordering and traversing
source incidence rows remain costs. No implicit degree/capture work was added to
ordinary graph mutations.

## Verification

- Final full Rust suite: **1,561 passed / 54 ignored**; strict all-target Clippy.
- Rebuilt/staged in-tree release; three existing FFI compile warnings remain.
- Final quiet maintained Python gate: **2,676 passed / 1 skipped / 15 warnings**;
  **266 executable documentation examples** pass.
- Native coverage checks exact selected edges, sparse IDs, isolates, loops,
  directed/undirected counts, cancellation, owner destruction/worker reads and
  5,000 sparse result rows with independent count expectations in both directions.
- Fourteen Python cases cover historical liveness independence, pending revised
  topology, exact edge subsets, empty shape, source independence, late errors
  and local/outer/persistence execution modes.

An earlier broad Python run overlapped benchmark compilation and failed the
existing 96x96 Cholesky worker-progress timing assertion. Quiet full reruns passed
without numerical/GIL code or test changes. This is observed mixed-load timing
sensitivity, not proof of a matrix implementation regression or definitive cause.
The known PyGraph cyclic-GC warning remains under 19.2. Black/isort are unavailable
in the active environment; Python edits are manually formatted, Rust changes use
scoped rustfmt and whitespace checks.

## Remaining scope

Addition stages need safe references to unallocated entities. Broader algorithms
over staged topology, outer/nested topology execution, DSL/IR lowering, naming/
cache policy and streaming remain in existing 09.4b.2/30.1/AR7.8. No new workbook
policy was inferred. Contract: `docs/api/captured-pipeline-degrees.md`.
**215 checked / 53 open / 268 total**, no new rows or broad-parent completion.
