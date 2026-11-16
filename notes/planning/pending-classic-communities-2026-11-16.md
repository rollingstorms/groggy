# Pending classic community wrappers — 2026-11-16

## Completed six-wrapper family

Connected components, LPA, Infomap, Louvain, Leiden and Girvan–Newman now use the
same AlgorithmFrame boundary as the four classic path and three centrality
wrappers. All thirteen built-in classic wrappers can follow supported topology
builders in an outer atomic pipeline and consume pending selected IDs, endpoints,
direction and property overlays. Results feed subsequent stages and publish once.
The continuation remains distinct from the private pending read source.

The six existing kernels and their options are unchanged. Component modes retain
weak/undirected versus strong reachability. Their CSR, assignment and grouped-edge
caches use selected endpoints and remain private to the pending scope. LPA keeps
typed seed labels and deterministic/seeded iteration. Infomap preserves direction
and trial/seed behavior. Louvain/Leiden remain undirected and unweighted, preserving
parallel instances and loops into their multilevel modularity inputs. Girvan–Newman
preserves every candidate edge and its optional strictly positive length. Local
candidate removals do not become graph mutations. The private Louvain persistence
helper borrows the scope and returns unit; it no longer owns a continuation.

## Verification

- Native classic integration target covers all thirteen wrappers after warming
  old live CSR, staging provisional node/edges, revising an endpoint, cascading a
  node removal and staging weights. Later builders read results, outside rows stay
  unchanged and there is one publication.
- 52 new Python cases compare pending computation with separately accepted
  selected execution across direction, sparse IDs, parallel instances, loops,
  seeded labels, component modes, weighted Girvan–Newman, empty inputs, late
  failures and unsupported directed domains. One-way routes independently require
  three strong components versus one weak/undirected component.
- Focused path/centrality/community/classic pipeline suite: **131 passed**.
- Full locked/offline native: **1,604 passed / 54 ignored / zero failed**, 223
  suite results. Strict all-target Clippy, Rust formatting and whitespace pass.
- Rebuilt in-tree release extension: **3,183 maintained Python passed / one
  skipped / 14 established warnings**, **285 executable documentation examples**.
  CPU-sensitive Python gate ran after native builds and workload measurements.
  Fresh artifact/platform qualification remains a separate release gate.

The initial native assertion assumed the public integer label was stored as the
Int enum variant. Canonical storage can compact it to SmallInt. The corrected
assertion uses the established AttrValue::as_int logical integer accessor; output
values and later copies must still agree. The benchmark accepts the same compact
integer representation. No storage policy or production algorithm was changed.

## Bounded current workload receipt

`notes/benchmarks/pending-classic-analysis-2026-11-16.json`: 36 native release cases,
three observations each, 100/1,000 initial undirected chain edges. One classic
centrality/community handle follows staged weights=2 and publishes once. Pending
cases replace the last node with a provisional neighbor of node 0 and revise the
last edge into a ring. Property-only cases retain the chain. These differ in shape
and answers and are **not matched before/after performance comparisons**.

At 1,000 edges, medians in milliseconds:

| Algorithm | Property-only chain | Pending ring |
| --- | ---: | ---: |
| PageRank | 2.488 | 5.087 |
| Closeness | 53.244 | 53.510 |
| Betweenness | 68.878 | 67.675 |
| LPA | 2.676 | 4.738 |
| Infomap | 267.132 | 284.261 |
| Connected components | 2.934 | 5.714 |
| Louvain | 3.519 | 5.480 |
| Leiden | 3.576 | 5.740 |
| Girvan–Newman | 21.957 | 19.999 |

Controls: max_iter=10, seed=7, Infomap num_trials=2, Girvan–Newman num_levels=1;
other defaults retained. Timings include preparation, preflight, computation and
publication, exclude fixtures/capture/construction/assertions/compilation. Finite
output checksums are diagnostics, not independent correctness references. No
allocation/RSS, FFI, convergence-quality or general scale guarantee. Reproduce:
atomic_builder_properties `--classic-pending-analysis` with
GROGGY_ATOMIC_BUILDER_PROFILE selecting a new receipt path.

## Existing checklist and open work

Advances 09.4b.2, 03.2/AR2, 30.1 and AR7.8, conformance 01.2/19.2/28.16 and measured
analytics 32.3. This finishes the fixed remaining classic-wrapper inventory; there
are no remaining built-in classic wrappers blocked merely by the old owner-only
adjacency source. The broader parents retain their separate exit criteria:
general topology-mutating loops, unsupported custom effects, alternate plans/DSL,
durable dispatch, streaming and wider scale/release qualification. Provisional
retained-history lifecycle still awaits the owner answer and remains guarded.
Wide pending captures still copy selected logical attributes.

No new checkbox rows or broad parent closures: **215 checked / 53 unchecked**,
**44 open release rows / nine deferred**. Unrelated owner files are untouched.
