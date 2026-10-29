# Shared structural summaries — 26.1 / AR7.5 / 28.15–28.16

Completed the next captured-read parity component and four existing Python
Graph/Subgraph placeholders: local/mean clustering and transitivity now delegate
to the same Rust kernels as immutable GraphView. Captured all-node coefficient
rows provide an owned f64 bulk result in ascending ID order. Density is shared by
native inherent/trait, captured and Python adapters with the correct direction.

## Definitions and implementation

- Clustering/transitivity use exact selected **simple, loop-free undirected**
  topology. Parallel mass and loops do not create neighbor pairs or triangles;
  excluded owner edges cannot contribute. Isolates contribute zero to the mean.
  Empty/zero-denominator summaries are zero. Directed definitions reject
  explicitly, consistent with the existing triangle/k-core step domain.
- Density counts all selected edge instances, including loops/parallel edges,
  against the directed or undirected loop-free denominator. It can exceed one;
  fewer than two nodes returns zero. Counts convert before multiplication, so
  usize arithmetic cannot overflow in the denominator.
- The existing builder triangle step and summaries use one checked, cancellation-
  aware oriented-neighbor intersection kernel. The builder moves its existing
  rows into that kernel rather than cloning full adjacency again. Existing
  builder publication, controls and k-core multiedge behavior remain unchanged.
- Native UndirectedStructure is an immutable topology-only computation input,
  sharing sorted ID/CSR Arcs. Connected preparation validates membership and
  reuses its instance/version-aware CSR cache. AlgorithmRead/GraphView prepare
  the same input from frozen topology. It retains no properties, history,
  writable owner or GraphView and is Send+Sync. No fake partial graph is built.
- Python Graph/Subgraph prepare topology under the GIL, then release it for
  computation; captured convenience methods release it through preparation too.
  Local IDs reject booleans, nonintegers, negatives, overflow and excluded IDs.
  Native preparation supports multiple computations without rebuilding CSR.

The former native clustering code took neighbors from the owner and could count
an excluded triangle edge; multiplicity and loops also distorted its denominator.
Its transitivity path scanned triples and leaked owner adjacency. Both now share
correct selected simple projection. Python placeholders no longer incorrectly
claim the native capability is absent. Connected density's previous undirected-
only formula now agrees with the already direction-aware trait contract.

## Evidence

Four native integration cases check all 64 four-node simple graphs against
independent closed-triple combinatorics, selected wedges with excluded edges,
loops/multiedges/isolates, prepared owner release/workers, empty/invalid IDs and
directed domains. Two native unit cases cover cancellation, malformed rows and
usize-scale density arithmetic. Thirty Python cases cover Graph, connected and
captured APIs, strict IDs, immutable lifetime, selected scope and CSR invalidation.
Existing builder triangle and cancellation tests retain the shared-kernel gate.
Guide and executable examples: `docs/api/structural-summaries.md`.

`benches/structural_summaries.rs`, five local release samples, setup reported
separately; raw receipt `notes/benchmarks/structural-summaries-2026-10-29.json`.

| Fixture | Nodes | First connected setup ms | All coefficients median ms | Transitivity median ms | Local node 0 median ms |
|---|---:|---:|---:|---:|---:|
| Chain | 100,000 | 25.780 | 4.310 | 4.179 | 0.004 |
| Star | 100,000 | 22.037 | 3.773 | 3.875 | 4.841 |
| Clique | 300 | 3.984 | 10.204 | 10.362 | 0.718 |

Repeated native connected preparation medians are 1.084–1.417 µs in those cases.
These measure the supported new paths, not a matched before/after speedup, Python
scalar-call budget, logical-heap or billion-node guarantee. Cold immutable capture
and captured CSR setup are separate fields in the raw receipt. Global summaries
allocate projected topology; dense triangle work still scales with intersections.
Local summaries allocate neighbor/row storage, not all neighbor-pair combinations.

Final full native/Python/docs/Clippy gates are recorded in the priority-window log.
No new checklist rows or full parent closure: **211 checked / 57 unchecked**.
The connected Subgraph/read accessors remain live. Full default ownership,
column/accessor migration, compound topology/custom effects and broader analytics
certification remain in their existing scopes.
