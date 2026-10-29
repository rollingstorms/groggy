# Scoped traversal trees, paths and cache validity — 26.1 / AR7.5 / 21.2

Completed the captured traversal read dependency and repaired the connected
selected-scope paths. GraphView supports BFS/DFS discovery trees, reachability,
shortest hop subgraphs and explicit weak/strong connectivity. One native kernel
consumes either immutable GraphView adjacency or a selected live topology reader.
The connected compatibility path copies no properties, graph or retained history.

## Scope and ownership repairs

Connected BFS/DFS/shortest paths previously traversed the whole owner and filtered
output afterward; a route could leave and re-enter a selected subgraph. Inducing
nodes could resurrect excluded owner edges. The connected-components fallback
used owner edges between selected nodes even when those instances were excluded.
All are now restricted during computation to selected membership. Components
reuse the existing BFS component kernel on selected CSR and preserve caching.

BFS/DFS return exact discovery trees, with ascending neighbor/edge ties and one
selected edge per discovered node. Loops and parallel alternatives do not become
extra tree edges. DFS now excludes queued back/cross edges from its result.
Shortest hop paths ignore weights explicitly; unreachable/out-of-scope returns
None. Captured results share state/time/group/write provenance through later
owner changes/release. Connected results retain their established live ownership.
Python captured computation releases the GIL. Its depth counts are checked.
Python connected empty is_connected now agrees with native/captured false.

## Measured compatibility regression fixed before commit

The first implementation called pin_view for every connected traversal. A 100k-
node three-hop read measured a 37.500 ms median despite an existing immutable
reader. After releasing the last reader, it recaptured properties and took
190.769 ms. This approach was replaced before commit.

The final topology-only live adapter shares the same tree/path kernel as frozen
reads. Complete selection validation is cached at the graph-space instance and
mutation version; no logical state or payload is pinned by that identity key.
The live reader borrows the existing topology snapshot and sorts only visited
selected rows. It never constructs a fake partial GraphView or a writable clone.

`benches/captured_view_traversal.rs`, raw provisional/repaired release receipts:
`notes/benchmarks/captured-view-traversal-2026-10-29.json`. Five samples over
1k/100k undirected chains, full selections, max_depth=3. The full-owner engine is
only a timing reference: it cannot substitute for selected-scope traversal.

| 100k case | Capturing adapter | Final topology adapter |
|---|---:|---:|
| Captured BFS median | 3.875 µs | 1.000 µs |
| Connected BFS median | 37,499.666 µs | 3.375 µs |
| Connected after last read pin released | 190,769.042 µs | 20.125 µs |

The repaired connected first call still costs 31.966 ms for complete selection
validation and topology preparation; later calls reuse validity/topology. The
repaired whole-owner reference median is 0.875 µs. Scope checks, deterministic row
ordering and compatibility result assembly have real overhead; no general
speedup, scalar-FFI budget or million-node throughput guarantee is claimed.
Each variant is one local release process and load/noise differ; the structural
allocation change, not fine timing differences, is the supported conclusion.

## Restoration must invalidate caches even if numeric versions repeat

Restoration replaces GraphSpace and can recreate its previous numeric version.
A numeric-only cache key can therefore reuse old CSR or component results after
endpoint rewiring/restoration. Connected scope validation, CSR and component
caches now use an instance-sensitive SpaceVersion: one identity marker per space
plus its ordinary mutation version. This marker retains no graph data/history.
A native regression deliberately restores two different topologies with equal
numeric space versions and verifies neighbors, components and paths refresh.
Python covers the same owner/retained-state boundary. Storage formats and public
numeric version inspection remain unchanged.

## Verification and fixed remainder

Six native cases cover every three-node directed edge set against independent
reachability/hop-distance calculations, depth and actual-edge ties, exact scope,
selected component fallback/cache, source release/workers and equal-version
restoration. Python cases additionally exercise connected compatibility, strict
depth types and empty/weak/strong definitions. The priority-window log records
final full native/Python/docs/Clippy gates and rebuilt in-tree extension.

The original connected Subgraph and its columns/accessors remain live. Captured
algebra/grouping, all thirteen classic algorithms and these tree/path operations
are complete read prerequisites; nonregistry structural summaries and the full
read/write/default-return migration remain in existing 26.1/AR4/AR7.5. Compound
topology/custom effects remain in 09.4b.2. No new checklist rows or parent closure:
211 checked / 57 unchecked.
