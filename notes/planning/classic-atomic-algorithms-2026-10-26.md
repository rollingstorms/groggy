# Classic algorithms in shared property transactions

Third slice in the priority window starting 2026-10-26 11:18:59 UTC.
Existing 09.4b.2 scope, no added checklist rows: **201 checked / 67 open**.

## Completed

All 13 built-in classic registry implementations now certify explicit outer
property transactions: PageRank/closeness/betweenness; Louvain/Leiden/Girvan–Newman/
LPA/connected components/Infomap; A*/BFS/DFS/Dijkstra. These algorithms retain the
same owner and have no topology mutations. Girvan–Newman removes candidate edges
only in its private computation, not the owner graph.

Moved the existing canonical StepScope property read/attachment implementation
to Context; StepScope delegates to it, and classic algorithms share it. Persistent
outer writes stage rather than publish. Source markers, A* heuristics, PageRank
personalization, LPA seeds and weighted kernels see earlier staged columns. The
minimum-pair weight validator and A* heuristic validator borrow overlay values
without copying columns; per-edge betweenness/Girvan–Newman paths also consult it
so parallel edges use the same weights. Existing standalone/owned transient output
and manual-mode behavior remains. No whole-graph copy or second staging store.

Native contract tests exercise all 13 families for one final publication, selected
scope, builder reads of classic output, frozen views, later managed-property failure
rollback and Context reuse. Staged invalid weights cover all five weighted families;
marker/personalization/heuristic rejection and positive LPA seed reads cover
additional property inputs. Python tests exercise every classic handle through the
FFI in successful and failing mixed-builder runs. Existing cancellation/owner/head
checks continue in the shared outer pipeline.

## Still open

The remaining 09.4b.2 component is now **topology compound acceptance**. Unsupported
custom algorithms still reject before prefix execution; supporting arbitrary
custom effects needs an explicit capability contract. Topology operations need a
prepared topology overlay plus identity/remapping/read semantics before they can
share one acceptance. Existing checked individual topology mutations are not a
whole-pipeline transaction. Do not close 09.4b.2 or its parents yet.

Broader effect-aware Python optimization/default execution policy stays AR7.8/30.1.
No new weighted algorithm modes, implicit maintenance, Graphiti integration or
checkpoint schedule were introduced.

Final gate totals are in priority-slices-2026-10-26.md.
