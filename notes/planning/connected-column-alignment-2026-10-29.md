# Native connected column alignment — 26.1 / AR7.5

Completed the native legacy column gap identified during captured accessor work.
Subgraph complete node/edge columns now align with ascending public node_ids/
edge_ids, retain absent rows as typed Null projections, and preserve each caller-
supplied ID position/repetition for explicit node sequences. Empty inputs return
empty vectors. Missing properties remain absent in graph storage; no mutation,
publication, graph capture or property/history copy is introduced.

The former native helpers iterated HashSets and skipped missing properties. That
could detach values from their row identity. They now delegate to the existing
attribute-first Graph.get_nodes_attrs/get_edges_attrs reader, resolving the
storage column once. Requested scope/liveness validates before copying values,
so inactive/out-of-scope/unknown IDs still reject rather than silently becoming
null. Corrupt indices continue to propagate the canonical reader's errors.
Native sorted ID getters reuse their existing cached order; first ordering still
costs a sort. Connected ownership remains live. The explicit GraphView accessor
keeps its frozen values and separate presence bits.

Python connected accessor indexing/order/absent-column behavior is not changed
by these native helper repairs. The in-tree extension is still rebuilt and the
complete root Python surface rechecked, since it links this Rust core.

## Verification

Four native integration cases cover sparse IDs, sorted complete node/edge
columns, exact typed f64/vector/large integers, explicit null and absent rows,
requested repeated/reversed IDs, rejected scope/inactive/unknown IDs, empty input,
nonpublication and live owner updates versus an older captured column.
Compilable examples: `tests/connected_column_alignment.rs`; guide:
`docs/api/connected-column-alignment.md`. No existing test was weakened or removed.

Release receipt: `notes/benchmarks/connected-column-alignment-2026-10-29.json`,
`benches/connected_column_alignment.rs`. Five alternating samples compare complete
aligned bulk columns with equivalent per-ID scalar reads on 1k/100k manual
isolates with typed integer/null/absent values. All resulting row vectors match.
Setup and ID ordering are excluded, as are Python/FFI and capture. Local medians:

| Nodes | Bulk column ms | Equivalent scalar column ms |
|---:|---:|---:|
| 1,000 | 0.094 | 0.105 |
| 100,000 | 19.106 | 20.053 |

This is a current-cost comparison, not an old/new speedup or general performance
claim. The main completed behavior is correct row alignment and one shared bulk
storage path. Full gates are recorded in the priority-window log.

Inventory remains **211 checked / 57 unchecked**, with no new checklist rows or
broad parent closure. Actual connected/default/filter return migration,
mutable/effectful facades and specialized view/viz ownership remain in existing
26.1/AR4/AR7.5/21.2; topology/custom compound effects remain in 09.4b.2.
