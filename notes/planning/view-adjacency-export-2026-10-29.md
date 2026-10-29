# Selected adjacency export — 2026-10-29

Existing 26.1 / AR2 / AR4 / AR7.5 read-facade component. No new checklist row.

## Boundary and implementation

Six pre-fix Python cases reproduced the graph placeholder, missing captured export,
and connected direction/ordering failures. The connected FFI implementation scanned
all selected edges separately for each node, read directed edges as undirected and
silently skipped endpoint-read errors. It had O(V*E) endpoint work.

Graph, connected Subgraph and captured GraphView now expose selected outgoing
adjacency with ascending node keys and neighbor IDs. Isolates have empty rows;
parallel occurrences and one loop occurrence remain. Excluded edge instances are
never reintroduced. Export sorting is explicit; neighbors() retains its existing
occurrence order, with the same selected multiset.

Rust AdjacencyRead owns shared node IDs and CSR only: no writable owner, property
payload or history pin. Connected preparation validates scope before cache reuse
and shares its existing topology cache. Captured preparation uses the existing
AlgorithmRead builder, then drops its property/source/index handles. Native read
objects remain usable after owner mutation/destruction and on workers. Output
ordered maps are independent owned buffers. Graph and connected FFI prepare on
the owner thread, then release the GIL for export; captured preparation/export
also release it. Python conversion still holds the GIL.

The change removes the quadratic edge scan; building the ordered map and sorting
neighbor rows still have costs. It introduces no new snapshot/storage format,
entity allocation, mutation/publication, implicit index or writable owner clone.
Current connected defaults/effectful migration remain open.

## Verification

Three native cases cover both directions, parallels/loops/isolates, empty inputs,
selected exclusion, warmed/cloned caches, invalid endpoint closure, detached/frozen
workers after owner deletion, removals and retained restoration. Nine Python cases
cover all three public forms, returned-dictionary independence and worker reads.
Full gate: **1,501 native / 54 ignored**, strict all-target core Clippy,
**2,554 root Python / 1 skipped**, **248 executed docs examples**. Full-ABI release
extension rebuilt and atomically staged; three existing unrelated warnings remain.
Scoped Rust format/Python syntax/diff checks pass; owner work remains untouched.

## Bounded release receipts

Python connected undirected chain, five warmed bound-method calls, dictionary/list
conversion included; setup/warmup and analytic checks excluded. Separate quiet
before/after builds, not alternating pairs. Old unordered rows are compared as
multisets and the O(V*E) baseline stops at 5k:

| Nodes | Former scan | Selected CSR export |
| --- | --- | --- |
| 1,000 | 14.906 ms | 0.120 ms |
| 5,000 | 382.161 ms | 0.778 ms |
| 100,000 | not run | 29.443 ms |

Separate native directed-chain preparation/export receipt: at 100k, cold connected
preparation **24.495 ms**, warm preparation **2.208 µs**, captured CSR preparation
**17.522 ms**, detached ordered-map export **2.152 ms**. Initial frozen owner
capture is measured separately. No extrapolated old 100k time, universal FFI/RSS
budget, whole-engine throughput or constant-time cold/export claim.

Raw: `../benchmarks/view-adjacency-python-2026-10-29.json` and
`../benchmarks/view-adjacency-native-2026-10-29.json`.
API: `../../docs/api/view-adjacency-export.md`.
**211 checked / 57 open**; no broad parent closure.
