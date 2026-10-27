# Current topology integrity — 33.2 component


## Current topology-cache integrity — 2026-10-27 (33.2 component)

Explicit check_integrity now captures and verifies the current eligible GraphSpace
cache before archive reconstruction, which previously replaced it with fresh
derived data. Edge columns, active IDs/endpoints, isolate membership, neighbor
maps and CSR rows/offsets/edge IDs are checked with multiplicity. Cold/stale caches
are left untouched; invalid headers and poisoned locks return contextual errors.
The existing incidence factory is shared by cache construction and validation.
Sorting/comparison run in the detached phase; ordinary mutations do no extra work.

Four native regression groups cover valid empty/directed/undirected/parallel/loop/
isolate inputs, malformed columns/ranges/content/versions, detached captures,
stale/cold behavior and poisoning. Python checks cover warm caches, rejected
rewiring, successful rewiring and immutable old views. Final gates: 1,390 native /
56 ignored, 2,051 root Python / 1 skipped, 185 executable documentation examples,
strict all-target Clippy and rebuilt release FFI (three existing warnings).

Bounded release receipt: `../benchmarks/current-topology-integrity-2026-10-27.json`.
For the 100k-node one-version fixture, median cold capture/verify was 179/376 ms;
warm capture/verify was 177/422 ms. Warm verification additionally checks current
cached incidences. This measures explicit full retained integrity, not ordinary
write/query latency, RSS or a production SLA.

The current-cache component is complete. Existing 33.2 stays partial for secondary
index/other cache/service diagnostics, and AR2 still owns broader adjacency-policy
conformance. Receipt fields and storage formats are unchanged. Inventory remains
**208 checked / 60 open**, fixed 268 rows.

## Verified boundary

The plan shares immutable cached arrays and owns flat authoritative active IDs and
endpoints. It holds no graph/pool borrow. An eligible built version requires a
snapshot at the same version. Edge column extents and exact topology precede CSR
inspection. Node IDs match sorted active membership, including isolates. Offsets
start at zero, are monotone, have one terminal entry per row, and end at equal
neighbor/edge column lengths. Rows and neighbor maps match expected incidence
multisets; malformed ranges fail before slicing.

Only diagnostic invocation captures/checks caches. It publishes, repairs, prunes
and rebuilds nothing, preserves current schema/head/history and keeps existing
receipt fields. Capture rejects dirty heads before copying. A detached plan
checks the captured source independently of later owner mutation/cache replacement.

The legacy current-cache incidence convention is preserved, including two
undirected loop incidences. The factory is shared to prevent verification policy
drift; this is an integrity check rather than global algorithm conformance. It
does not certify secondary query indexes, Subgraph component caches, frozen lazy
caches or the currently unused optional cached adjacency matrix field.

Physical verify_storage sees reconstructed state because derived caches are not
persisted. check_integrity adds current in-memory cache inspection. The distinction
and native ownership/GIL behavior are documented in docs/api/retained-integrity.md.

## Next fixed priorities

- 09.4b.2 / AR3: complete compound topology publication and typed effects; certified
  property pipelines and unknown-effect rejection already have bounded evidence.
- 26.1 / AR4 / AR7.5: migrate ordinary connected Subgraph ownership to immutable
  state pins while retaining its full read/write/algorithm API. GraphView pins are
  complete within their documented supported surface.
- 18.2–18.3 / 21.1–21.2 / AR1–AR2: complete index/query lifecycle and caller
  conformance with explicit state/liveness/adjacency definitions.
- Keep numerical production gates, storage migration/platform work and service
  deployment in their existing scopes. Graphiti remains deferred behind the core.

No new checklist rows or owner-policy questions were introduced by this slice.
