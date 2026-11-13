# Existing-edge endpoint stages — 2026-11-12

Existing scopes: 09.4b.2 / AR3, 30.1 / AR7.8. Completed bounded existing-edge
revision stage; additions and broader computation/execution scopes remain open.

`graph.set_edge_endpoints(revisions=[[edge, source, target], ...], effective_at=...)`
uses the existing native step interpreter and checked GraphBatch publication.
Rows must reference selected active existing edges and selected active endpoints.
Strict row/control decoding happens before execution; whole-target validation
happens before staging. No allocation or owner mutation occurs in the stage.

Certified prefix properties, multiple endpoint rows and removals accept as one
delta. Duplicate edge revisions (within/across stages), explicit edge removal,
removed destination nodes, inactive/outside targets and invalid effective endpoint
timelines reject the whole plan. IDs, properties and external identity survive
revision. Empty rows publish nothing. Existing phase, cancellation, stale-head,
outer property-only and output rollback controls are reused.

Pending graph materialization now applies endpoint revisions before cascades,
matching canonical batch ordering. An edge moved off a removed endpoint survives
with its original properties. This holds regardless of staging order. Earlier
provisional captures remain immutable; a later revision can rescue an edge absent
from an earlier capture. Explicit removal and revision still conflict. New
adjacency and edge table projections use the revised pairs; retained views and
effective-time reconstruction preserve old pairs at the correct timestamps.

Verification: **1,558 Rust tests passed / 54 ignored**, strict all-target Clippy,
rebuilt/staged release, **2,662 maintained Python tests passed / 1 skipped /
15 warnings**, **265 executable documentation examples**. Three existing FFI
warnings and the known writable Graph cyclic-GC warning remain. Native tests
cover both directions, cascade rescue, clocks, duplicates, exact scope and
late failure. Python also covers both staging orders, effective-time history,
loops/parallels, typed captures and original property rollback.

Performance receipt: native release `atomic_builder_properties --endpoint-revisions`,
five samples on 1k/100k-node directed paths. Includes revision, node removal,
pending graph materialization/emission, checked acceptance and surviving selection.
Excludes fixture/initial capture/compilation/assertions. Medians: 1.146 ms at 1,000 nodes, 220.863 ms at 100,000 nodes. Raw samples:
`../benchmarks/pipeline-endpoint-revisions-2026-11-12.json`. No speedup, FFI,
allocation/RSS or billion-node claim; selected-state copies remain explicit costs.

Public contract: `docs/api/pipeline-endpoint-revisions.md`.
No rows added; **215 checked / 53 open / 268 total**.
