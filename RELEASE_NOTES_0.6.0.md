# Groggy 0.6.0 — release preparation

Status: unreleased. This document describes the proposed embedded-core release,
not a claim that every roadmap requirement or release gate is complete.

## Scope

Groggy combines a Rust graph engine, immutable graph/table analysis, temporal
history and numerical operations behind a Python API. This release consolidates
the implemented core and changes several defaults from the 0.5 series.

Both packaging entry points exclude macOS Finder metadata, and the artifact gate
rejects contaminated payloads.

Implemented capabilities, within their documented boundaries:

- Checked bulk mutations and automatic Python mutation publication, with one
  delta per accepted batch and explicit effective timestamps or a batch clock.
- Retained-state and effective-time reads, explicit checkpoints, tracked restore,
  branches and three-way merge machinery.
- Immutable graph selections, pinned queries and read indexes, table source
  identity and explicit property writeback.
- Consolidated matrix/numeric foundations, f64 properties, temporal value types,
  and expanded graph and numerical algorithms.
- Retained archives and journaled durable storage with explicit recovery,
  checkpoint, compaction and purge operations.
- Classic BFS, DFS, Dijkstra and A* now consume pending selected topology and
  values through shared native read frames, preserving one outer acceptance.
- Classic PageRank, closeness and betweenness also share the pending input
  boundary, including staged weights/personalization and directed closeness.
- All six classic community wrappers now use the same selected pending boundary,
  preserving component modes, seeded labels and existing graph-type restrictions.
- Sequential root loops now stage supported topology with refreshed selected reads,
  global allocation positions and one acceptance; fixed child topology stays guarded.
- Direct node/map/column and edge builder attachments now accept explicit effective
  timestamps in atomic persistent runs and reject unsupported parameters.
- Selected retained lifecycle/window helpers now follow surviving pending IDs
  after topology changes; provisional IDs remain guarded against invented history.
- Accepted snapshot acquisition and node/edge retained diffs now compose after
  pending topology while keeping unpublished working state separate from history.
- Fixed child frames validate every node/edge attachment target before staging,
  publication or transient output, including full-owner history diff maps.
- Pending child captures share one accepted base and prepared topology, copy only
  frame values and revalidate changed targets, reducing measured repeated allocation.
- Pending subgraph producers and certified for-each bodies now use exact child
  selections, shared nested parsing and execution, and immutable emitted sources.
- Long atomic runs reclaim expired weak capture metadata and oversized registry
  capacity while preserving live earlier table attachment authority.
- Temporal value decay now accepts pending selected nodes/edges and ordered
  columns, preserving explicit input identity and one atomic acceptance.
- Flow-map update/residual primitives now use checked integer/finite arithmetic
  and compose with pending topology without rebinding explicit map IDs.
- Filters/order, numerical maps, fused/community primitives and sampling masks
  now compose after pending topology through shared selected readers.
- Typed pipeline values, atomic property programs and selected topology effects.
  Certified loads, initialization, arithmetic and attachments can follow staged
  topology using pending selected IDs and the same one-delta acceptance.
  Structural, neighbor, expression-transform and path primitives now read that
  pending source, including staged weights and revised endpoints. Explicit
  captured-table result attachments preserve source row identity and optional
  per-property effective times in that same atomic run. Certified nested compute
  loops and message-passing blocks share pending reads and typed bindings.
  Local atomic programs can stage node/edge additions and run 13 existing
  algorithms on immutable pending captures. Supported outer pipelines now share
  topology effects and accept once; general topology loops, unadapted effects
  and remaining DSL/streaming work retain their checklist boundaries.

See the [flat checklist](notes/planning/graph-engine-flat-checklist.md) for exact
completion boundaries and the [core workflow](docs/api/core-release-workflow.md)
for an executable example across the supported surfaces.

## Migration from the 0.5 series

| Area | 0.6.0 direction | Application action |
| --- | --- | --- |
| Ordinary Python writes | Supported mutations publish checked changes automatically. | Remove commits used solely to make writes visible; retain intentional checkpoint labels. `Graph(manual=True)` is explicit compatibility. |
| `commit()` | An explicit checkpoint of an accepted state; pending manual writes have a documented compatibility path. | Do not assume each checkpoint creates another mutation version. |
| Selections | Default subgraph selections pin immutable state. | Refresh/reselect deliberately after a write. Connected compatibility is explicit. |
| Property floats | Ordinary Python floats and float vectors use f64. | Existing f32 values retain their original precision; widening cannot recover lost bits. |
| Restoration | `restore_state(..., mode="switch" or "tracked")` makes intent explicit. | Choose a reference switch or one new tracked restoration delta; supply a retained target state. |
| Time | Effective timestamps describe accepted changes; reads create no history. | Supply Unix microseconds explicitly when using domain time. Recorded cutoffs are optional explicit read inputs. |
| Persistence | Active export and recoverable retained storage have different scope. | Use documented retained archives/journal APIs for history. Do not treat active JSON as a full backup. |
| Indexes | Immutable, transient read objects. | Build/rebuild explicitly after changes or reopening; required unavailable indexes fail rather than silently scanning. |

Version metadata now derives from the binding crate, and `groggy.__version__`
reports the loaded extension version. Both Cargo crates must advance together.
The standard Maturin backend builds distributions without importing the extension
or regenerating stubs during the build. Stub updates are an explicit maintainer
step; checked-in stubs and the typing marker are included in the wheel.

The release targets CPython 3.8–3.12 (macOS arm64: 3.9–3.12). Both package
entry points now declare `>=3.8,<3.13`, matching the current PyO3 0.20.3 build
limit. The nested build no longer forces its experimental forward-compatibility
bypass. Supporting newer interpreters requires a separate dependency upgrade and
qualification; it is not implied by a successful local 3.9 build.

## Boundaries and open risks

- Live Graph and connected owner access remains creator-thread bound. Native
  finalization is now safe on any thread, including cyclic GC and release after
  the creator exits. Detached immutable GraphView remains the supported worker
  read interface; this does not add concurrent writable graph access. See
  [Python owner lifetimes](docs/api/python-owner-lifetimes.md), including the
  required migration for explicit native Rust handle types.
- A multi-client database service and arbitrary concurrent writers are outside
  this embedded release scope.
- General schema/format upgrade support, broad platform recovery evidence and
  large-history memory/latency work remain open. Do not assume arbitrary older
  retained files can be migrated, or claim power-loss safety on every platform.
- Exact vector retrieval, lexical ranking and hybrid retrieval remain planned.
  Storing a vector property does not imply an implemented search engine.
- Supported implementation is not a promise of numerical/graph parity across
  every operation, workload size or external library.

## Qualification

Use the current [release status](RELEASE_STATUS.md) for executed checks and open
gates. Local qualification does not certify the full platform matrix. No release
tag, upload or publication is implied by changing the development version.
Use the [release runbook](docs/api/release-qualification.md) for artifact-backed
tests, source rebuilds and the publication boundary.
