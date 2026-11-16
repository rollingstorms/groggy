# Expanded 0.6.0 scope — 2026-11-15

## Owner decision

Include all remaining groups from the hierarchical 53-entry outline in 0.6.0,
except group 8 (deferred retrieval and external integration).

- 44 unchecked rows are now in the 0.6.0 target; overlapping parents remain counted.
- The nine deferred IDs are 20.1, 20.2, 22.1, 23.1, 23.2, 24.1, 25.1, 25.2 and 32.4.
- Provenance 17.1–17.2 and concurrency/service 09.5/33.3 are included in the expanded target.
- Preserve the fixed 268-row inventory. Update existing requirements and evidence.
- Previous qualification proves the bounded candidate at its recorded source;
  it does not certify the unfinished expanded scope. Repeat final gates after implementation.
- Open owner policy choices remain open. Performance/scale requirements need
  explicit, measurable exit criteria before broad guarantees can be closed.

## Implemented chunk: outer atomic topology execution

Local atomic builders stage additions, removals and endpoint revisions and expose
immutable pending captures to 13 existing graph kernels. The outer coordinator now
shares those effects across builders with one acceptance and exact final selection.
See `outer-atomic-topology-2026-11-15.md` for implementation and verification.

### Intended outcomes

1. Certify supported topology and captured-read effects before any prefix executes.
2. Share one pending mutation across multiple participating algorithms/builders;
   provisional IDs and selected topology remain consistent across stage boundaries.
3. Let supported later captures/computations observe earlier staged topology.
   Unadapted operations continue to reject explicitly.
4. Accept exactly once at the outer boundary. Errors, cancellation, stale ownership,
   schema failures and invalid timelines preserve graph/history/IDs and prior outputs.
5. Return the exact accepted selection, including added entities and excluding removals.
6. Cover directed/undirected, sparse IDs, parallel edges, loops, explicit clocks,
   failed late stages and successful history/storage round trips with native/Python tests.

Preserve existing public APIs and ordinary non-atomic execution semantics. General
callback effects, topology loops, journal-owner dispatch and remaining DSL lowering
retain separate exit criteria in their existing parent rows.

### Existing checklist coverage

09.4b.2, AR3, 30.1 and AR7.8; evidence also contributes to 05.3 and 01.2.
Completing this chunk does not automatically close every overlapping parent.

### Implemented follow-up: pending topology properties

Certified property loads, initialization, arithmetic and attachments now execute
against the pending selected state, including provisional IDs and surviving edge
instances. They reuse canonical validation and the existing one-batch publication.
See `pending-topology-properties-2026-11-15.md` for evidence and remaining boundaries.
Broad parents stay open; the 215 checked / 53 unchecked inventory is unchanged.

### Implemented follow-up: selected structural/neighbor/path reads

The existing structural, neighbor, expression-transformation and path primitives
now consume the pending selected source. Pure variable reductions are also
certified. See `pending-topology-readers-2026-11-15.md`. Arbitrary callbacks and
unadapted families retain their guards; broader parent exit criteria stay open.

### Implemented follow-up: captured table attachment

`table.attach` now stages explicitly selected/aliased result columns from captures
issued by the same atomic run. It reuses source table identity checks, weak capture
authority, canonical property validation and optional per-property effective times.
See `captured-table-attachment-2026-11-15.md`. Historical/imported tables are not
implicitly authorized by matching numeric IDs.

### Implemented follow-up: nested pending computation

Certified compute loops and message-passing blocks now use the shared native step
executor and selected CSR boundary. Staged property reloads, nested frames and
typed graph/table/snapshot bindings retain source identity. Current parent pins
can be borrowed by children without an extra copy. See
`nested-pending-execution-2026-11-15.md` for evidence and remaining guards.

### Implemented follow-up: core compute readers

Seventeen existing filtering/ordering, numerical map, fused, community and mask
sampling step types now compose after pending topology. Shared source validation
and selected CSR replace owner-only reads. See `pending-core-compute-2026-11-15.md`
for gates, benchmark and remaining subgraph-frame boundaries.

### Implemented follow-up: checked flow maps

The two existing flow primitives now consume explicit edge maps after pending
topology and reuse checked expression arithmetic. Overflow/nonfinite/domain
failures preserve the full atomic prefix. See `checked-flow-compute-2026-11-15.md`.

### Implemented follow-up: pending temporal decay

Value decay now respects selected pending IDs/endpoints and retained immutable
sources for node maps, ordered columns and edge maps. See
`pending-temporal-decay-2026-11-15.md`; retained-history temporal stages retain
their explicit guards. The note specifies the next root/child frame boundary.

### Implemented follow-up: capture authority lifecycle

Expired weak capture metadata/capacity is reclaimed with amortized work during
long runs, preserving live earlier source-bound tables. See
`capture-authority-churn-2026-11-15.md`, which also summarizes this priority pass.

### Implemented follow-up: exact subgraph frames

Native root/child selection ownership now supports pending iteration, neighborhoods,
emission and certified for-each bodies with immutable outputs and one acceptance.
Nested bodies share parsing and selected execution. See
`pending-subgraph-frames-2026-11-15.md`; its release benchmark identifies repeated
root copying as the next performance task.

### Next chunks

1. Pending-frame materialization cost, then general topology-mutating execution, public
   facade lowering and execution/DSL coverage.
2. Property/schema/provenance and persistence-boundary completion.
3. Archive/recovery memory, numerical/algorithm certification and scale targets.
4. Remaining API/module consolidation and service/concurrency implementation.
5. Full conformance, capability documentation and fresh release qualification.

Tests, failure injection and documentation accompany each implementation chunk.

### Implemented follow-up: projected frame materialization

Pending child captures now retain one accepted source, prepare topology by
structural revision, validate changed property targets incrementally and copy only
selected frame values. Topology changes still revalidate every staged target.
See `projected-frame-materialization-2026-11-15.md` for matched native runtime and
requested-allocation measurements, regression gates and remaining boundaries.
This advances existing execution/performance parents without adding rows.

### Implemented follow-up: fixed child attachments

Fixed for-each execution frames now constrain node/edge attachments, including
live property-overlay frames, transient results and nested loops. Full-owner
retained diff maps remain readable but cannot authorize a child to write sibling
IDs. Existing root explicit owner-ID semantics remain intact. See
`fixed-child-attachments-2026-11-15.md` for reproduction and qualification.

### Implemented follow-up: retained history inside pending programs

Explicit retained snapshots and node/edge diffs now compose after pending topology
through the shared immutable-input execution capability. Accepted history remains
separate from pending additions, removals, endpoints and values. Late invalid
history/attachment inputs retain whole-run rollback. See
`pending-retained-history-2026-11-15.md`. Selected lifecycle/window readers still
need the staged-entity history decision; general topology loops remain guarded.

### Implemented follow-up: selected retained temporal windows

All four lifecycle/window primitives now follow surviving pending membership for
accepted IDs, while reading only retained history. Nested/outer execution, source
validation and rollback share the common interpreter. Provisional IDs still reject
while the owner question remains open. See `pending-temporal-windows-2026-11-15.md`.

### Implemented follow-up: classic path wrappers

The four classic BFS/DFS/Dijkstra/A* handles now consume pending selected topology
and values through a shared native AlgorithmFrame. Private read pins stay separate
from the live continuation, and bulk endpoint reads share one authority/owner
borrow. Later builders see staged results and the coordinator publishes once.
The three centrality wrappers now also share the pending input boundary; see `pending-classic-centrality-2026-11-15.md`. The remaining six community wrappers are now adapted as well; see `pending-classic-communities-2026-11-16.md`. All thirteen classic wrappers share pending input and one acceptance; broader execution/lowering requirements remain open. See `pending-classic-paths-2026-11-15.md`.
