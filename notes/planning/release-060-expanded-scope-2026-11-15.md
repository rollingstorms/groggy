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

## Next chunk: outer atomic topology execution

Existing local atomic builders stage additions, removals and endpoint revisions,
and expose immutable pending captures to 13 existing graph kernels. The outer
Pipeline coordinator and inherited builder checks still certify property-only
execution. Finish that boundary using the existing staged mutation machinery.

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

### Following chunks

1. Remaining property operations after topology and execution/DSL coverage.
2. Property/schema/provenance and persistence-boundary completion.
3. Archive/recovery memory, numerical/algorithm certification and scale targets.
4. Remaining API/module consolidation and service/concurrency implementation.
5. Full conformance, capability documentation and fresh release qualification.

Tests, failure injection and documentation accompany each implementation chunk.
