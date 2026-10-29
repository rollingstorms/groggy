# Captured community computations — AR7.5 / 26.1 / 28.16

Completed the missing immutable input surface for Louvain, Leiden, Infomap,
Girvan–Newman and label propagation. Together with the preceding traversal,
centrality and component work, all thirteen classic registry algorithms now have
canonical captured read computation. This is a dependency of the connected-view
migration, not completion of that migration.

## Implementation and ownership

- Reuse existing community kernels; connected input preparation, staged property
  reads and checked attachment retain their existing paths. No full Graph clone,
  writable snapshot owner or retained-history clone is introduced.
- AlgorithmRead retains its existing frozen GraphView, sorted IDs, mapping and
  CSR. Community methods prepare specialized kernel topology as needed. Repeated
  native callers can reuse AlgorithmRead; convenience methods prepare per call.
- Preserve selected edge instances, sparse node IDs, isolates, loops, direction,
  existing seeded behavior and native algorithm budgets. Louvain/Leiden/Girvan–
  Newman reject directed inputs; Infomap/LPA preserve directed semantics.
- Girvan–Newman reads frozen positive lengths; excluded bad edges do not reject
  a selected problem. LPA reads frozen arbitrary seed values. Context supplies
  cancellation and telemetry, never staged values or publication, even when
  persist_results is true. Results are ascending-ID owned rows.
- Python wrappers translate/validate types and release the GIL through input
  preparation/computation; arbitrary AttrValue labels translate after computation.
  Default controls and bounded two-level Infomap limits remain explicit in
  `docs/api/captured-community.md`.

## Verification

Five new native regressions cover exact connected/captured kernel parity,
nonpublication, seed-value preservation, owner mutation/drop, worker transfer,
empty/isolate inputs, cancellation, direction and invalid controls/selected
weights. Seventy-nine Python cases add count/seed/domain validation, exact edge
selection, concurrent readers and effective-time corrections through durable
close/reopen. Existing classic atomic pipeline tests preserve staged label/weight
reads and one checked acceptance on the connected path.

Full gates are recorded in the priority-window log after completion.

## Release preparation and computation receipt

`benches/captured_communities.rs`, raw five-sample interleaved release results:
`notes/benchmarks/captured-communities-2026-10-29.json`. Fixture: five-node cliques
connected by bridges, one isolate, no properties/history. Connected calls disable
publication; view calls include CSR preparation; prepared calls reuse native
AlgorithmRead. Capture is measured separately. Single-process local measurements
compare ownership paths in the same implementation, not old/new versions or
production guarantees.

| Algorithm | Nodes | Connected median ms | View median ms | Prepared median ms |
|---|---:|---:|---:|---:|
| Louvain | 3,000 | 3.441 | 3.027 | 2.301 |
| Leiden | 3,000 | 5.741 | 5.166 | 4.650 |
| Infomap, 3 trials | 3,000 | 1,122.701 | 1,034.357 | 1,024.534 |
| Girvan–Newman, 4 removals | 100 | 0.536 | 0.517 | 0.508 |
| LPA | 3,000 | 2.404 | 1.766 | 1.073 |

The 3,000-node CSR preparations cost 0.468–0.526 ms in these independent cases.
At 300 nodes, Infomap's view median is 11.342 ms versus connected 11.176 ms,
about 1.5% slower; the smaller cost difference does not establish equivalence or
an all-size performance improvement. Specialized community topology remains
O(V+E) additional preparation; Infomap search and Girvan–Newman removal costs
remain those documented for their existing kernels.

## Fixed remaining boundary

26.1/AR4/AR7.5: connected Subgraph and builder accessors still read their live
owner, and cannot be declared immutable merely because GraphView is complete.
21.2: legacy filter return ownership remains to migrate. 09.4b.2: topology/custom
multi-stage acceptance remains separate. 28.09 hierarchical Infomap and the rest
of 28.16's scale/reference/options certification remain open. No new checklist
rows or broad parent closures; fixed inventory remains 211 checked / 57 open.
