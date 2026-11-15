# Pending selected structural, neighbor and path reads — 2026-11-15

## Completed scope

The following existing native primitives now execute against the pending selected
source after topology, in local atomic builders or shared outer atomic runs:

- Five structural primitives: degree, weighted degree, incident weight sum,
  triangle count and k-core marking.
- Neighbor collection and explicit-direction neighbor aggregation.
- Expression mapping, asynchronous neighbor-mode update and checked in-place
  variable-map updates. Arbitrary mapping callbacks remain uncertified.
- Shortest-distance maps, loopless k-shortest paths and seeded random walks.
- Seven variable-map reductions: reduce, standard deviation, median, mode,
  quantile, entropy and histogram. These require no graph capture.

## Shared design

Subgraph now owns a private selected-read validation/direction boundary: live
selections validate the owner; immutable captures retain their own membership,
endpoints and direction. Algorithms use captured endpoint readers rather than
retained writable-owner pool records. Collection reuses the checked CSR boundary;
structural, neighbor, transformation and path helpers share selected identity.

The existing pending scope revision cache and canonical property attachment
boundary remain authoritative. No parallel algorithm engine, writable graph
clone, extra publication or permanent capture pin is introduced.

Existing definitions are preserved: directed degree is outgoing, undirected
incident degree counts a self-loop once, weighted incident sums count both ends,
neighbor aggregation honors explicit direction, and expression neighbor_values
uses distinct neighbors. Directed triangle/core variants still reject explicitly.

## Verification

Full locked/offline Rust suite: **1,583 passed / 54 ignored / zero failed**
across 223 suite results. Strict all-target Clippy, Rust formatting and whitespace
checks passed. Rebuilt in-tree release extension: **2,829 maintained Python tests
passed / one skipped / 14 established warnings**, plus **273 executable docs
examples**. The three established binding build warnings are unchanged. No wheel
was installed. Native gates used the in-tree target and disabled incremental
compilation; Python/docs used the configured Python 3.9 interpreter and
`notes/scripts/verify_immutable_view_migration.py`. Black/isort/pre-commit remain
unavailable locally. Unrelated owner review/benchmark work was untouched.

Native release benchmark (`atomic_builder_properties -- --pending-readers`),
five observations each: **4.090 ms** median at 1k initial nodes/edges and
**30.737 ms** at 10k. The directed ring workflow stages one new node/two weighted
edges, then computes degree, incoming aggregation and weighted shortest paths,
attaches distances and publishes once. Fixture, initial capture, compilation and
assertions are excluded. This measures one bounded workflow, with no before/after
speedup, allocation/RSS, FFI or general scale claim. Raw observations are in
`notes/benchmarks/pending-topology-readers-2026-11-15.json`.
Native/Python coverage includes sparse selected IDs, provisional nodes and edges,
revised endpoints, parallel edges, loops, cascades, staged weights, retained
captures, multiple outer builders and errors after valid staged prefixes.

Existing unsupported-order guards now test unadapted sampling primitives because
degree/neighbor readers are newly certified. Exhaustive mathematical references
for existing structural/path/transform kernels remain in the maintained suites.

## Progress and remaining work

Evidence advances existing **09.4b.2, 30.1 and AR7.8**; their wider requirements
remain open. Inventory stays **215 checked / 53 unchecked**, with **44 unchecked
rows in release scope**. No new checklist row is introduced.

Unadapted sampling/community stages, general topology loops, captured-table
attachment, journal-owner dispatch, remaining lowering and naming/history policy
remain in their existing parent rows. The next slice is explicit captured-table
result attachment with retained row identity and one canonical publication.

User contract/example: [pending selected readers](../../docs/api/pending-topology-readers.md).
