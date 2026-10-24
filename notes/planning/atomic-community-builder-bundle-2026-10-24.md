# Checked community helper bundle — 2026-10-24

Closes **09.4b.2x / 09.4b.2y / 09.4b.2z**, extending **30.1 / AR7.8**.
Continues the requested larger cadence: three related helpers in one reviewed,
verified implementation commit. These repair existing engine functionality;
broader execution-boundary parents remain partial. Existing workbook decisions
remain the authority; no additional policy approval is required.

## Deterministic checked seeding (09.4b.2x)

Certify singleton, degree and random community initialization. Singleton/random
validate active selected nodes without an edge scan. IDs convert to i64 with a
checked conversion. Random labels use stable ID traversal and an isolated RNG;
seeded calls neither reset nor consume shared RNG state. Require positive k for
random assignment and reject supplied malformed/nonpositive k or negative/invalid
serialized seeds. Direct native k values must keep every generated label in i64.

Degree initialization uses checked selected adjacency rather than swallowed
owner-degree errors. Rank by outgoing/incidence degree descending and node ID
ascending; take ceil(N/10) hubs (at least one for nonempty graphs), then assign
remaining nodes in ID order using the first sorted already labeled neighbor or
their own ID. Parallel incidences stay distinct; a stored loop counts once in
this initialization heuristic. Empty selections remain valid after controls
validate. Outputs install only after successful preparation/cancellation checks.

## Synchronous propagation (09.4b.2y)

Certify label_propagate_step with active selected topology and endpoint closure.
Directed votes use outgoing edges, undirected both directions. An explicit self
vote is always present, parallel edges vote separately and a stored self-loop adds
one vote. A pass reads fixed input labels; it never sees earlier output rows.
Canonical typed serialization replaces Debug category identity and provides
stable ties. Present nonfinite labels reject; sparse missing selected neighbors
retain singleton-ID fallback. Isolated nodes keep their label.

Accept both node maps and node columns, repairing composition from ordinary
property loads. Map rows sort by ID; explicit column order remains. Sparse rows
stay sparse, foreign/duplicate source IDs reject, and source/target aliasing is
safe. A shared checked input/output path serves propagation and modularity.

## Actual directed/undirected local modularity gains (09.4b.2z)

The existing modularity_gain returned an edge-fraction proxy, not a change in the
modularity objective. Replace it with removal-plus-insertion delta calculations
for the standard unweighted resolution-1 objectives. Undirected Q uses community
internal edges and squared degree volumes; directed Q uses in/out volumes.
Precompute degrees/community totals once, then collect incident community contacts
per result node. Undirected loops count twice in degree; directed loops once per
in/out degree. Loop internal-edge terms cancel when moved; parallel edges retain
multiplicity. Avoid full partition scans per candidate.

Return max(0, improving move gains) over other communities connected by a non-self
selected edge. Directed candidates include incoming and outgoing contacts. Staying
is the zero baseline; no fresh singleton or nonneighbor community is considered.
Results do not apply moves or return destinations. Every node evaluates against
the unchanged partition. Empty-edge selections return zero.

Partition labels must be integers; malformed values no longer silently become
node-ID communities. Missing selected rows retain ID fallback and sparse output
rows. Callers should avoid explicit labels colliding with those fallback IDs.
All results must be finite. Existing proxy scores intentionally change.
Formula reference and derivation scope: `../../docs/api/builder-community.md`.

## Costs and remaining work

Share existing checked selected-node/adjacency helpers; graphs/history are not
cloned. Working storage includes input labels, selected adjacency, degrees,
community totals/contact counts and output columns. Degree/category/adjacency
sorting adds work, while modularity now computes an actual objective rather than
the cheaper proxy. Cancellation covers preparation and node passes and is checked
before result installation; large rows/sorts have no bounded interruption promise.
Native core owns all logic; no FFI business logic or external graph execution is
introduced. The independent test objectives and all graph fixtures are in-tree.

Whole outer transactions, optimizer property effects, mutable storage escapes,
subgraph-producing sampling, temporal/path steps and composer history/source pins
remain separate open work. These helpers are not a complete optimizer or weighted
modularity API. No numerical/algorithm parent is closed by this bundle alone.

## Verification and release receipt

**964 native tests passed, 52 ignored; 1,489 root Python API tests passed,
1 skipped; 60 executable documentation examples passed; strict all-target core
Clippy passed.** The focused community/LPA Python suite passed 41 cases; the
focused native community suite passed eight tests. Builds used offline locked
dependencies and in-tree artifacts. The root Python gate excludes the owner
benchmark test and the separate legacy modules survey. Native regressions independently recompute full
modularity before/after moves over all 63 nonempty simple undirected four-node graphs and
511 nonempty directed three-node graphs with every binary partition, plus explicit
parallel-edge/loop cases and separate empty selections. Additional tests cover RNG isolation, stable hub ties,
sparse column order, foreign IDs and cancellation preserving prior outputs.
Python tests cover owned/persistent output, strict controls, synchronous voting,
map inputs/staged loads, full-objective examples and rollback.

Release harness: `benches/community_builder_steps.rs`, seven modes at 1,000/10,000
nodes and edges with five observations each: singleton/degree/random seeding,
directed/undirected propagation, directed/undirected modularity gains. Fixture/spec
preparation excluded; calculation, attachment and one checked publication
included. Check every result and accepted state. No matched speedup, RSS/FFI,
allocation or production-scale claim.

Raw receipt: `../benchmarks/community-builder-bundle-2026-10-24.json`.

| Workload | Nodes (= edges) | Median ms, five observations |
| --- | ---: | ---: |
| singleton | 1,000 | 2.099 |
| degree | 1,000 | 1.813 |
| random | 1,000 | 1.281 |
| labels_undirected | 1,000 | 1.721 |
| labels_directed | 1,000 | 1.430 |
| modularity_undirected | 1,000 | 1.697 |
| modularity_directed | 1,000 | 1.745 |
| singleton | 10,000 | 9.933 |
| degree | 10,000 | 12.980 |
| random | 10,000 | 9.905 |
| labels_undirected | 10,000 | 14.612 |
| labels_directed | 10,000 | 14.102 |
| modularity_undirected | 10,000 | 17.933 |
| modularity_directed | 10,000 | 17.652 |
