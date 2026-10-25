# Bounded subgraph sampler bundle — 2026-10-24

Closes **09.4b.2aa / 09.4b.2ab / 09.4b.2ac**, extending **30.1 / AR7.8**.
Continues the requested larger delivery cadence. This batch repairs selected
iteration, bounded expansion and consistent emission. The nested for-each
executor stays uncertified and is now explicitly tracked as **09.4b.2ad** under
the already open execution-boundary parent; this is a decomposition of existing
work, not a new feature requirement or approval gate.

## Ordered selected iteration (09.4b.2aa)

Certify iterate_nodes/iterate_edges in property-atomic builder algorithms/loops.
Prepare active selected identities, endpoint closure, adjacency and endpoint maps
once per step. Node iteration emits ascending singleton IDs and only their
selected self-loops; edge iteration emits ascending edge IDs with exactly that
edge and its endpoints. Preserve parallel edges as distinct seeds. Previously,
owner-wide node induction could reintroduce edges omitted from a parent selector.
Outputs prepare privately, with cancellation before installation; no graph state
is published by selector construction.

## Invocation-bounded neighborhoods (09.4b.2ab)

Replace owner-wide unified_neighborhood with BFS over checked invocation-selected
adjacency. Directed expansion is outgoing; undirected expansion uses both
orientations. Seeds and outputs cannot escape the call's node/edge selection.
Source arrays preserve order and must share the exact owner with valid endpoint
closure. Numeric ID coincidence across different owners is insufficient.

Support source node/edge maps directly, repairing the public nonoptimized/legacy
mask-to-neighborhood path. Source keys are seeds regardless of values: node keys
become singleton seeds; edge keys become endpoint seeds. Map order is ascending
ID. Array inputs validate entirely before computation. At hops=0, induce selected
edges on seed nodes; large hop limits terminate on empty frontier. Source and
target may alias. Hops must be a supplied nonnegative integer or default 1.

## Consistent bounded emission (09.4b.2ac)

Emit from node maps, edge maps or arrays with selected identity validation.
Apply induced consistently in both modes and every source type. True induces
only invocation-selected edges; false preserves exact array/edge-map edges and
leaves node-map selections edgeless. Per-item maps are ordered; arrays preserve
input order; unified arrays union memberships before optional induction.

Previously, per-item flags were ignored and induction used owner edges. Corrected
membership/content IDs may differ from those historical results. Preserve empty
conventions: unified empty maps produce one empty selector, empty arrays produce
none, per-item empty inputs produce none. Supplied mode/induced controls must have
the declared text/Boolean types. All failures preserve the previous target array.
Standalone sampler validation recognizes its emitted subgraph output and no
longer warns that the pipeline must attach a graph property.

## Ownership and remaining boundary

Results are independently owned membership sets connected to the same legacy
owner. They are not pinned immutable property snapshots; later owner changes can
remain visible. This batch explicitly does not close legacy source-pin/view
migration. No owner/history cloning, graph topology mutation or new FFI business
logic is introduced.

Standalone sampler pipelines still reject atomic=True. Certified selector steps
can participate inside the property-atomic algorithm executor, whose outer batch
owns property acceptance. Nested for-each remains default-uncertified: its old
executor takes only the first explicitly emitted child selector, falls back on
input even for explicit empty output, and lacks recursive effect preflight and
complete transient-output ownership. These are the concrete next 09.4b.2ad tasks.
Unsupported-capability fixtures now name that still-uncertified step.

## Costs and verification

Selected adjacency/endpoints use O(V+E) working storage per step plus sorted rows.
Induction visits selected rows of output nodes rather than scanning the whole
owner graph per seed. Overlapping output memberships still allocate separately;
per-seed BFS can visit the entire selection. Content IDs and sorting add work.
Cancellation covers validation, traversal and final installation, with no bounded
latency promise for allocation/sorting. Native tests cover selected edge subsets,
parallel/loop identities, every emission source/mode/flag, large hops, foreign
owners, invalid endpoint closure, empty conventions and prior-output preservation.
Python tests cover sampler IR/legacy optimized/unoptimized paths, selected scope
and property-atomic loops.

Final gates: **969 native tests passed, 52 ignored; 1,516 root Python tests passed,
1 skipped; 64 executable documentation examples passed; strict all-target core
Clippy passed.** Focused sampling/sampler/atomic Python run: 140 passed. Native
sampling module: 13 passed. Offline locked dependencies and in-tree extension
artifacts were used. Root Python excludes the owner benchmark test and separate
legacy modules survey.

API/examples: `../../docs/api/builder-subgraph-sampling.md`.
Release harness: `benches/subgraph_sampler_steps.rs`. Six modes at 1,000/10,000
nodes and edges, five observations each, time selector iteration, 0/1-hop expansion,
mask-unified and array-unified emission. Fixture/spec preparation and validation
excluded; every output membership/order/cardinality, induction and zero graph
publications checked. This is not a matched speedup, RSS/FFI/allocation or
production-scale claim.

Raw receipt: `../benchmarks/subgraph-sampler-bundle-2026-10-24.json`.

| Workload | Nodes (= edges) | Median ms, five observations |
| --- | ---: | ---: |
| iterate_nodes | 1,000 | 1.304 |
| iterate_edges | 1,000 | 1.418 |
| neighbors_zero | 1,000 | 1.687 |
| neighbors_one | 1,000 | 1.776 |
| mask_unified | 1,000 | 0.212 |
| array_unified | 1,000 | 1.060 |
| iterate_nodes | 10,000 | 8.555 |
| iterate_edges | 10,000 | 10.158 |
| neighbors_zero | 10,000 | 13.270 |
| neighbors_one | 10,000 | 16.550 |
| mask_unified | 10,000 | 2.058 |
| array_unified | 10,000 | 10.844 |
