# Immutable selections over shared read state — 2026-10-22

AR7.5b adds the canonical GraphView membership representation over PinnedReadState.
It is available natively and through thin Python adapters. Existing Subgraph/table
adapters still require migration; AR7.5/AR4/26.1 remain open.

## Public behavior

- Graph.read_view() captures current state, including uncheckpointed properties.
  Subgraph.pin_view() explicitly freezes an existing selection's current state;
  it fails if its selected IDs are no longer active. Neither operation creates a
  checkpoint or records a mutation.
- GraphView is immutable and holds no working-graph reference. Its values and
  membership survive owner mutation/deactivation/destruction. Working writes
  continue through the graph/legacy adapters while this read representation stays
  frozen. GraphView exposes no mutation API.
- Full views borrow captured membership; clones and nested selections share the
  pin. Selected membership is Arc-owned and cheap to clone. Children cannot add
  IDs or edges excluded by their parent, even if both endpoints remain selected.
- select_nodes induces edges only within the parent's existing edge scope;
  select_edges includes selected instances and their endpoints. Duplicate input
  IDs collapse to set membership. Empty selections and a single isolated node
  are valid. Every selected edge must have both endpoints selected.
- Node/edge attribute and endpoint reads reject IDs outside the selection. Missing
  attributes return none; explicit null remains a present property and is
  distinguishable with has_*_attribute/attribute_keys. Attribute types are retained.

## Indexed topology

The shared state now stores edge IDs aligned with sorted neighbor occurrences and
an incident-edge index. This lets non-induced selections filter by edge identity
in degree-proportional work, without scanning the entire graph or including
unselected parallel facts. Counts do not allocate neighbor/edge result vectors.

Neighbors/out_degree use outgoing instances for directed graphs and both endpoint
orientations for undirected graphs. degree counts all incident instances once;
in_degree counts directed targets, or incident degree when undirected. Parallel
edge instances remain distinct and a self-loop counts once in each applicable
operation. Neighbor order is neighbor ID then edge ID; incident order is edge ID.
These explicit GraphView semantics do not certify every legacy live caller.

## Python and owned projections

Python exposes GraphView through the package and native extension. read_view,
pin_view, properties, topology, selections and copy delegate to Rust. Selection
and ID-order work release the GIL over Send/Sync immutable data. Working-state
capture remains on its owner/GIL boundary. GraphView can be read by worker threads;
this does not implement multi-user graph writers.

Native to_snapshot explicitly materializes only the selected state. Editing a
subsequently constructed working graph cannot change the view. This is not a
portable retained-group expansion or a new table writeback policy. No implicit
refresh, timestamp reconstruction or history entry is introduced.

## Evidence and next work

Native/Python tests cover nested non-induced/parallel scopes, directed incoming/
outgoing incidence, loops/isolates, null/absent, high integers, owner mutation/drop,
legacy explicit pinning, independent projection and threaded reads. Criterion
covers full-view construction and selected-view clone alongside capture/clone.

Next: migrate existing graph/table adapters and algorithm/query consumers onto
this common read representation, retaining a separate working mutation owner.
Do not describe ordinary legacy Subgraph/table reads as frozen until that migration
and its read/write/cache regressions are complete.

Executed: full locked offline native suites; strict all-target core Clippy
(clean); rebuilt native extension/package exports; read-view/temporal/property/
persistence/identity/table Python gate: **176 passed, 1 existing skip**. Both
Python examples in `docs/immutable-read-views.md` executed successfully. Idle
Criterion estimates and the measured capture-index cost increase are in
`notes/benchmarks/immutable-graph-view-2026-10-22.md`.
