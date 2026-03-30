# Milestone 0: Baseline and Guardrails

## Scope
This document establishes the pre-refactor baseline for the viz redesign on branch `develop-viz-redo`.

## Baseline Snapshot
- Baseline commit: `2393e454`
- Baseline branch: `develop-viz-redo`
- Existing RFC: [VIZ_ENGINE_RFC.md](/Users/michaelroth/Documents/Code/groggy/notes/viz_module/VIZ_ENGINE_RFC.md)

## Current Public Viz Surface (Inventory)

### Rust entry point
- `Graph::viz() -> VizModule` in `/Users/michaelroth/Documents/Code/groggy/src/api/graph.rs`.

### Rust viz module surface
- `VizBackend`, `VizModule`, `InteractiveViz`, `RealTimeVisualization` in `/Users/michaelroth/Documents/Code/groggy/src/viz/mod.rs`.
- Compatibility streaming wrapper `StreamingServer` in `/Users/michaelroth/Documents/Code/groggy/src/viz/streaming/server.rs`.

### Python-facing surface
- `VizAccessor` (`show`, `server`, `update`) exposed via:
- `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/viz_accessor.rs`
- `/Users/michaelroth/Documents/Code/groggy/python-groggy/python/groggy/_groggy.pyi`
- `.viz` getters are exposed from multiple object types in:
- `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/api/graph.rs`
- `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/storage/*.rs`
- `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/subgraphs/*.rs`

## Known Drift / Loose Surface (Pre-Refactor)
- API doc examples mention `graph.viz().widget()` / `.serve(...)` in `/Users/michaelroth/Documents/Code/groggy/src/api/graph.rs`, but these are not canonical `VizModule` methods.
- Python package comments reference `graph.graph_viz()` in `/Users/michaelroth/Documents/Code/groggy/python-groggy/python/groggy/__init__.py`, but that method is not a canonical public endpoint.
- `VizBackend::Streaming` is currently redirected to realtime in the implementation path.
- Legacy/placeholder/deprecation markers are concentrated in viz and related FFI paths.

## Baseline Risk Signal Count
- `TODO|placeholder|Legacy|deprecated|DISABLED|NotImplemented` markers in targeted viz-related paths: **89**
  - Command:
  - `rg -n "TODO|placeholder|Legacy|deprecated|DISABLED|NotImplemented" src/viz python-groggy/src/ffi/viz_accessor.rs python-groggy/src/ffi/storage/table.rs python-groggy/src/ffi/storage/accessors.rs python-groggy/src/ffi/storage/nodes_array.rs python-groggy/src/ffi/storage/edges_array.rs python-groggy/python/groggy/__init__.py src/api/graph.rs`

## Baseline Smoke Tests

### Rust
- Command: `cargo test viz:: -- --nocapture`
- Result: `102 passed; 0 failed` (plus filtered-out unrelated test binaries)
- Observed runtime: compile ~14.66s, viz test execution ~0.41s.

### Python
- Command: `pytest -q tests -k viz`
- Result: `2 passed, 545 deselected in 0.73s`

## Guardrails for Milestones 1-2

### Must Not Regress
- `.viz` accessor availability from existing object families (Graph/Subgraph/Table/Accessor paths).
- Interactive session startup path from Python (`show` and `server` must still function via compatibility route).
- Graph and table data delivery through a single snapshot/update contract once introduced.
- Existing `cargo test viz::` and `pytest -k viz` smoke gates.

### Allowed to Change Early
- Internal module organization under `src/viz`.
- Protocol and state internals, if compatibility adapters preserve current call sites.
- Naming cleanup from legacy terms to `local`/`remote` model names.

### Disallowed Before Milestone 5 Approval
- Removing compatibility shims used by current Python entry points.
- Deleting current frontend control affordances without equivalent replacement.
- Breaking public import/stub discoverability for `VizAccessor`.

## Compatibility Surface to Preserve Temporarily
- Python:
- `obj.viz.show(...)`
- `obj.viz.server(...)`
- `obj.viz.update(...)`
- Rust:
- `Graph::viz()`
- `VizModule::show/render/save/interactive/static_viz` (may become wrappers)

## Milestone 0 Exit Criteria Status
- [x] Feature inventory captured.
- [x] Legacy/deprecated path signal identified.
- [x] Baseline smoke tests executed and recorded.
- [x] Guardrails and compatibility commitments documented.
