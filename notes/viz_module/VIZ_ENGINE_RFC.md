# Viz Engine RFC

## Status
- Proposed
- Owner: Visualization module maintainers
- Target: Groggy `0.6.x` foundation, iterative rollout

## Summary
We should redesign the visualization module around one lightweight interactive engine that supports:
- Graph visualization (`graph` view)
- Table/database visualization (`table` view)
- Linked graph+table exploration (`split` view)

In this model, **honeycomb is a view mode/layout plugin**, not the architecture center.

## Problem Statement
Current viz code has several overlapping paths:
- Parallel rendering/runtime layers (legacy streaming + realtime)
- Duplicate honeycomb logic paths
- API/documentation drift
- Placeholder and shim code in key workflows

This creates maintenance drag and makes it harder to ship a consistent interactive experience.

## Goals
- Single runtime pipeline for interactive visualization.
- Single state model for graph + table.
- Shared protocol for local (`local`) and remote (`remote`) operation.
- Keep runtime lightweight (fast startup, low memory, incremental updates).
- Treat layout/projection algorithms (including honeycomb) as plugins.

## Non-Goals
- Rebuilding all UI styling/themes in phase 1.
- Perfect feature parity with every legacy edge-case before migration starts.
- Supporting multiple incompatible client protocols.

## Conceptual Model

### Views
- `graph`: Canvas/WebGL graph rendering.
- `table`: Virtualized rows/columns.
- `split`: Graph + table with linked selections and filters.

### Honeycomb
- Honeycomb is one selectable graph view mode.
- It sits behind a common `LayoutKernel` interface.
- It shares interaction semantics with other modes (zoom/pan/select/filter).

## Architecture

### 1) DataModel
Canonical source snapshot + typed patches.
- Graph entities: nodes, edges, attributes
- Tabular entities: schemas, rows, sort/filter windows
- Versioned updates for replay/sync

### 2) StateModel
UI/runtime state only:
- Active view (`graph|table|split`)
- Selection, hover, filters, sorting
- Viewport/camera
- Active layout mode + params

### 3) LayoutKernel
Pluggable layout/projection interface:
- `honeycomb`
- `force_directed`
- `circular`
- `grid`

All implementations consume the same canonical graph input and return positions.

### 4) RenderCore
- Graph renderer: Canvas2D baseline, optional WebGL acceleration
- Table renderer: windowed virtualization only
- No backend-specific graph logic duplication

### 5) InteractionCore
One typed command layer:
- Pointer / wheel / keyboard
- Selection / filter / sort / camera
- Layout parameter changes

### 6) Runtime
- `local`: local in-process runtime
- `remote`: websocket-based runtime

Both use the same protocol/message types and state transitions.

## Protocol (Single Contract)

### Snapshot
Initial data payload:
- graph nodes/edges/meta
- table schema + initial windows
- default state/layout

### Patch
Incremental updates:
- graph deltas (node/edge add/remove/change)
- table deltas (window/row/schema updates)
- state deltas (optional)

### Control
Client-to-engine commands:
- change layout/mode
- set filters/sort
- set selection
- camera operations

### Event
Engine-to-client notifications:
- selection changed
- view changed
- perf stats
- warnings/errors

## Public API Direction

### Rust
- `graph.viz().open(mode, layout, options) -> VizSession`
- `graph.viz().render_html(options) -> String`
- `graph.viz().export(path, format, options) -> Result`

### Python
- `g.viz.show(...)` / `g.viz.server(...)` as compatibility shims initially
- New canonical names should mirror Rust session model over time

## Lightweight Performance Defaults
- Start with Canvas2D
- Virtualized table windows only
- Coalesced updates per frame
- Incremental layout recompute where possible
- Bounded caches and histories

## Repo Organization Proposal

```text
src/viz/
  mod.rs
  api/          # public facade, options, results
  model/        # canonical graph/table/state structs
  pipeline/     # embedding/projection/layout/interpolation/quality
  honeycomb/    # unified honeycomb kernel only
  runtime/      # engine loop + protocol + transport
  render/       # graph/table rendering adapters
```

## Migration Plan

### Phase 1: Foundation
- Add canonical DataModel + StateModel + protocol types.
- Introduce new runtime engine skeleton behind feature flag.

### Phase 2: Honeycomb Unification
- Move all honeycomb math/assignment/autoscale into one kernel.
- Route existing honeycomb callers through the unified kernel.

### Phase 3: Unified Views
- Implement `graph`, `table`, and `split` on one runtime.
- Add linked selection/filter behavior between graph and table.

### Phase 4: API Cutover
- Keep compatibility shims for existing calls.
- Update docs/examples to canonical API.
- Emit clear deprecation warnings for old paths.

### Phase 5: Cleanup
- Remove legacy streaming facade and duplicate code paths.
- Delete dead fallback/shim code and stale documentation references.

## Milestones (Approval-Gated)

This plan is intentionally split into **large milestones** with explicit approval checkpoints.
Each milestone is sized for sustained autonomous work, but bounded to avoid architectural drift.

### Milestone 0: Baseline and Guardrails
- Duration target: 2-3 days
- Objective: lock current behavior and define migration safety rails.
- Deliverables:
- feature inventory of current viz entry points (Rust + Python)
- legacy/deprecated path list and removal candidates
- baseline smoke tests and perf snapshot for comparison
- Exit criteria:
- agreed list of compatibility surfaces to preserve temporarily
- agreed “must not regress” behaviors for graph/table interaction
- Approval gate:
- approve inventory + guardrails before structural refactor begins

### Milestone 1: Canonical Core Models
- Duration target: 4-6 days
- Objective: introduce unified `DataModel`, `StateModel`, and protocol types without cutting over runtime.
- Deliverables:
- typed model modules (`model/`) for graph + table + shared state
- protocol enums/structs for `Snapshot`, `Patch`, `Control`, `Event`
- model-level tests for serialization, versioning, and state transitions
- Exit criteria:
- current code can construct canonical models in parallel
- protocol is stable enough to serve both local and remote runtime modes
- Approval gate:
- approve core schema and protocol contract before runtime integration

### Milestone 2: Unified Runtime Skeleton
- Duration target: 5-7 days
- Objective: build single runtime loop using canonical models and protocol, initially behind a feature flag.
- Deliverables:
- `runtime/engine` with command handling and update broadcast
- `runtime/transport` for local adapter and websocket adapter
- compatibility adapter so existing entry points can route into new runtime
- Exit criteria:
- graph and table snapshots render through one runtime path
- control messages flow end-to-end in both local and remote modes
- Approval gate:
- approve runtime skeleton and control flow before layout/plugin unification

### Milestone 3: Honeycomb + LayoutKernel Unification
- Duration target: 5-7 days
- Objective: merge duplicate honeycomb/layout logic into one `LayoutKernel` plugin system.
- Deliverables:
- shared layout interface and plugin registry
- unified honeycomb kernel (grid, assignment, autoscale, interaction params)
- migration of existing honeycomb callers to the shared kernel
- Exit criteria:
- no duplicate honeycomb computation path remains active
- honeycomb behavior is available in both graph-only and split views
- Approval gate:
- approve honeycomb interaction/parameter model before frontend polish

### Milestone 4: Unified Views (Graph/Table/Split)
- Duration target: 5-8 days
- Objective: deliver one interactive experience with linked graph/table behavior.
- Deliverables:
- graph view renderer + table virtualized renderer under one state model
- split view with linked selection/filter/sort
- view switching and persistence of shared state
- Exit criteria:
- selection in graph updates table context and vice versa
- large-table behavior remains lightweight via windowing
- Approval gate:
- approve UX semantics for linked views before deprecating legacy APIs

### Milestone 5: API Cutover and Compatibility Window
- Duration target: 3-5 days
- Objective: expose canonical API while keeping temporary compatibility shims.
- Deliverables:
- final Rust public API shape (`open`, `render_html`, `export`)
- Python mapping strategy with compatibility aliases
- migration docs for old-to-new call patterns
- Exit criteria:
- canonical API is documented and test-covered
- compatibility shims emit clear deprecation guidance
- Approval gate:
- approve public API contract before destructive cleanup

### Milestone 6: Removal and Hardening
- Duration target: 4-6 days
- Objective: remove legacy code paths and finalize reliability/performance pass.
- Deliverables:
- deletion of legacy wrappers/shims/unused branches
- test suite updates and end-to-end reliability pass
- performance comparison report against Milestone 0 baseline
- Exit criteria:
- legacy removal complete with no failing compatibility commitments
- performance and stability metrics within agreed thresholds
- Approval gate:
- final go/no-go for merge and long-term support branch strategy

## Autonomous Execution Rules
- Work continuously within the active milestone scope only.
- Do not start the next milestone without explicit approval.
- At each gate, provide:
- concise changelog
- test/perf evidence
- open risks and tradeoffs
- If a milestone reveals a major architecture conflict, stop and request direction
  before continuing.

## Recommended Check-In Cadence
- Lightweight async updates every 1-2 days during a milestone.
- Formal gate review at the end of each milestone.
- Optional midpoint review for Milestones 2-4 (higher integration risk).

## Deletion Candidates (Post-Cutover)
- Legacy compatibility wrappers that only reroute to realtime.
- Duplicate honeycomb implementations and utility drift.
- Placeholder-only interactive HTML branches not wired to runtime.
- Dead FFI helper methods retained only for old control flow.

## Testing Strategy
- Unit tests for model/protocol/layout kernel.
- Golden tests for layout output invariants.
- Integration tests for graph/table/split linked behavior.
- End-to-end tests for both local and remote modes.

## Acceptance Criteria
- One engine powers graph, table, and split views.
- Honeycomb is a plugin layout with no duplicate implementation.
- Same control protocol works in local and remote modes.
- Legacy paths are removed or explicitly deprecated with migration docs.
