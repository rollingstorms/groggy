Realtime Viz Integration — Waterfall Plan (4–5 Phases)

This plan replaces the “hardcoded example” path and fully wires Realtime rendering to the DataSource via a Realtime Viz Accessor, a WS/HTTP server, and the N-D embedding & controls on the client. Each phase has goals, deliverables, acceptance criteria, and notes on likely touchpoints (module/class names are illustrative; align them to your repo names).

⸻

Phase 1 — Access Layer: Realtime VizAccessor (engine-facing adapter)

Goal: Create a clean, testable bridge from DataSource → Realtime Engine, independent of transport/UI.

Work
	•	Define trait RealtimeVizAccessor (or reuse existing VizAccessor name if you prefer to generalize) that exposes:
	•	initial_snapshot() → {nodes, edges, positions, attrs, meta}
	•	subscribe(tx) → streams EngineUpdate (node/edge add/remove, attr changes, coordinates, selections)
	•	(Optional) apply_control(ControlMsg) for server-side reactions to client controls.
	•	Implement DataSourceRealtimeAccessor:
	•	Pulls from Arc<dyn DataSource>: get_graph_nodes(), get_graph_edges(), get_graph_metadata().
	•	Layout: tries engine layout; if not available, calls compute_layout(algorithm) on DataSource.
	•	Converts DS rows/attrs into engine‐native structs (stable IDs, typed values).
	•	Define internal message schema:
	•	EngineSnapshot { nodes: Vec<Node>, edges: Vec<Edge>, pos: Vec<[f32; D]>, meta: Meta }
	•	EngineUpdate::NodeAdded/Removed/Changed, Edge*, AttrChanged, PositionDelta, SelectionChanged, etc.
	•	Add small conversion helpers (e.g., attr_value_to_display_text, attr_value_to_json) so bootstrap HTML doesn’t leak tagged enums.

Deliverables
	•	realtime_viz_accessor.rs (trait + DS impl)
	•	engine_messages.rs (Snapshot/Update enums)
	•	Unit tests for snapshot conversion & streaming mock updates

Acceptance
	•	Pure Rust unit test can “play” a DS with 100 nodes/200 edges → initial_snapshot() returns consistent IDs, positions, and attrs
	•	Accessor can push a synthetic update stream into a channel without any server or UI running

Notes/Touchpoints
	•	New module: viz/realtime/accessor/
	•	Depends on: DataSource, engine model types, attr value utilities

⸻

Phase 2 — Transport Layer: Realtime Server (HTTP + WebSocket)

Goal: Serve a minimal page and a WS endpoint that bridges engine messages to clients. Reuse the Streaming server’s working pieces where practical.

Work
	•	Add a Realtime route to existing StreamingServer or create RealtimeServer:
	•	HTTP: /realtime/ serves a single HTML + JS bundle (canvas + controls shell)
	•	WS: /realtime/ws attaches to a broadcast channel
	•	Server lifecycle:
	•	Start server with RenderOptions.port (fallback default)
	•	On first client connect:
	•	Send EngineSnapshot immediately (no blank screen)
	•	Subscribe client to broadcast updates (EngineUpdate*)
	•	Bind client → server control channel for commands (e.g., change layout, filter, dim, dimensionality k)
	•	Serialization:
	•	JSON message frames {type: "snapshot" | "update" | "control_ack", payload: ...}
	•	Validate large payload performance (chunking snapshots > N MB, gzip if you already use it)
	•	Edge cases:
	•	Multiple clients: simple broadcast fan-out
	•	Late joiners: receive the latest snapshot, then updates
	•	Backpressure: drop-old or coalesce PositionDelta bursts (configurable)

Deliverables
	•	realtime_server.rs (or streaming_server.rs extended with /realtime)
	•	ws_bridge.rs (server <-> engine channels)
	•	Basic HTML/JS skeleton (can reuse styles/components from Streaming)

Acceptance
	•	viz.server(Realtime) starts an HTTP server; navigating to /realtime/ opens a WS connection and renders a non-empty graph from live DataSource via Accessor
	•	Refreshing the page replays snapshot without panics; multi-client works

Notes/Touchpoints
	•	Borrow HTTP & WS bootstrap code from Streaming to minimize new moving parts
	•	Keep route names distinct (/stream/… vs /realtime/…) to lower confusion during migration

⸻

Phase 3 — Engine Wiring: Snapshot, Deltas, and State Sync

Goal: Make the Realtime Engine a first-class producer/consumer of Accessor + WS streams, with predictable state.

Work
	•	Engine API:
	•	engine.load_snapshot(snapshot: EngineSnapshot)
	•	engine.apply(update: EngineUpdate)
	•	engine.subscribe(tx: Sender<EngineUpdate>) for broadcasting out changes (e.g., physics loop, server-side ops)
	•	Sync policy:
	•	Initial: only Accessor pushes (DS-driven)
	•	Optional later: engine can generate deltas (layout iterations, simulation ticks); throttle to ≤ N FPS
	•	Ordering & idempotence:
	•	Guarantee deterministic merge order (snapshot precedes any update)
	•	De-dupe or coalesce repeated AttrChanged/PositionDelta within a tick
	•	Error handling:
	•	If DS lacks positions and engine layout fails, Accessor must fall back to DS layout method; surface a warning, not a crash
	•	Memory:
	•	Large graphs: accept a “thin snapshot” mode (lazy attribute hydration by column on demand)

Deliverables
	•	engine/mod.rs updates (snapshot/delta API)
	•	engine_sync.rs for ordered apply/merge & coalescing
	•	Tests: snapshot→updates → final state equals expected set/positions

Acceptance
	•	Realtime render shows correct counts, ID stability, and attribute visibility
	•	Applying scripted EngineUpdate sequences changes the canvas instantly; no client reload required

Notes/Touchpoints
	•	Keep the message schema stable; client relies on it (Phase 4)

⸻

Phase 4 — Client UI: N-Dim Embedding & Advanced Controls

Goal: Expose the “sophisticated n-dimensional embedding” and advanced controls in the browser, driven by server WS.

Work
	•	Embedding controls:
	•	Dimensionality selector k ∈ {2,3, …, n}; show which embedding is active (e.g., PCA/UMAP/force/PD&F)
	•	Projection toggles: 2D view, 3D orbit, small multiples (if supported)
	•	Recompute pipeline:
	•	Client emits ControlMsg::ChangeEmbedding { method, k, params }
	•	Server applies: either calls engine compute, or proxies to Accessor/DS compute_layout(algorithm)
	•	On success, engine broadcasts a PositionDelta/EmbeddingChanged burst
	•	Graph controls:
	•	Filter/search (by attr values), degree sliders, highlight on hover/select
	•	Attribute panels: table view + pinned tooltips (ensure your display helpers are used)
	•	Play/pause if there’s a simulation loop; FPS display for diagnostics
	•	Bootstrapping:
	•	On connect, render from initial snapshot with zero flicker
	•	Defer heavy widgets until first paint (progressive UI)
	•	Performance:
	•	Virtualized drawing (retain from Streaming if available)
	•	RequestAnimationFrame loop applies batched deltas per frame
	•	Coalesce rapid PositionDelta updates client-side

Deliverables
	•	realtime_client.js (WS client, state store, renderer glue)
	•	Minimal components for controls (dimensionality, method dropdown, filters, search)
	•	CSS reuse from Streaming (rounded cards, table styles)

Acceptance
	•	Change embedding method/k from the UI and see positions update in real time
	•	Filters and search update the scene without reloads
	•	Client reconnect cleanly restores state (snapshot + recent updates)

Notes/Touchpoints
	•	Keep message types versioned: {version, type, payload} for future migrations
	•	Consider small test datasets for UI dev (1k nodes, 5k edges) plus a stress dataset (~50k/200k)

⸻

Phase 5 — Observability, Hardening & Migration

Goal: Make it reliable in dev and staging, instrumented, and with a clear path off the “hardcoded example”.

Work
	•	Telemetry & logs:
	•	Server: connection counts, bytes/s, backlog, dropped frames, apply times
	•	Engine: snapshot load time, update throughput, coalescing stats
	•	Client: FPS, render queue depth, dropped frames
	•	Health & readiness:
	•	/healthz returns OK when Accessor + Engine are live
	•	/statez includes counts and last update timestamps
	•	Config:
	•	RenderOptions { backend: Realtime, port, embedding_default, thin_snapshot, max_fps, coalesce_ms }
	•	CLI switches or env vars for easy tuning
	•	Error surfacing:
	•	Client toast/banner on server errors; retry with exponential backoff
	•	Migration plan:
	•	Deprecate hardcoded example: behind a feature flag initially, then remove
	•	Keep Streaming backend routable in parallel until Realtime hits stability bar
	•	Docs & examples:
	•	End-to-end example: viz.server(Realtime).show() with a real DataSource
	•	Message schema doc (JSON examples for snapshot & updates)
	•	Troubleshooting guide (common WS issues, CORS/ports, “blank screen” checklists)

Deliverables
	•	Metrics counters/timers (your preferred crate) + logs
	•	Health endpoints & config plumbing
	•	Migration notes & updated README/guide

Acceptance
	•	Load tests don’t crash the server; telemetry dashboards show stable throughput
	•	Feature flag off → hardcoded example path is unreachable; on → still testable during transition
	•	Clear, reproducible instructions to stand up Realtime against a real DataSource

⸻

Implementation Map (Quick Pointers)
	•	Backend selection: extend viz.show()/viz.server() → match VizBackend::Realtime { … } to:
	1.	Build DataSourceRealtimeAccessor
	2.	engine.load_snapshot(accessor.initial_snapshot())
	3.	Start RealtimeServer (or /realtime route) with WS bridge
	4.	accessor.subscribe(engine_tx) and engine.subscribe(broadcast_tx)
	•	Message schema (example):

{ "version": 1, "type": "snapshot", "payload": { "nodes": [...], "edges": [...], "pos": [[x,y],[...]], "meta": {...} } }
{ "version": 1, "type": "update",   "payload": { "kind": "PositionDelta", "items": [{"id": 42, "dx":0.1, "dy":-0.03}] } }
{ "version": 1, "type": "control",  "payload": { "kind": "ChangeEmbedding", "method":"pca", "k":3, "params":{} } }

	•	Risk hotspots & mitigations
	•	Blank page on connect: ensure snapshot is sent immediately after WS open.
	•	Layout stalls: set a server-side compute timeout; fall back to DS layout.
	•	Throughput spikes: coalesce PositionDelta updates (server & client).
	•	Schema drift: version field; reject/translate unknown versions.

⸻

Done = “Definition of Ready/Done” Summary
	•	Phase 1 (Accessor): Unit-tested snapshot/updates from real DataSource without any server/UI.
	•	Phase 2 (Server): /realtime/ serves a page; WS connects; snapshot paints; multiple clients OK.
	•	Phase 3 (Engine): Snapshot + deltas apply deterministically; scripted updates reflect in UI.
	•	Phase 4 (Client): N-D embedding control works end-to-end; filters/search stable; reconnect restores.
	•	Phase 5 (Ops/Migration): Telemetry + health; feature-flagged removal of hardcoded example; docs updated.

If you want, I can turn this into a repo PR skeleton (files/folders + stubbed functions) that compiles, so you can fill in the bodies.