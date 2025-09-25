# Warning Inventory (Placeholders)

`cargo check` currently emits warnings that correspond to planned extensions. Rather than
silencing them, we are tracking the affected areas here so future work can plug the
remaining gaps without hunting through build logs.

## storage/advanced_matrix/**
- Numerous unused imports (`NumericType`, `BackendHint`, `ComputeBackendExt`, etc.) in
  `backends`, `benchmarks`, and `neural` modules.
- Placeholder fields such as `energy` and `profile_data` in neural profiling structs.
- These modules back the planned BLAS/Numpy/Neural backends; the scaffolding is in place,
  but the concrete operations and benchmark harnesses have not been wired in yet.

## storage/array/**
- Helper traits (`RichDisplay`, SIMD optimizations, profiling helpers) are referenced but
  unused while we finish the streaming/table integration. The warnings mark the entry
  points where array-specific formatting and acceleration will land.

## storage/matrix/matrix_core.rs
- Display-related imports (`DisplayEngine`, `DisplayConfig`) and type helpers are present
  so matrices can eventually participate in the unified display pipeline. Implementation
  work is pending, so the imports are currently unused.

## viz/realtime/performance.rs
- Monitoring types (`PerformanceAlertSystem`, `MessageQueueMonitor`, `ProfileData`, etc.)
  contain dormant fields. They model the metrics we want to surface once the realtime UI
  exposes performance dashboards.

## viz/realtime/server/ws_bridge.rs
- Control acknowledgement helpers (`handle_control_message`, `send_control_ack`) are ready
  for UI↔engine round-tripping once the client surfaces per-command confirmation.

## viz/layouts/mod.rs
- Layout simulation retains unused fields/methods (`energy`, `pixel_to_hex`) that map to
  roadmap items such as energy reporting and hex-grid snapping.

## viz/streaming/util.rs
- `attr_value_to_json` is a compatibility shim for the deprecated streaming stack; we are
  keeping it until all legacy callers migrate to the realtime accessor API.

## Bin targets (visual_test.rs, visual_test_with_data.rs)
- Example binaries import extra helpers (e.g., `AttrValue`) while documenting the intended
  end-to-end flows. They double as runnable notebooks for future tutorials.

Whenever we convert one of these placeholders into live functionality, remove the matching
entry here so we can keep the warning list honest.
