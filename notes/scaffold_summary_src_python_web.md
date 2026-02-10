# Scaffold/Validation Summary (src, python-groggy, web)
## High-level counts
- TODO/FIXME: 183
- todo! macro: 8
- unimplemented! macro: 0
- #[ignore] tests: 7
- panic: 53

## TODO/FIXME counts by area
- src: 131
- python-groggy/src/ffi: 31
- python-groggy/python: 18
- web: 2

## Top TODO/FIXME files (by count)
### src
- src/viz/realtime/engine.rs: 18
- src/storage/table/base.rs: 9
- src/api/graph.rs: 8
- src/lib.rs: 8
- src/state/history.rs: 6
- src/state/ref_manager.rs: 6
- src/storage/matrix/matrix_core.rs: 6
- src/storage/table/integration_tests.rs: 6
- src/utils/config.rs: 6
- src/errors.rs: 5
### python-groggy/src/ffi
- python-groggy/src/ffi/storage/table.rs: 7
- python-groggy/src/ffi/api/graph.rs: 4
- python-groggy/src/ffi/storage/accessors.rs: 4
- python-groggy/src/ffi/storage/components.rs: 3
- python-groggy/src/ffi/storage/matrix.rs: 3
- python-groggy/src/ffi/subgraphs/subgraph.rs: 3
- python-groggy/src/ffi/experimental.rs: 2
- python-groggy/src/ffi/storage/array.rs: 1
- python-groggy/src/ffi/storage/edges_array.rs: 1
- python-groggy/src/ffi/storage/nodes_array.rs: 1
### python-groggy/python
- python-groggy/python/groggy/_groggy.cpython-39-darwin.so: 16
- python-groggy/python/groggy/builder/ir/nodes.py: 2
### web
- web/app.js: 1
- web/index.html: 1

## Topic deep links
### batch_executor (src/algorithms/execution/batch_executor.rs)
- L608: #[ignore] // TODO: Fix test - needs proper subgraph setup and Graph API
- L610: // TODO: Need proper Graph -> Subgraph conversion for test setup
### temporal (core)
- src/temporal/index.rs
- src/temporal/mod.rs
- src/temporal/snapshot.rs
### history forest (core)
- src/api/graph.rs
- src/lib.rs
- src/state/change_tracker.rs
- src/state/history.rs
- src/state/mod.rs
- src/state/ref_manager.rs
- src/state/space.rs
- src/state/state.rs
- src/storage/pool.rs
- src/temporal/index.rs
- src/temporal/snapshot.rs
- src/traits/edge_operations.rs
- src/traits/graph_entity.rs
- src/traits/mod.rs
- src/traits/node_operations.rs
- src/utils/strategies.rs
### viz realtime
- src/viz/realtime/accessor/engine_messages.rs
- src/viz/realtime/accessor/mod.rs
- src/viz/realtime/accessor/realtime_viz_accessor.rs
- src/viz/realtime/engine.rs
- src/viz/realtime/engine_sync.rs
- src/viz/realtime/interaction/globe_controller.rs
- src/viz/realtime/interaction/math.rs
- src/viz/realtime/interaction/mod.rs
- src/viz/realtime/interaction/pan_controller.rs
- src/viz/realtime/mod.rs
- src/viz/realtime/server/mod.rs
- src/viz/realtime/server/realtime_server.rs
- src/viz/realtime/server/ws_bridge.rs