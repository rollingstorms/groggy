# Cargo Check Unused Methods Analysis

## Raw Output from `cargo check`

```
warning: associated function `check_availability` is never used
  --> src/storage/advanced_matrix/backends/numpy.rs:30:8

warning: method `as_mut_ptr` is never used
   --> src/storage/advanced_matrix/memory.rs:133:8

warning: method `ensure_same_graph` is never used
   --> src/storage/advanced_matrix/neural/autodiff.rs:853:8

warning: methods `attr_value_to_primitive_value` and `attr_value_to_display_value` are never used
    --> src/storage/table/base.rs:2521:8

warning: method `convert_to_agg_functions` is never used
   --> src/subgraphs/composer.rs:627:8

warning: method `compute_force_directed_layout` is never used
   --> src/subgraphs/visualization.rs:379:8

warning: method `render_streaming` is never used
   --> src/viz/mod.rs:284:8

warning: method `generate_update` is never used
    --> src/viz/realtime/engine.rs:2300:14

warning: methods `handle_control_message` and `send_control_ack` are never used
   --> src/viz/realtime/server/ws_bridge.rs:569:14

warning: method `pixel_to_hex` is never used
    --> src/viz/layouts/mod.rs:1208:8

warning: function `attr_value_to_json` is never used
  --> src/viz/streaming/util.rs:24:4
```

## Categorization by Module/Domain

### Storage Layer
- **Advanced Matrix Backends**: `check_availability` (numpy.rs:30) - Backend availability checking
- **Memory Management**: `as_mut_ptr` (memory.rs:133) - Low-level pointer access
- **AutoDiff System**: `ensure_same_graph` (autodiff.rs:853) - Cross-graph validation
- **Table Operations**: `attr_value_to_primitive_value`, `attr_value_to_display_value` (base.rs:2521) - Attribute conversion utilities

### Graph Operations
- **Subgraphs**: `convert_to_agg_functions` (composer.rs:627) - Aggregation function conversion
- **Subgraph Visualization**: `compute_force_directed_layout` (visualization.rs:379) - Layout algorithm

### Visualization System
- **Core Viz**: `render_streaming` (viz/mod.rs:284) - Streaming render capability
- **Realtime Engine**: `generate_update` (realtime/engine.rs:2300) - Update generation
- **WebSocket Bridge**: `handle_control_message`, `send_control_ack` (ws_bridge.rs:569) - Control message handling
- **Layout System**: `pixel_to_hex` (layouts/mod.rs:1208) - Coordinate conversion
- **Streaming Utilities**: `attr_value_to_json` (streaming/util.rs:24) - JSON serialization

## Analysis by Priority/Impact

### High Priority (Core Infrastructure)
1. **AutoDiff Cross-Graph Validation** (`ensure_same_graph` - autodiff.rs:853)
   - Critical for tensor operation safety
   - May be needed for future cross-graph operations
   - Consider keeping for defensive programming

2. **Memory Management** (`as_mut_ptr` - memory.rs:133)
   - Low-level memory access for performance-critical paths
   - May be needed for FFI or unsafe optimizations
   - Consider keeping for future use

### Medium Priority (Feature Infrastructure)
3. **Streaming/Realtime Infrastructure**
   - `render_streaming` (viz/mod.rs:284)
   - `generate_update` (realtime/engine.rs:2300)
   - `handle_control_message`, `send_control_ack` (ws_bridge.rs:569)
   - These may be part of incomplete streaming features

4. **Layout System** (`pixel_to_hex` - layouts/mod.rs:1208)
   - Coordinate conversion for hexagonal layouts
   - May be used in honeycomb projection

### Low Priority (Utility Functions)
5. **Attribute Conversion** (`attr_value_to_primitive_value`, `attr_value_to_display_value` - base.rs:2521)
   - Table display utilities
   - May be used in future table formatting

6. **Aggregation Functions** (`convert_to_agg_functions` - composer.rs:627)
   - Subgraph aggregation utilities
   - May be part of incomplete aggregation features

7. **Serialization** (`attr_value_to_json` - streaming/util.rs:24)
   - JSON conversion utility
   - May be used in streaming or API features

### Lowest Priority (Backend Support)
8. **Backend Availability** (`check_availability` - numpy.rs:30)
   - NumPy backend availability checking
   - May be dead code if NumPy backend is not active

9. **Force-Directed Layout** (`compute_force_directed_layout` - visualization.rs:379)
   - Alternative layout algorithm
   - May be superseded by energy embedding system

## Recommendations

### Immediate Actions
- **Remove**: `check_availability` (likely dead code)
- **Remove**: `compute_force_directed_layout` (superseded by energy embedding)

### Conditional Removal
- **Review and possibly remove**: Attribute conversion utilities if not used in display pipeline
- **Review and possibly remove**: `attr_value_to_json` if streaming doesn't use JSON

### Keep for Future
- **Keep**: `ensure_same_graph` (defensive programming for autodiff)
- **Keep**: `as_mut_ptr` (performance-critical memory access)
- **Keep**: Streaming/realtime methods (may be part of incomplete features)
- **Keep**: `pixel_to_hex` (may be used in honeycomb layouts)

### Investigation Needed
- Check if streaming infrastructure methods are part of active development
- Verify if layout conversion methods are used in current visualization pipeline
- Confirm if attribute conversion methods are needed for future table display features