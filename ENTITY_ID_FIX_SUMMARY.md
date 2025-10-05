# Entity ID Mismatch Fix

## Problem Description

There was a mismatch between entity IDs (node_ids and edge_ids) from the graph/graph tables and how they were being used in the visualization system. This caused incorrect attribute mapping when using visualization parameters that referenced columns (e.g., `node_color='object_name'`, `edge_label='method_name'`).

## Root Cause

The issue was in `src/viz/realtime/accessor/realtime_viz_accessor.rs` in two key functions:

1. **`convert_nodes()`** (lines 433-518)
2. **`convert_edges()`** (lines 538-620)

Both functions were incorrectly using entity IDs as array indices when resolving visualization parameters:

```rust
// WRONG: Using node_id as array index
node.color = self.resolve_string_param(&config.node_color, node_id as usize, &sanitized_attributes);

// WRONG: Using edge_id as array index  
edge.color = self.resolve_string_param(&config.edge_color, edge_id as usize, &sanitized_attributes);
```

### Why This Was Wrong

When iterating over nodes and edges, the code uses `.enumerate()` which provides:
- `idx`: The position in the array (0, 1, 2, 3, ...)
- `graph_node` or `graph_edge`: The actual entity data with its ID

Entity IDs (node_id, edge_id) are not necessarily sequential or starting from 0. They are the actual graph entity identifiers. However, when resolving VizParameters that reference arrays or columns, we need to use the **array index** (`idx`), not the entity ID.

For example:
- Graph might have node IDs: [10, 20, 30, 40, 50]
- Array indices are: [0, 1, 2, 3, 4]
- When resolving `node_color` from a column, we need index 0-4, not 10-50

## The Fix

Changed all parameter resolution calls to use `idx` instead of `node_id as usize` or `edge_id as usize`:

### In `convert_nodes()`:

Changed approximately 10 locations from:
```rust
self.resolve_string_param(&config.node_color, node_id as usize, &sanitized_attributes)
self.resolve_f64_param(&config.node_size, node_id as usize, &sanitized_attributes)
// etc.
```

To:
```rust
self.resolve_string_param(&config.node_color, idx, &sanitized_attributes)
self.resolve_f64_param(&config.node_size, idx, &sanitized_attributes)
// etc.
```

### In `convert_edges()`:

Changed approximately 8 locations from:
```rust
self.resolve_string_param(&config.edge_color, edge_id as usize, &sanitized_attributes)
self.resolve_f64_param(&config.edge_width, edge_id as usize, &sanitized_attributes)
// etc.
```

To:
```rust
self.resolve_string_param(&config.edge_color, idx, &sanitized_attributes)
self.resolve_f64_param(&config.edge_width, idx, &sanitized_attributes)
// etc.
```

## Files Changed

- `src/viz/realtime/accessor/realtime_viz_accessor.rs`

## Testing

Created validation test in `test_entity_id_validation.py` that:
1. Creates a graph with known entity IDs
2. Sets attributes with unique values
3. Verifies attributes are correctly stored and retrieved
4. Confirms the mapping is correct

Test output:
```
✓ Node 0: name='Node_0', value=0
✓ Node 1: name='Node_1', value=1
✓ Node 2: name='Node_2', value=2
✓ Node 3: name='Node_3', value=3
✓ Node 4: name='Node_4', value=4

✓ Edge 0: label='Edge_0'
✓ Edge 1: label='Edge_1'
✓ Edge 2: label='Edge_2'
✓ Edge 3: label='Edge_3'

TEST PASSED: Entity IDs are correctly mapped
```

## Impact

This fix ensures that when using visualization parameters like:
```python
g.viz.show(
    layout='circular', 
    node_label='object_name', 
    edge_label='method_name', 
    node_color='object_name', 
    color_scale_type='categorical',
    node_size_range=(5, 20)
)
```

The attributes are correctly mapped from the graph's attribute storage to the visualization system, regardless of the actual entity ID values.

## Note on GraphDataSource

The `GraphDataSource` in `src/viz/streaming/graph_data_source.rs` was also examined. While it does sort node_ids and edge_ids (lines 32-34 and 62-63), this is correct behavior to ensure consistent ordering. The real issue was in how those IDs were being used as array indices in the accessor.
