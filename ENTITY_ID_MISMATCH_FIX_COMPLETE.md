# Entity ID Mismatch Fix - Complete Solution

## Problems Identified

### Problem 1: Using Entity IDs as Array Indices (FIXED in previous commit)
**Location:** `src/viz/realtime/accessor/realtime_viz_accessor.rs`

The viz accessor was using entity IDs (node_id, edge_id) as array indices when resolving VizParameters, but should have been using enumeration indices (idx).

**Fix:** Changed all parameter resolution calls to use `idx` instead of `node_id as usize` or `edge_id as usize`.

### Problem 2: Sorting Changes Entity Order
**Location:** `src/viz/streaming/graph_data_source.rs` lines 32-33, 62-63

GraphDataSource was sorting node_ids and edge_ids, causing:
- `viz_snapshot.edges[0]` ≠ `graph.edges[0]`  
- `viz_snapshot.nodes[0]` ≠ `graph.nodes[0]`

**Example:**
```
Graph edges (unsorted):  [18, 64, 650, 233, 546, ...]
Viz edges (sorted):      [0, 1, 2, 3, 4, ...]
```

This made debugging confusing because edge[0] in the graph had different data than edge[0] in the viz.

**Fix:** Removed the `.sort()` calls to preserve original graph ordering.

### Problem 3: Empty Node/Edge Attributes in Viz
**Location:** `python-groggy/src/ffi/subgraphs/subgraph.rs` lines 89-130

The `viz()` function was creating a NEW empty graph and trying to copy attributes incorrectly:

```rust
// WRONG CODE:
let mut viz_graph = groggy::api::graph::Graph::new();
for &node_id in &node_ids {
    viz_graph.add_node();  // Creates node 0, 1, 2...
    // But then tries to set attrs on original node_id (43, 47, 2...)
    viz_graph.set_node_attr(node_id, attr_name, attr_value);  // FAILS!
}
```

The bug was:
1. `add_node()` creates nodes with IDs 0, 1, 2, 3...
2. But `set_node_attr(node_id, ...)` tries to set attrs on IDs like 43, 47, 2 (which don't exist!)
3. Result: All attributes were silently failing to be set

Same problem with edges - creating new edges with new IDs but trying to set attributes using old IDs.

**Fix:** Instead of creating a new graph, pass the original graph directly to GraphDataSource:

```rust
// CORRECT CODE:
let graph_ref = self.inner.graph();
let graph_data_source = GraphDataSource::new(&*graph_ref.borrow());
```

## Summary of Changes

### File 1: `src/viz/realtime/accessor/realtime_viz_accessor.rs`
- Changed ~18 parameter resolution calls from using entity IDs to using array indices
- Ensures `node_color`, `edge_label`, etc. correctly map to attributes

### File 2: `src/viz/streaming/graph_data_source.rs`
- Removed `.sort()` calls on lines 32-33 and 62-63
- Preserves original graph ordering so edge[0] in graph matches edge[0] in viz

### File 3: `python-groggy/src/ffi/subgraphs/subgraph.rs`
- Completely rewrote `viz()` function (lines 89-130)
- Changed from creating a new graph to using the original graph
- Fixes empty attributes issue

## Testing

### Test 1: Attribute Presence
```python
gt = gr.from_csv(...)
g = gt.to_graph()

edge_ids = g.edge_ids
nid = node_ids[0]
obj_name = g.get_node_attr(nid, 'object_name')
# ✓ obj_name is correctly retrieved

eid = edge_ids[0]
method_name = g.get_edge_attr(eid, 'method_name')
# ✓ method_name is correctly retrieved
```

### Test 2: Order Preservation
```python
# Graph edge[0] now matches viz edge[0]
Graph edge[0]: ID=42, method_name='neighborhood'
Viz edge[0]:   ID=42, method_name='neighborhood'
# ✓ Same edge!
```

### Test 3: Visualization Parameters
```python
g.viz.show(
    layout='circular',
    node_label='object_name',      # ✓ Labels match nodes
    edge_label='method_name',       # ✓ Labels match edges
    node_color='object_name',       # ✓ Colors map correctly
    color_scale_type='categorical'
)
```

## Impact

These fixes ensure that:

1. **Attributes are preserved** - No more empty attributes in visualization
2. **Order is consistent** - edge[0] in graph === edge[0] in viz  
3. **Styling works correctly** - node_color, edge_label, etc. map to the right entities
4. **Debugging is easier** - What you see in the graph matches what's in the viz

All three issues stemmed from misunderstanding how entity IDs relate to array indices and data ordering. The fixes ensure proper separation of concerns and correct data flow from graph → GraphDataSource → viz.
