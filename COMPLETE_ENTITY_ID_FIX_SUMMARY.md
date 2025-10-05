# Complete Entity ID & Edge Mapping Fix

## Overview

Fixed **5 critical bugs** causing entity ID mismatches, edge source/target mapping errors, and non-deterministic behavior in the groggy visualization and CSV import system.

---

## Bug 1: Array Index Confusion in Viz Parameters ✅
**File:** `src/viz/realtime/accessor/realtime_viz_accessor.rs`

**Problem:** The viz accessor was using entity IDs (e.g., 43, 47, 2...) as array indices when resolving VizParameters from columns.

**Fix:** Changed ~18 parameter resolution calls to use enumeration index `idx` instead of `node_id as usize` or `edge_id as usize`.

**Impact:** Node/edge colors, labels, sizes now correctly map to attributes.

---

## Bug 2: Sorting Broke Entity Order ✅
**File:** `src/viz/streaming/graph_data_source.rs` lines 32-33, 62-63

**Problem:** GraphDataSource was sorting node_ids and edge_ids, causing `graph.edges[0]` ≠ `viz.edges[0]`.

**Fix:** Removed `.sort()` calls to preserve original graph ordering.

**Impact:** Edge/node ordering now consistent between graph and visualization.

---

## Bug 3: Empty Attributes in Visualization ✅
**File:** `python-groggy/src/ffi/subgraphs/subgraph.rs` lines 89-130

**Problem:** The `viz()` function created a new empty graph with nodes 0,1,2... but tried to set attributes using original node IDs (43,47,2...) which didn't exist, causing silent failures.

**Fix:** Pass the original graph directly to GraphDataSource instead of creating a new one.

**Impact:** All node/edge attributes now appear in visualization.

---

## Bug 4: Non-Deterministic Node ID Assignment ✅
**File:** `python-groggy/python/groggy/imports.py` line 165

**Problem:** Using `list(set(...))` for unique nodes meant iteration order was random (hash-based), so the same `object_name` got different node IDs on each run.

**Before:**
```
Run 1: InvalidInputError -> 2
Run 2: InvalidInputError -> 25  
Run 3: InvalidInputError -> 7
```

**Fix:** Changed to `sorted(set(...))` to ensure alphabetical, deterministic assignment.

**After:**
```
Run 1, 2, 3: InvalidInputError -> 2  (always the same)
```

**Impact:** Node IDs are now consistent across runs, enabling reliable CSV export/import.

---

## Bug 5: Edge Source/Target Mapping Mismatch ✅
**File:** `python-groggy/python/groggy/imports.py` lines 408-410

**Problem:** After creating expanded nodes with synthetic result_type nodes, the code called `_apply_node_id_mapping` again which RE-CREATED the mapping, destroying the carefully sorted order. This caused edges to connect to the wrong nodes!

**Before:**
```
Edge attributes: 'AttrName' -> 'unknown_return_type'
Edge endpoints: AggregationResult(38) -> unknown_return_type(37)
✗ MISMATCH!
```

**Root Cause:**
1. Initial mapping created with sorted nodes: `AttrName=0, BaseArray=1, ...`
2. Synthetic nodes added for missing result_types
3. `_apply_node_id_mapping()` called AGAIN, creating NEW mapping from CSV row order (not sorted)
4. Result: `DisplayConfig=0, GroggyError=1, ...` (wrong!)

**Fix:** Manually apply the `expanded_node_mapping` to expanded nodes instead of calling `_apply_node_id_mapping` again:

```python
# OLD (wrong - re-creates mapping):
expanded_nodes_mapped_dict = _apply_node_id_mapping(expanded_nodes_dict, "nodes", ...)

# NEW (correct - uses existing mapping):
expanded_nodes_mapped_dict = expanded_nodes_dict.copy()
if node_id_column in expanded_nodes_mapped_dict:
    expanded_nodes_mapped_dict["node_id"] = [
        expanded_node_mapping[str(node)] 
        for node in expanded_nodes_mapped_dict[node_id_column]
    ]
```

**After:**
```
Edge attributes: 'EdgesTable' -> 'unknown_return_type'
Edge endpoints: EdgesTable(10) -> unknown_return_type(37)
✓ MATCH!
```

**Impact:** Edges now correctly connect nodes based on their string identifiers.

---

## Verification

All issues resolved and verified:

```python
import groggy as gr

# Test 1: Deterministic IDs (3 runs)
✓ Node IDs are deterministic

# Test 2: Edge mappings (50 edges tested)
✓ All 50 tested edges map correctly

# Test 3: Attributes preserved
✓ All attributes preserved

# Test 4: Visualization works
g.viz.show(
    layout='circular',
    node_label='object_name',
    edge_label='method_name',
    node_color='object_name',
    color_scale_type='categorical'
)
✓ All parameters map correctly
```

---

## Files Modified

1. `src/viz/realtime/accessor/realtime_viz_accessor.rs` - Fixed array index confusion
2. `src/viz/streaming/graph_data_source.rs` - Removed sorting
3. `python-groggy/src/ffi/subgraphs/subgraph.rs` - Fixed viz() function
4. `python-groggy/python/groggy/imports.py` (line 165) - Made node IDs deterministic
5. `python-groggy/python/groggy/imports.py` (lines 408-410) - Fixed edge mapping

---

## Summary

These five interconnected bugs were causing a cascade of issues:

1. **Non-deterministic IDs** → Different mappings each run
2. **Re-mapped expanded nodes** → Destroyed sorted order
3. **Wrong edge endpoints** → Edges connected to wrong nodes
4. **Empty attributes** → No data in visualization
5. **Wrong viz parameters** → Colors/labels on wrong entities

All fixed! The system now works correctly end-to-end.
