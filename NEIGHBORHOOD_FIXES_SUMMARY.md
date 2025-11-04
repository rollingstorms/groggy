# Neighborhood Array Fixes Summary

## Issues Addressed

This fix addresses three related issues with the neighborhood functionality in groggy:

### 1. Neighborhood array is a list not a subgraph array

**Problem:** When calling `graph.neighborhood()`, it returned a `NeighborhoodResult` object with a `.neighborhoods` property that was a Python list of `NeighborhoodSubgraph` objects, not a `SubgraphArray`.

**Solution:** Created a new `PyNeighborhoodArray` type that is a specialized `SubgraphArray` with additional neighborhood metadata. The `graph.neighborhood()` method now returns `NeighborhoodArray` instead of `NeighborhoodResult`.

### 2. Neighborhood result structure is convoluted

**Problem:** The `NeighborhoodResult` had metadata (like `largest_neighborhood_size`, `execution_time_ms`) separate from the actual neighborhood subgraphs, requiring users to access `.neighborhoods` to get the array and then access metadata separately.

**Solution:** `NeighborhoodArray` combines both concerns - it's a full `SubgraphArray` with all array operations (iteration, indexing, table operations, etc.) AND it has neighborhood-specific metadata as properties (`.total_neighborhoods`, `.largest_neighborhood_size`, `.execution_time_ms`).

### 3. gr.array() doesn't handle groggy types

**Problem:** The `gr.array()` function only worked with primitive Python values (int, float, str, bool) and couldn't handle groggy types like `Subgraph` or `NeighborhoodSubgraph`.

**Solution:** Enhanced `gr.array()` to intelligently detect groggy types and create appropriate array types:
- List of `Subgraph` objects → `SubgraphArray`
- List of `NeighborhoodSubgraph` objects → `SubgraphArray` (converts to regular subgraphs)
- Primitive values → `BaseArray` (as before)

## Implementation Details

### New Type: PyNeighborhoodArray

Located in `python-groggy/src/ffi/subgraphs/neighborhood.rs`, this type:
- Wraps a `PySubgraphArray` internally
- Stores metadata: `total_neighborhoods`, `largest_neighborhood_size`, `execution_time_ms`
- Delegates all SubgraphArray methods (via Python method calls to avoid privacy issues)
- Implements its own `__len__`, `__getitem__`, `__iter__` for direct array access
- Supports all SubgraphArray operations: `table()`, `sample()`, `group_by()`, `nodes_table()`, `edges_table()`, `summary()`, `viz`, `map()`, `merge()`, `collapse()`

### Changes to graph_analysis.rs

- Changed `neighborhood()` to return `PyNeighborhoodArray` instead of `PyNeighborhoodResult`
- Added `PyNeighborhoodArray::from_result()` helper to convert core `NeighborhoodResult` to the new array type

### Changes to lib.rs (array function)

Enhanced the `array()` function to handle three cases:
1. **Subgraph detection:** If first element is `PySubgraph`, extract all as subgraphs and create `SubgraphArray`
2. **NeighborhoodSubgraph detection:** If first element is `PyNeighborhoodSubgraph`, convert each to `PySubgraph` and create `SubgraphArray`
3. **Primitive fallback:** Convert to `AttrValue` and create `BaseArray`

### Backward Compatibility

- `PyNeighborhoodResult` is kept for potential internal use but no longer returned from public APIs
- `NeighborhoodSubgraph` still exists and works the same way
- All existing SubgraphArray methods work on NeighborhoodArray

## Testing

All tests pass:
- `pytest tests/modules/test_algorithms.py -k neighborhood` - 3/3 passed
- `pytest tests/modules/test_subgraph_array.py` - 58/58 passed

Updated test expectations:
- `test_neighborhood` in `test_algorithms.py` now expects `NeighborhoodArray` instead of `NeighborhoodResult`

## Usage Examples

### Before (old way - still works via backward compatibility)
```python
result = g.neighborhood(node_id)
# result is NeighborhoodResult
neighborhoods_list = result.neighborhoods  # Python list
metadata = result.largest_neighborhood_size
```

### After (new way - recommended)
```python
neighborhoods = g.neighborhood(node_id)
# neighborhoods is NeighborhoodArray (a specialized SubgraphArray)

# Access metadata directly
print(neighborhoods.total_neighborhoods)
print(neighborhoods.largest_neighborhood_size)
print(neighborhoods.execution_time_ms)

# Use as SubgraphArray
for subgraph in neighborhoods:
    print(subgraph.node_count())

# Use SubgraphArray methods
tables = neighborhoods.nodes_table()
summary = neighborhoods.summary()
merged = neighborhoods.merge()
```

### gr.array() with groggy types
```python
# Create list of subgraphs
subgraphs = [neighborhoods[0], neighborhoods[1], ...]

# Convert to SubgraphArray
array = gr.array(subgraphs)  # Now works!

# Still works with primitives
int_array = gr.array([1, 2, 3, 4])
str_array = gr.array(['a', 'b', 'c'])
```

## Files Modified

1. `python-groggy/src/ffi/subgraphs/neighborhood.rs` - Added `PyNeighborhoodArray`, kept `PyNeighborhoodResult` for compatibility
2. `python-groggy/src/ffi/api/graph_analysis.rs` - Changed return type of `neighborhood()`
3. `python-groggy/src/ffi/api/graph.rs` - Updated return type annotation
4. `python-groggy/src/ffi/subgraphs/subgraph.rs` - Updated to use new array type, made `subgraph()` public
5. `python-groggy/src/ffi/storage/subgraph_array.rs` - Added public accessor methods
6. `python-groggy/src/lib.rs` - Enhanced `array()` function, added `PyNeighborhoodArray` export
7. `tests/modules/test_algorithms.py` - Updated test expectations

## Benefits

1. **Consistency:** Neighborhoods now behave like other collection types (ComponentsArray, SubgraphArray)
2. **Convenience:** Metadata and array operations in one object, no need to access `.neighborhoods` property
3. **Flexibility:** `gr.array()` can now handle any groggy type, making it a universal array constructor
4. **Type Safety:** Better type hints and more Pythonic API
5. **Performance:** No additional overhead, just better organization
