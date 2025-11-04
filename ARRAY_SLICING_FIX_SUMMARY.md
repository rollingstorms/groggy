# Array Slicing with Arrays Fix Summary

## Issue Addressed

**Problem:** BaseArray and NumArray could only be sliced with Python primitives (integers, lists, slices), but not with other groggy arrays. This prevented useful patterns like fancy indexing and boolean masking using groggy's own array types.

**Solution:** Enhanced the `python_index_to_slice_index` function to recognize and handle BaseArray and NumArray as index arguments, enabling full NumPy-style fancy indexing within groggy.

## Implementation

### Changes Made

**File: `python-groggy/src/ffi/utils/indexing.rs`**
- Enhanced `python_index_to_slice_index()` to detect BaseArray and NumArray
- For NumArray:
  - Bool dtype → converted to `SliceIndex::BoolArray` for boolean masking
  - Numeric dtypes → converted to `SliceIndex::List` for fancy indexing
- For BaseArray:
  - Checks if all values are integers → `SliceIndex::List` for fancy indexing
  - Checks if all values are booleans → `SliceIndex::BoolArray` for boolean masking
  - Returns error if mixed types

**File: `python-groggy/src/ffi/storage/num_array.rs`**
- Added public helper methods:
  - `to_int64_vec()` - Convert any numeric array to Vec<i64> for indexing
  - `to_bool_vec()` - Convert any array to Vec<bool> for boolean masking

### Supported Indexing Patterns

#### 1. Integer Array Indexing (Fancy Indexing)
```python
arr = gr.array([10, 20, 30, 40, 50, 60, 70, 80])
indices = gr.array([0, 2, 4, 6])
result = arr[indices]  # [10, 30, 50, 70]
```

#### 2. Boolean Masking with BaseArray
```python
arr = gr.array([10, 20, 30, 40, 50])
mask = gr.array([True, False, True, False, True])
result = arr[mask]  # [10, 30, 50]
```

#### 3. NumArray with Integer Indices
```python
arr = gr.num_array([1.5, 2.5, 3.5, 4.5, 5.5])
indices = gr.num_array([0, 2, 4])
result = arr[indices]  # [1.5, 3.5, 5.5]
```

#### 4. NumArray with Boolean Mask
```python
arr = gr.num_array([10, 20, 30, 40, 50])
mask = gr.bool_array([True, True, False, False, True])
result = arr[mask]  # [10, 20, 50]
```

#### 5. Conditional Indexing (Most Powerful!)
```python
arr = gr.num_array([10, 25, 30, 15, 50, 5])
# Create boolean mask from condition
mask = arr > 20
result = arr[mask]  # [25, 30, 50]
```

#### 6. Negative Indices
```python
arr = gr.array([10, 20, 30, 40, 50])
indices = gr.array([0, -1, -2])
result = arr[indices]  # [10, 50, 40]
```

### Type Conversions

The implementation handles automatic type conversions:
- **NumArray** numeric types (int32, int64, float32, float64) → converted to i64 for indexing
- **NumArray** bool type → used directly for boolean masking
- **BaseArray** with Int/SmallInt values → extracted as i64 for indexing
- **BaseArray** with Bool values → extracted for boolean masking

## Benefits

1. **NumPy-style Operations:** Enables familiar data analysis patterns from NumPy/Pandas
2. **Conditional Filtering:** Natural syntax for filtering: `arr[arr > threshold]`
3. **Consistency:** Both BaseArray and NumArray support the same indexing operations
4. **Performance:** Direct conversion from internal representations without Python round-trip
5. **Type Safety:** Clear error messages for invalid indexing patterns

## Testing

All test scenarios pass:
- Integer array indexing with BaseArray ✓
- Boolean masking with BaseArray ✓
- Integer array indexing with NumArray ✓
- Boolean masking with NumArray ✓
- Conditional filtering (arr > value) ✓
- Negative indices in arrays ✓

## Comparison with Original Behavior

### Before
```python
arr = gr.num_array([10, 25, 30, 15, 50, 5])
mask = arr > 20  # Returns boolean NumArray

# Had to convert to Python list
result = arr[list(mask.to_list())]  # Awkward!
```

### After
```python
arr = gr.num_array([10, 25, 30, 15, 50, 5])
mask = arr > 20  # Returns boolean NumArray

# Works directly!
result = arr[mask]  # Clean and natural!
```

## Implementation Notes

- The `SliceIndex` enum in the core library already supported both integer lists and boolean arrays, so no core changes were needed
- The FFI layer now properly recognizes groggy array types and converts them to the appropriate `SliceIndex` variant
- Error handling provides clear messages when array types don't match expected patterns (e.g., mixing integers and booleans)
- Conversion methods are efficient, working directly with internal storage rather than going through Python

## Files Modified

1. `python-groggy/src/ffi/utils/indexing.rs` - Enhanced index detection and conversion
2. `python-groggy/src/ffi/storage/num_array.rs` - Added public conversion methods

## Related Features

This fix complements the neighborhood array fixes, as both enable more Pythonic and natural APIs:
- Neighborhood fixes: Made neighborhoods behave as proper arrays
- Array slicing fixes: Made arrays sliceable with other arrays

Together, these enable powerful data analysis workflows entirely within groggy's type system.
