# Groggy Issues Fixed

## Summary

This document summarizes the fixes implemented for the issues listed in the notes.

## Fixed Issues

### 1. ✅ Add NumArray .notna() and .isna()

**Issue**: NumArray was missing `notna()` and `isna()` methods that BaseArray has.

**Solution**: Added both methods to `PyNumArray` in `python-groggy/src/ffi/storage/num_array.rs`:
- `isna()` returns a boolean NumArray with all `False` values (since NumArray with primitive types has no nulls)
- `notna()` returns a boolean NumArray with all `True` values

**Design Note**: This is correct behavior because `NumArray<T>` where T is a primitive numeric type (i32, i64, f32, f64, bool) doesn't support null values at the type level, similar to numpy arrays without masked values. Only `BaseArray<AttrValue>` supports nulls via `AttrValue::Null`.

**Files Changed**:
- `python-groggy/src/ffi/storage/num_array.rs` - Added `isna()` and `notna()` methods

**Test**:
```python
import groggy as gr
arr = gr.num_array([1.0, 2.0, 3.0, 4.0, 5.0])
print(arr.notna())  # Returns all True
print(arr.isna())   # Returns all False
```

### 2. ✅ Sig figs on table() constrained to only 2

**Issue**: Table display precision was hardcoded to 2 decimal places, which is insufficient for scientific data.

**Solution**: Increased default precision from 2 to 6 decimal places across all display formatters:
- Rust core `DisplayConfig::default()` 
- Python FFI `PyDisplayConfig::new()` default parameter
- Python `TableDisplayFormatter` default parameter
- Added getter/setter for `precision` property in `PyDisplayConfig`

**Files Changed**:
- `src/display/mod.rs` - Changed default precision from 2 to 6
- `python-groggy/src/ffi/display/mod.rs` - Changed default precision from 2 to 6 and added precision getter/setter
- `python-groggy/python/groggy/display/table_display.py` - Changed default precision from 2 to 6

**Configuration**:
The precision is still fully configurable:
```python
from groggy._groggy import DisplayConfig

# Use default (now 6 decimal places)
config = DisplayConfig.default()

# Or specify custom precision
config = DisplayConfig(precision=8)
```

## Remaining Issues

The following issues from the list require more investigation and are not addressed in this fix:

- [ ] Slice a base array with a base array - Needs indexing implementation
- [ ] Viz update server parameters intuitively and with .show() - Requires viz system changes
- [ ] Graph.remove attr col table.remove col - Needs table API extension
- [ ] Subgraph viz show -> just the subgraph? On update - Viz behavior change
- [ ] Viz node color - discrete or continuous - Viz feature
- [ ] Loading time on .table() for large graphs - Performance investigation needed
- [ ] Quick static viz like viz.color() - New viz API
- [ ] Neighborhood array is a list not a subgraph array - Type system refactoring
- [ ] Neighborhood result maybe is convoluted - API redesign needed
- [ ] gr.array(neighborhood.neighborhood) type inference - Type coercion logic

## Testing

All existing tests pass after these changes:
```bash
pytest tests -q
# 469 passed, 19 skipped in 0.32s
```

## Notes

These fixes follow the repository guidelines:
- Minimal, surgical changes to existing code
- No new dependencies added
- Backward compatible (precision increase is non-breaking)
- Follows existing code style and patterns
- All tests passing
