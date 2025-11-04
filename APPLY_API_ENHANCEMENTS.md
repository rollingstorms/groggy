# Apply API Enhancements

## Summary

Added `persist` and `return_profile` parameters to the `apply()` method and `Pipeline` class to provide more control over algorithm execution and profiling capabilities.

## Changes Made

### 1. Python API Updates

**File: `python-groggy/python/groggy/pipeline.py`**
- Updated `apply()` function to accept `persist` and `return_profile` parameters
- Updated `Pipeline.run()` method to accept these parameters
- Updated `Pipeline.__call__()` to forward these parameters
- Added `Pipeline.last_profile()` method to retrieve the last execution profile
- Added `_last_profile` instance variable to store profile information

**File: `python-groggy/python/groggy/__init__.py`**
- Updated `_subgraph_apply()` wrapper to accept and forward `persist` and `return_profile` parameters

### 2. API Signature

```python
def apply(subgraph, algorithm_or_pipeline, persist=True, return_profile=False):
    """
    Apply an algorithm or pipeline to a subgraph.
    
    Args:
        subgraph: The subgraph to process
        algorithm_or_pipeline: Algorithm handle, list of handles, or Pipeline
        persist: Whether to persist algorithm results as attributes (default: True)
        return_profile: If True, return (subgraph, profile_dict); otherwise just subgraph (default: False)
    
    Returns:
        Processed subgraph (or tuple with profile if return_profile=True)
    """
```

### 3. Usage Examples

#### Basic usage (backward compatible)
```python
result = subgraph.apply(algorithm("community.connected_components", output_attr="cc"))
```

#### Get profiling information
```python
result, profile = subgraph.apply(
    algorithm("centrality.pagerank"),
    return_profile=True
)
print(f"Algorithm took {profile['run_time']:.6f}s")
```

#### Skip persistence for faster execution
```python
result = subgraph.apply(
    algorithm("community.connected_components"),
    persist=False  # Don't write attributes, just compute
)
```

#### Combined usage
```python
result, profile = subgraph.apply(
    [algo1, algo2, algo3],
    persist=False,
    return_profile=True
)
```

### 4. Profile Dictionary Structure

The profile dictionary returned when `return_profile=True` contains:

- `build_time`: Time to build the pipeline (seconds)
- `run_time`: Total execution time (seconds)
- `timers`: Dictionary of detailed timing breakdowns per algorithm step
- `subgraph_clone_time`: Time to clone the subgraph (seconds)
- `persist_results`: Boolean indicating if results were persisted
- `outputs`: Dictionary of algorithm-specific outputs

### 5. Performance Impact

Benchmark results on connected components algorithm show:

**With `persist=True` (default):**
- Similar performance to direct API calls
- Results written as node/edge attributes
- Overhead from attribute writes: ~1.2-2x compared to view API

**With `persist=False`:**
- Significantly faster: 0.07-0.34x of view API time
- No attribute writes
- Useful for temporary computations or when only the result structure matters

### 6. Backward Compatibility

All changes are fully backward compatible:
- Default `persist=True` maintains existing behavior
- Default `return_profile=False` returns only the subgraph (existing behavior)
- Existing code continues to work without modifications

### 7. Testing

- All existing pipeline tests pass
- Added support for `Pipeline.last_profile()` method (required by existing tests)
- Verified with `pytest tests -q -k "pipeline or apply"`: 41 passed, 1 skipped

### 8. Updated Benchmark

The `notes/development/benchmark_cc_view_vs_apply.py` script now demonstrates:
- Comparison of `view().connected_components()` vs `apply()` with both persistence modes
- Detailed profiling information from the pipeline execution
- Clear visibility into where time is spent (core algorithm vs attribute writes)

## Notes

- The Rust FFI layer (`python-groggy/src/ffi/api/pipeline.rs`) already supported `persist_results` parameter at line 148
- No Rust code changes were required
- The enhancement only involved exposing existing functionality through the Python API layer
