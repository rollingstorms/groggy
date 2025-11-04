# Profiling Feature Implementation Complete

## Summary

Added comprehensive per-step profiling capabilities to the builder pipeline system. The infrastructure was already present in the Rust Context struct; this work exposes it conveniently through Python.

## Changes Made

### 1. Python API (`python-groggy/python/groggy/pipeline.py`)

Added `print_profile()` function that formats and displays profiling data:

```python
def print_profile(profile: dict, show_steps: bool = True, show_details: bool = False):
    """
    Pretty-print profiling information from a pipeline run.
    
    Args:
        profile: Profile dictionary returned from pipeline.run(..., return_profile=True)
        show_steps: If True, show per-step timings (default: True)
        show_details: If True, show detailed call counters and stats (default: False)
    """
```

**Features:**
- Overall timing breakdown (build vs run time)
- Per-step timing with percentage of total runtime
- Optional detailed call counters (function-level profiling)
- Optional statistics from Context

### 2. Export in Main Module (`python-groggy/python/groggy/__init__.py`)

Exported `print_profile` from the main `groggy` module for easy access:

```python
from groggy import print_profile
```

### 3. Integration with Benchmark Script

Updated `benchmark_builder_vs_native.py` to support profiling:

- Added `SHOW_PROFILING` flag at the top (set to `False` by default)
- Integrated `print_profile()` calls for both native and builder algorithms
- Works with both `return_profile=True` on `apply()` calls

## Usage Examples

### Basic Usage

```python
import groggy as gg

# Create graph and algorithm
g = gg.Graph(directed=True)
g.add_edges([(0, 1), (1, 2), (2, 0)])
sg = g.to_subgraph()

# Run with profiling
from groggy.algorithms import centrality
result, profile = sg.apply(
    centrality.pagerank(max_iter=10),
    return_profile=True
)

# Print profiling report
gg.print_profile(profile)
```

### With Builder Algorithms

```python
from groggy import builder

b = builder("custom_algo")
nodes = b.init_nodes(default=1.0)
doubled = b.core.mul(nodes, 2.0)
b.attach_as("result", doubled)

algo = b.build()
result, profile = sg.apply(algo, return_profile=True)

# Show detailed profiling with call counters
gg.print_profile(profile, show_details=True)
```

### In Benchmarks

```python
# Enable profiling in benchmark_builder_vs_native.py
SHOW_PROFILING = True  # Set at top of file

# Profiling output will be shown for each algorithm run
```

## Output Format

```
================================================================================
Pipeline Profiling Report
================================================================================

Total Time:                         1.605 ms
  Build:                            1.074 ms ( 66.9%)
  Run:                              0.531 ms ( 33.1%)

--------------------------------------------------------------------------------
Per-Step Timings
--------------------------------------------------------------------------------
Step                                                  Time (ms)   % of Run
--------------------------------------------------------------------------------
[0] centrality.pagerank                                   0.518       97.6%
[1] core.init_nodes                                       0.034        5.9%
[2] core.mul                                              0.000        0.1%
================================================================================
```

With `show_details=True`, additional sections show:
- Detailed call counters (calls, total time, average time)
- Statistics recorded by algorithms

## Testing

Created test scripts:
- `test_profiling.py` - Basic profiling functionality test
- `test_benchmark_profiling.py` - Integration with benchmark-style workflows

Both tests demonstrate:
✅ Native algorithm profiling
✅ Builder algorithm profiling  
✅ Pipeline profiling with multiple algorithms
✅ Per-step timing breakdown

## Next Steps for Debugging

The profiling infrastructure is now complete and working. The remaining issue in `benchmark_builder_vs_native.py` is the PageRank algorithm divergence and LPA state leak, which are separate algorithmic correctness issues to debug:

1. **PageRank normalization failure** - The builder PageRank hits "cannot normalize: total magnitude below epsilon" which suggests the rank distribution is degenerating during iteration
2. **LPA state leak** - The warning about redefining variables suggests the loop unrolling isn't correctly isolating iteration state

These can now be debugged more effectively using the profiling data to identify which specific steps are problematic.

## Files Modified

- `python-groggy/python/groggy/pipeline.py` - Added `print_profile()` function
- `python-groggy/python/groggy/__init__.py` - Exported `print_profile`  
- `benchmark_builder_vs_native.py` - Integrated profiling support

## Files Created

- `test_profiling.py` - Basic profiling test
- `test_benchmark_profiling.py` - Benchmark-style profiling test
- `PROFILING_FEATURE_COMPLETE.md` - This document
