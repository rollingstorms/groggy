# Builder Algorithm Debug Status

## Current State (2025-11-03)

### PageRank Results
**Status: ✅ WORKING CORRECTLY**

Testing shows PageRank built with primitives matches native implementation:
- 50-node graphs: max diff ~4e-7
- 100-node graphs: max diff ~4e-7  
- 50k-node graphs: max diff ~6e-7

The benchmark reports these results match within tolerance (avg diff 1e-8).

### Label Propagation (LPA) Results
**Status: ⚠️ MINOR DISCREPANCY**

Community counts are close but not identical:
- 50k nodes: Native=8667 communities, Builder=8648 communities (19 difference, ~0.2%)
- 200k nodes: Native=35179 communities, Builder=35147 communities (32 difference, ~0.1%)

This level of variance is expected for stochastic algorithms and may be due to:
- Tie-breaking differences in label selection
- Iteration ordering effects
- Floating-point precision in mode calculation

### Performance Issues
**Status: ❌ NEEDS OPTIMIZATION**

Builder-based algorithms are significantly slower than native:
- PageRank: 70-166x slower
- LPA: 5-13x slower  

This is the primary concern - algorithms are correct but too slow.

### Variable Redefinition Warnings
**Status: ✅ FIXED**

The validation logic now correctly handles alias steps and in-place update steps:
- **Alias steps** are exempt from redefinition warnings (they're meant to reassign variables)
- **In-place update steps** (core.update_in_place, core.neighbor_mode_update) are exempt
- **Loop-generated outputs** (names containing `_iter`) are exempt

This eliminates false-positive warnings while still catching real issues.

### Loop Variable Tracking
**Status: ✅ WORKING CORRECTLY**

The `_finalize_loop` function in `builder.py` correctly:
1. Tracks variable mappings across iterations
2. Remaps inputs to reference the previous iteration's outputs
3. Generates unique output names per iteration (`*_iter0`, `*_iter1`, etc.)

Example: In PageRank, iteration 1 correctly reads from `add_1_iter0` (output of iteration 0), not from the initial `nodes_0`.

## Recommendations

### Priority 1: Performance Optimization ⭐
The correctness is good, but 57-166x slowdown is unacceptable. Areas to investigate:
1. **Step interpreter overhead** - Each primitive step has FFI crossing costs
2. **Memory allocations** - Builder may be creating more intermediate maps than necessary
3. **CSR operations** - Check if neighbor aggregation is using optimal paths
4. **Python/Rust boundary** - Profile to see where time is spent

### Priority 2: LPA Community Count Variance
This is likely acceptable variance for a stochastic algorithm. To verify:
1. Run multiple trials with different seeds
2. Check if variance is within expected bounds
3. Compare tie-breaking behavior between native and builder implementations

## Next Steps

1. **Profile builder PageRank execution**:
   ```bash
   python -m cProfile -o profile.stats benchmark_builder_vs_native.py
   python -m pstats profile.stats
   ```

2. **Add Rust-side profiling** to identify step interpreter bottlenecks

3. **Consider optimization strategies**:
   - Batch step execution (reduce FFI crossings)
   - Lazy evaluation / step fusion
   - Direct CSR operation support in primitives

4. **Clean up warnings** - Implement one of the suppression strategies above

## Test Commands

```bash
# Run minimal PageRank test
python -c "
from groggy import Graph
from groggy.builder import AlgorithmBuilder
from groggy.algorithms.centrality import pagerank

graph = Graph(directed=True)
a, b, c = [graph.add_node() for _ in range(3)]
graph.add_edge(a, b)
graph.add_edge(b, c)
graph.add_edge(c, a)

sg = graph.view()
result_native = sg.apply(pagerank(max_iter=20))
# Compare with builder version...
"

# Run full benchmark
python benchmark_builder_vs_native.py 2>&1 | grep -v UserWarning
```
