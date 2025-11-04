# Builder Primitives Debug Session - Stopping Point

**Date**: 2025-11-04  
**Status**: 🔴 Critical divergence in large graphs

## Problem Summary

The builder-based PageRank and LPA implementations work correctly on small graphs but diverge significantly on large graphs:

- **50k nodes**: PageRank max diff = 0.00006 (acceptable)
- **200k nodes**: PageRank max diff = 0.349 (❌ CRITICAL - 34.9% error!)
- LPA also shows different community counts

## What We Know Works

### Small Graph (5-node cycle)
Created a directed cycle (0→1→2→3→4→0) and verified:
- ✅ Native and builder both return 0.2 for all nodes after 1 iteration
- ✅ All intermediate values match (degrees, contributions, neighbor_sums, teleport)
- ✅ Mathematical correctness: 0.85 * 0.2 + 0.03 = 0.2

### Builder Implementation
The PageRank builder implementation (`build_pagerank_algorithm()` in benchmark_builder_vs_native.py) correctly implements:
1. Uniform initialization (1/N)
2. Out-degree computation
3. Safe reciprocal for division
4. Sink detection and handling
5. Neighbor aggregation (incoming edges)
6. Damping factor application
7. Teleport term
8. Sink redistribution

## Key Implementation Details

### Neighbor Aggregation Direction
- **Native PageRank**: Uses CSR with outgoing edges, PUSHES rank forward
  ```rust
  for neighbor_idx in csr.neighbors(idx) {
      next_rank[neighbor_idx] += contrib;  // Push to targets
  }
  ```

- **Builder PageRank**: Uses CSR with incoming edges (swapped at line 616), PULLS rank backward
  ```rust
  .map(|(source, target)| (target, source))  // Swap to build incoming
  // Then aggregates from IN-neighbors
  ```

**These are mathematically equivalent** - just different iteration styles.

### Degree Computation
- `NodeDegreeStep` (structural.rs:67-78) correctly counts OUT-degrees for directed graphs
- Counts edges where node is the source
- Matches native PageRank's degree calculation

## The Mystery: Why Large Graphs Diverge

Given that:
1. Small graphs work perfectly
2. The math is correct
3. The primitives are implemented correctly
4. Both approaches are mathematically equivalent

**Possible causes:**

### 1. Graph Structure Interpretation Issue
- The benchmark uses **undirected graphs** by default (line 92)
- Undirected edges become bidirectional
- Maybe native and builder handle undirected graphs differently?
- **Test**: Run benchmark with `Graph(directed=True)` instead

### 2. Floating Point Accumulation
- Small errors compound over 100 iterations
- But we're seeing 34.9% error, not 0.001% error
- Unlikely to be the root cause

### 3. CSR Building Differences
- Native uses `add_reverse_edges: false` with natural (source, target) ordering
- Builder swaps to (target, source) for incoming edges
- Maybe there's an edge case in how undirected edges are handled?

### 4. Normalization Issue
- Builder normalizes once after all iterations (line 61)
- Native normalizes... when? Need to check
- If native normalizes per-iteration, results could diverge

### 5. State Leak Between Runs
- We saw earlier that running tests in sequence caused issues
- Maybe variables aren't being cleared properly between runs?
- But benchmark creates new graphs each time...

### 6. Primitive Bug Under Scale
- One of the primitives (neighbor_agg, mul, add, etc.) might have a bug
- Bug only manifests with large data (e.g., HashMap iteration order?)
- Need to add debug logging to primitives

## Debugging Strategy

### Phase 1: Isolate Graph Type
```python
# Change line 92 in benchmark_builder_vs_native.py to:
graph = Graph(directed=True)
```
If this fixes it, the issue is in undirected graph handling.

### Phase 2: Check Normalization
Compare native vs builder:
- Does native normalize per-iteration or once at end?
- If per-iteration, change builder to match

### Phase 3: Add Intermediate Dumps
Modify builder to dump values after each iteration:
```python
with builder.iterate(max_iter):
    # ... existing code ...
    builder.attach_as(f"ranks_iter_{i}", ranks)  # Dump each iteration
```
Compare iteration-by-iteration with native.

### Phase 4: Profile Primitives
Add debug logging to:
- `neighbor_agg` - verify CSR correctness
- `normalize_sum` - verify normalization
- `reduce_scalar` - verify sink mass calculation

### Phase 5: Simplify to Minimal Repro
Create a medium-sized graph (1000 nodes) where divergence appears:
- Easier to debug than 200k
- Can dump full state
- Can compare native vs builder step-by-step

## Performance Note

Builder is **200x slower** than native PageRank:
- 50k: 9.4s vs 0.04s
- 200k: 49.8s vs 0.2s

This is expected since we're composing primitives vs hand-optimized code, but still worth optimizing later.

## Next Steps

**Priority 1: Fix Correctness**
1. Test with directed graphs
2. Check normalization strategy
3. Add iteration dumps
4. Create minimal repro

**Priority 2: Fix Performance**
- Only after correctness is established
- Profile hotspots
- Maybe compile primitives into fused operations

## Files Modified Today

- `benchmark_builder_vs_native.py` - updated PageRank/LPA implementations
- `debug_pr_detailed.py` - created for step-by-step debugging
- All primitives in `src/algorithms/steps/` - implemented and tested individually
- `python-groggy/python/groggy/builder.py` - alias resolution fixes

## Test Commands

```bash
# Run full benchmark
python benchmark_builder_vs_native.py

# Run small debug case
python debug_pr_detailed.py

# Run unit tests
pytest tests/test_builder_pagerank.py -q
pytest tests/test_builder_core.py -q
```

## References

- Native PageRank: `src/algorithms/centrality/pagerank.rs:290-330`
- Neighbor aggregation: `src/algorithms/steps/aggregations.rs:576-680`
- Node degrees: `src/algorithms/steps/structural.rs:14-88`
- Builder PageRank: `benchmark_builder_vs_native.py:12-64`
