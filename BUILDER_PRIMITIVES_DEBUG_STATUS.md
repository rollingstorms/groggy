# Builder Primitives Debugging Status

**Date**: 2025-11-04  
**Context**: Debugging PageRank and LPA builder implementations after completing Phase 1-3 primitive additions

---

## Summary

Completed all primitive implementations from the plan (recip, compare, where, reduce_scalar, broadcast_scalar, neighbor_agg, collect_neighbor_values, mode, update_in_place, histogram, clip, auto-scalar detection, graph constants). Fixed the benchmark graph creation bug (was adding duplicate edges in undirected graphs). PageRank now works well on 50k nodes but shows significant divergence on one or more outlier nodes in the 200k graph. LPA has minor community count differences and produces warnings about variable redefinition.

---

## Current Status

### ✅ Fixed Issues

1. **Graph Creation Bug** (benchmark_builder_vs_native.py:87-111)
   - **Problem**: Undirected graphs had both `A->B` and `B->A` added explicitly, doubling all degrees
   - **Fix**: Changed to add only one edge per pair in undirected graphs
   - **Result**: PageRank convergence improved dramatically on 50k graph

2. **Native PageRank Output** (benchmark_builder_vs_native.py:144)
   - **Problem**: Print statement was missing the actual value
   - **Fix**: Added `{native_val:.6f}` to format string

3. **Tolerance Adjustment** (benchmark_builder_vs_native.py:189-194)
   - **Problem**: Tolerance was too lenient (1e-6)
   - **Fix**: Changed to stricter tolerance (5e-7) with graduated warnings

### ⚠️ Remaining Issues

#### 1. PageRank Outlier Divergence on Large Graphs

**Benchmark Results:**
- **50k nodes**: max diff = 0.00006 (acceptable, avg = 0.000005)
- **200k nodes**: max diff = 0.349 (unacceptable, avg = 0.0000046)

**Symptoms:**
- Average difference is excellent (<0.000005) on both graph sizes
- One or more outlier nodes have huge divergence in 200k graph
- Small test cases (3-node chains, cycles) work perfectly
- 1000-node test shows systematic under-counting for high-PageRank nodes (builder values ~27-31% lower)

**Hypothesis:**
The issue appears when there are specific topological patterns (hubs, dangling nodes, or sink clusters). The builder implementation may be:
- Incorrectly handling certain edge configurations in the CSR
- Missing contributions from specific neighbor patterns
- Having a numerical stability issue that amplifies with graph size

**Debug Evidence:**
- `debug_pr_divergence.py`: Small graphs work perfectly
- `debug_pr_200k_sample.py`: 1000-node graph shows high-degree nodes systematically lower
- Both implementations normalize to sum=1.0, so it's not a global normalization issue

**Next Steps:**
1. Instrument `neighbor_agg` to log CSR structure for outlier nodes
2. Add detailed logging in PageRank builder to track rank propagation per iteration
3. Create a targeted test that reproduces the 200k outlier pattern on a smaller graph
4. Check if the issue is related to node ID distribution (dense vs sparse indexing)
5. Verify CSR endpoint swapping is consistent between native and builder

#### 2. LPA Variable Redefinition Warnings

**Warning Output:**
```
/Users/michaelroth/Documents/Code/groggy/python-groggy/python/groggy/builder.py:1119: UserWarning: 
Pipeline validation: Step 3 (core.update_in_place): redefines variable 'nodes_0'
Pipeline validation: Step 6 (core.update_in_place): redefines variable 'nodes_0_iter0'
Pipeline validation: Step 9 (core.update_in_place): redefines variable 'nodes_0_iter1'
...
```

**Problem:**
- The `_finalize_loop` alias resolution is creating iteration-specific variable names
- `update_in_place` then redefines these variables, triggering validation warnings
- Despite warnings, LPA runs and produces results close to native (community counts off by 1-2)

**Root Cause:**
- Loop unrolling in builder.py:883 (`_finalize_loop`) creates aliases like `nodes_0_iter0`, `nodes_0_iter1`
- The `core.neighbor_mode_update` step uses `update_in_place` semantics
- Variable redefinition detection sees the same variable being written multiple times

**Next Steps:**
1. Review `_finalize_loop` alias bookkeeping logic
2. Ensure iteration i reads from the output of iteration i-1, not the original map
3. Consider whether `update_in_place` should be marked as exempt from redefinition warnings
4. Test if the community count difference is related to the alias issue or a separate convergence difference

#### 3. Performance: Builder 234-260x Slower than Native (PageRank)

**Benchmark:**
- 50k nodes: Builder 7.5s vs Native 0.032s = **234x slower**
- 200k nodes: Builder 57s vs Native 0.219s = **260x slower**

**LPA is better:**
- 50k: 3.65x slower
- 200k: 3.76x slower

**Analysis:**
PageRank builder creates many intermediate primitives per iteration:
- `core.mul` (contrib = ranks * inv_degrees)
- `core.where` (zero out sinks)
- `core.neighbor_agg` (sum neighbor contributions)
- `core.mul` (apply damping)
- `core.broadcast_scalar` (teleport term)
- `core.mul` (compute teleport map)
- `core.where` (sink mask)
- `core.reduce_scalar` (sink mass)
- `core.mul` (sink per-node)
- `core.mul` (damped sinks)
- `core.add` (combine components) x2
- `core.normalize_sum` (after loop)

Each primitive involves:
- HashMap lookups/insertions
- Iteration over all nodes
- Variable management overhead

Native PageRank does all of this in tight loops with pre-allocated buffers.

**Mitigation Options:**
1. Add a fused `pagerank_iteration` primitive that combines the math steps
2. Optimize primitive execution to reduce HashMap overhead
3. Add buffer reuse across primitives within an iteration
4. Profile to identify the hottest primitives and optimize those first

---

## Test Files Created

1. **debug_pr_divergence.py** - Tests simple 3-node topologies (chain, cycle, sink)
2. **debug_pr_200k_sample.py** - 1000-node test showing systematic under-counting
3. **debug_undirected_issue.py** - Verified edge duplication bug in undirected graphs

---

## Benchmark Configuration

**Current Settings:**
- 50k nodes, ~250k edges (avg degree ≈5)
- 200k nodes, ~1M edges (avg degree ≈5)
- PageRank: damping=0.85, max_iter=100, tolerance=5e-7
- LPA: max_iter=10

**Random Seed:** 42 (deterministic)

---

## Code Locations

### Builder Implementation
- **PageRank**: benchmark_builder_vs_native.py:12-64
- **LPA**: benchmark_builder_vs_native.py:67-84
- **Graph Creation**: benchmark_builder_vs_native.py:87-111

### Rust Primitives
- **neighbor_agg**: src/algorithms/steps/aggregations.rs:554-776
- **node_degree**: src/algorithms/steps/structural.rs:14-89
- **neighbor_mode_update**: src/algorithms/steps/community.rs (registered in transformations.rs)

### Native Algorithms
- **PageRank**: src/algorithms/centrality/pagerank.rs:290-330 (core iteration)
- **LPA**: src/algorithms/community/lpa.rs

### Builder Core
- **Loop Unrolling**: python-groggy/python/groggy/builder.py:883 (_finalize_loop)
- **Variable Management**: python-groggy/python/groggy/builder.py:1119 (validation warnings)

---

## Recommendations

### Immediate Priorities

1. **PageRank Outlier Investigation** [P0]
   - Create a minimal reproducer for the 200k outlier pattern
   - Add detailed logging/instrumentation to builder PageRank
   - Compare CSR structures between native and builder for affected nodes

2. **LPA Variable Warnings** [P1]
   - Fix `_finalize_loop` alias propagation
   - Verify iteration chaining is correct
   - Add test that validates multi-iteration algorithms don't leak state

3. **Performance Analysis** [P2]
   - Profile builder PageRank to identify bottlenecks
   - Consider adding fused primitives for hot paths
   - Measure per-primitive overhead

### Long-term Improvements

1. **Builder Optimization**
   - Buffer pooling/reuse across primitives
   - Lazy evaluation for primitive chains
   - Compiled pipeline execution mode

2. **Testing**
   - Add regression tests for various graph topologies
   - Automated tolerance validation
   - Performance benchmarks in CI

3. **Documentation**
   - Add builder best practices guide
   - Document primitive performance characteristics
   - Create examples for common algorithm patterns

---

## Related Documents

- **STYLE_ALGO.md** - Algorithm implementation best practices
- **PHASE1_CSR_COMPLETE.md** - CSR optimization work
- **BUILDER_COMPLETION_SUMMARY.md** - Earlier builder work

---

**Next Session Start Point:**

Begin by creating a targeted reproducer for the PageRank outlier issue. Look for:
- High-degree hubs in the 200k graph
- Sink nodes or weakly connected components
- Nodes with unusual in-degree/out-degree ratios

Then add instrumentation to both native and builder PageRank to compare:
- CSR neighbor lists for the divergent nodes
- Intermediate rank values per iteration
- Contribution sums from incoming neighbors

Once the root cause is identified, fix and re-run the full benchmark suite.
