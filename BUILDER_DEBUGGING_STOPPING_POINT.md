# Builder Debugging Stopping Point
**Date**: 2025-11-03  
**Context**: PageRank and LPA builder implementations showing correctness and performance issues

## Current State

### What Works
- ✅ PageRank **50k nodes**: Matches native within 6e-7 tolerance (values match!)
- ✅ All primitives implemented: `recip`, `compare`, `where`, `reduce_scalar`, `broadcast_scalar`, `neighbor_agg`, `neighbor_mode_update`, etc.
- ✅ Graph constants auto-detection working
- ✅ Basic pipeline execution functioning

### What's Broken

#### 1. PageRank Correctness on Larger Graphs [P0]
```
50k nodes:  Max diff = 0.00000063  ✅
200k nodes: Max diff = 0.28686176  ❌ (28% error!)
```

**Symptoms**:
- Works perfectly on 50k but diverges catastrophically on 200k
- Not a numerical precision issue (28% is huge)
- Suggests algorithmic bug that compounds with scale

**Likely Causes**:
a) **Degree computation bug** - `NodeDegreeStep` (line 64 of structural.rs) uses `graph_ref.out_degree()` instead of subgraph-aware degree counting
   - This means filtered/subgraph views get wrong degrees
   - Would explain scale-dependent failure (more degree mismatches on larger graphs)
   
b) **Alias resolution in loops** - The loop unrolling may still have subtle bugs where intermediate results aren't properly threaded between iterations

c) **Neighbor aggregation** - `neighbor_agg` might be double-counting or missing edges at scale

#### 2. LPA Community Count Mismatch [P1]
```
Native:  8667 communities
Builder: 8648 communities (19 fewer)
```

**Warnings Emitted**:
```
UserWarning: Pipeline validation: Step 3 (core.update_in_place): redefines variable 'nodes_0'
UserWarning: Pipeline validation: Step 6 (core.update_in_place): redefines variable 'nodes_0_iter0'
UserWarning: Pipeline validation: Step 9 (core.update_in_place): redefines variable 'nodes_0_iter1'
```

**Analysis**:
- The community count is close (99.8% match) but not exact
- Warnings show loop variable aliasing is triggering "redefines" warnings
- `update_in_place` semantics may not be matching native async LPA behavior
- Could be tie-breaking differences or iteration ordering issues

#### 3. Performance Issues [P2]
```
PageRank: 60-137x slower than native
LPA:      12x slower than native
```

**Contributing Factors**:
- Python/Rust FFI overhead per primitive call
- No CSR/columnar optimization in builder paths
- Multiple temporary allocations per iteration
- Suboptimal neighbor traversal in expressions

## Critical Fix Needed: Subgraph-Aware Degrees

### The Bug
**File**: `src/algorithms/steps/structural.rs:64`
```rust
let degree = graph_ref.out_degree(node).unwrap_or(0);
```

This uses the **parent graph's degree**, not the **subgraph's filtered degree**.

### Why This Breaks PageRank
- PageRank divides rank by out-degree: `contrib = rank[i] / out_degree[i]`
- If we use parent graph degrees on a filtered subgraph, the division is wrong
- Wrong degrees → wrong contributions → wrong ranks
- Compounds over iterations → catastrophic divergence

### The Fix
Revert to subgraph-aware degree computation:
```rust
let degree = subgraph.degree(node).unwrap_or(0);
```

Or if filtering is needed:
```rust
let degree = subgraph.neighbors(node)
    .map(|neighbors| neighbors.count())
    .unwrap_or(0);
```

### Why This Was Changed
Previous debugging sessions noted degree issues and tried using `graph_ref.out_degree()` directly, but this broke the subgraph contract. The builder's `node_degrees()` **must** respect the active subgraph view.

## Recommended Next Steps

### Phase 1: Fix Correctness [P0]

1. **Fix `NodeDegreeStep` subgraph awareness** (30 min)
   - Revert to `subgraph.degree(node)` or equivalent
   - Add test: `test_node_degree_on_filtered_subgraph`
   - Verify degrees match subgraph edge count

2. **Validate PageRank on 200k** (15 min)
   - Rebuild: `maturin develop --release`
   - Re-run benchmark: `python benchmark_builder_vs_native.py`
   - Target: Max diff < 1e-5 on all graph sizes

3. **Debug LPA variable aliasing** (45 min)
   - Trace why `update_in_place` triggers "redefines" warnings
   - Check if loop unrolling is creating correct alias chains
   - Verify async update semantics match native implementation
   - Add logging to `neighbor_mode_update` to see actual label propagation

### Phase 2: Performance [P2]

Only after correctness is 100% validated:

1. Profile builder PageRank with `py-spy` or similar
2. Identify hottest primitive calls
3. Consider batching or CSR-aware fast paths
4. Target: <10x slowdown vs native (acceptable for DSL)

### Phase 3: Testing & Validation

1. Add regression tests:
   - `test_pagerank_builder_large_graph` (200k nodes)
   - `test_lpa_builder_matches_native_community_count`
   - `test_node_degree_respects_subgraph_filter`

2. Benchmark suite improvements:
   - Add more graph topologies (sparse, dense, power-law)
   - Test on directed vs undirected
   - Validate with known ground truth (Zachary Karate Club, etc.)

## Key Insights

### Degree Computation is Critical
Every centrality algorithm depends on accurate degree counting. Using parent graph degrees on subgraph views silently corrupts all downstream computations. This is the **most likely root cause** of the PageRank divergence.

### Alias Resolution Still Fragile
The loop unrolling and variable aliasing system is complex and warnings suggest it's still not quite right. The `_resolve_operand` fixes helped but didn't fully solve the aliasing problem in iteration contexts.

### Expression System vs Primitives
We successfully moved away from map_nodes expressions for PageRank/LPA, but the primitives themselves must be correct. A bug in `node_degrees` or `neighbor_agg` affects every algorithm that uses them.

### Scale Reveals Bugs
The 50k→200k jump exposed issues that weren't visible in small tests. Always benchmark at multiple scales to catch compound errors.

## Files to Investigate

**Immediate Priority**:
- `src/algorithms/steps/structural.rs:42-72` - NodeDegreeStep (THE MAIN SUSPECT)
- `benchmark_builder_vs_native.py:12-58` - PageRank builder implementation
- `python-groggy/python/groggy/builder.py:883-950` - Loop finalization and aliasing

**Secondary**:
- `src/algorithms/steps/transformations.rs:147-214` - update_in_place async logic
- `src/algorithms/steps/aggregation.rs` - neighbor_agg implementation
- `python-groggy/python/groggy/builder.py:1119` - Pipeline validation warnings

## Test Commands

```bash
# Rebuild
maturin develop --release --skip-install

# Run benchmark (quick check)
python benchmark_builder_vs_native.py

# Run specific tests
pytest tests/test_builder_pagerank.py -xvs
pytest tests/test_builder_lpa.py -xvs
pytest tests/test_builder_core.py::test_builder_node_degrees_directed_chain -xvs

# Check for warnings
python benchmark_builder_vs_native.py 2>&1 | grep -A2 "UserWarning"
```

## Success Criteria

Before closing this issue:

1. ✅ PageRank max diff < 1e-5 on ALL graph sizes (50k, 200k, larger)
2. ✅ LPA community counts match native exactly (±0)
3. ✅ No "redefines variable" warnings in pipeline validation
4. ✅ Regression tests added for both algorithms at scale
5. 🎯 Performance <20x slowdown acceptable for now (optimize later)

## Notes

- The degree bug is **high confidence** - this is almost certainly the root cause
- Don't optimize performance until correctness is perfect
- The primitive approach is sound, just need to fix the implementation bugs
- Consider adding more granular logging/debugging in primitives for future issues
