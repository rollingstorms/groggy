# Builder PageRank State Leak Investigation — Stopping Point

## What We Did

Added atomic execution counter and diagnostics to track state leakage between builder pipeline runs.

### Changes Made

1. **src/algorithms/builder.rs**
   - Added `AtomicU64` execution counter to track each pipeline execution
   - Added optional debug output (via `GROGGY_DEBUG_PIPELINE` env var) to log:
     - Pipeline start/complete with execution ID
     - Variable count before/after each step
   - Allows tracking state across multiple runs

2. **src/algorithms/steps/core.rs**
   - Added `count()` method to `StepVariables` to expose the number of variables in scope

3. **Created debug scripts**
   - `debug_pagerank_state.py` — tests 3-node directed chain, multiple runs
   - `debug_pagerank_50nodes.py` — tests 50-node random graph

## Key Findings

### ✅ NO State Leakage Detected

Running the benchmark with `GROGGY_DEBUG_PIPELINE=1` shows:
```
[exec_0] Pipeline 'custom_pagerank' starting with 327 steps
[exec_0] Pipeline complete, final variables: 326
[exec_1] Pipeline 'custom_lpa' starting with 12 steps
[exec_1] Pipeline complete, final variables: 11
[exec_2] Pipeline 'custom_pagerank' starting with 327 steps
[exec_2] Pipeline complete, final variables: 326
[exec_3] Pipeline 'custom_lpa' starting with 12 steps
[exec_3] Pipeline complete, final variables: 11
```

- Each execution gets a fresh `StepVariables::default()` (line 128 of builder.rs)
- Variable counts are consistent across runs
- No ghost entries from previous executions

### ❌ PageRank Algorithm Still Incorrect

**3-node test (A→B→C):**
- Builder vs native diff: **3.87e-07** ✅ (within tolerance)
- Consistent across multiple runs (no state leakage)

**50-node random graph:**
- Max diff: **4.07e-02** ❌ (way above 5e-07 tolerance)
- Builder consistently underestimates high-rank nodes
- Example worst case:
  ```
  Node 9: native=0.061802, builder=0.021058, diff=0.040745
  ```

**200k-node graph (from benchmark):**
- Max diff: **0.287** ❌
- Still massive divergence on larger graphs

### ✅ LPA Working Well

From benchmark results:
- 50k nodes: 13,206 vs 13,201 communities (very close)
- 200k nodes: 35,179 vs 35,147 communities (very close)
- Top communities match almost perfectly in size and membership
- **No state leakage issues**

## Root Cause Analysis

The "state leakage" hypothesis was **incorrect**. The actual problem is **algorithmic correctness** in the PageRank builder implementation.

### Why Small Graphs Work But Large Graphs Fail

The 3-node directed chain works fine because it's a trivial case where the algorithm can't accumulate much error. As graph size and connectivity increase, the error compounds across iterations.

### Suspect Areas

1. **Sink node handling** — The sink redistribution logic may be incorrect
2. **Neighbor aggregation** — `core.neighbor_agg` might not be computing the right neighbor contributions
3. **Degree weighting** — The `core.recip` and subsequent multiplication may have edge cases
4. **Normalization timing** — We normalize after each iteration AND at the end; this might cause issues

## Next Steps

### P0: Debug PageRank Algorithm Logic

1. **Add per-iteration diagnostics**
   - Dump intermediate values (contrib, neighbor_sum, sink_mass) for a few nodes
   - Compare builder iteration 0 vs native iteration 0 to find where divergence starts

2. **Verify primitive correctness individually**
   - Test `core.neighbor_agg` in isolation with known inputs
   - Test `core.recip` edge cases (zero degrees, very small values)
   - Test `core.where` with boolean masks

3. **Simplify PageRank to minimal case**
   - Build a version without sink handling
   - Build a version without normalization per iteration
   - Gradually add complexity back until it breaks

4. **Cross-check with NetworkX**
   - Use NetworkX's PageRank on the same 50-node graph
   - Verify our native implementation matches NetworkX
   - Then fix builder to match both

### P1: Performance Optimization

Once correctness is established:
- Builder is 40-45x slower than native for PageRank
- Need to profile where the time goes (likely many small steps vs batched native code)

## Diagnostic Commands

```bash
# Enable pipeline execution tracking
GROGGY_DEBUG_PIPELINE=1 python benchmark_builder_vs_native.py

# Test specific graph sizes
python debug_pagerank_state.py        # 3-node chain
python debug_pagerank_50nodes.py      # 50-node random
```

## Conclusion

The investigation **ruled out state leakage** as the cause of PageRank divergence. The StepVariables are properly isolated per execution, and each pipeline starts fresh. The real issue is that the **PageRank algorithm implementation itself is incorrect** for non-trivial graphs.

Focus should shift to:
1. Debugging the PageRank algorithm logic step-by-step
2. Verifying each primitive works correctly in isolation
3. Finding where the builder and native implementations diverge mathematically

---
**Status**: Investigation complete, pivoting to algorithm correctness debugging  
**Date**: 2025-11-03  
**Files Modified**: 
- `src/algorithms/builder.rs` (atomic counter + diagnostics)
- `src/algorithms/steps/core.rs` (count() method)
- Created: `debug_pagerank_state.py`, `debug_pagerank_50nodes.py`
