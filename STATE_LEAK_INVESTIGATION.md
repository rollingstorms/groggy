# Builder PageRank State Leakage Investigation

## Executive Summary

**Bug**: Builder-based PageRank gives completely wrong results on first execution after running a different algorithm, but correct results on second execution.

**Impact**: All builder algorithms are potentially affected. Tests fail when run in sequence but pass when run individually.

**Root Cause (Hypothesis)**: CSR adjacency cache is incorrectly shared or keyed, causing graph2's algorithm to use graph1's CSR structure on first run.

**Status**: Reproduced with minimal test case (single iteration). Ready for detailed instrumentation to identify exact cause.

## Problem Summary

When running builder-based PageRank tests in sequence, the second test shows incorrect values on its **first execution** but correct values on **second execution**. This suggests cached state is contaminating across test runs.

## Reproduction

```python
# Run test_builder_pagerank_basic (3-node cycle graph)
# Then run test_builder_pagerank_matches_native (5-node complex graph)
# Result: matches_native FAILS with ~3e-5 error on first run

# Run only test_builder_pagerank_matches_native alone
# Result: PASSES

# Run test_builder_pagerank_matches_native twice on same graph
# Result: First run FAILS, second run PASSES
```

## Test Script Confirmation

`debug_cross_graph_contamination.py` confirms:
- algo1 on graph1 (3 nodes) → correct
- algo2 on graph2 (5 nodes, 1st time) → **WRONG** (differs from native)
- algo2 on graph2 (5 nodes, 2nd time) → **CORRECT** (matches native)

## Investigation Steps Taken

### 1. Confirmed: `.view()` returns same instance
- `graph.view()` is cached and returns the same subgraph object
- This is expected behavior

### 2. Confirmed: No global mutable state in algorithms
- Searched for `static mut`, `lazy_static`, `RefCell` in algorithms/
- Only found `GLOBAL_STEP_REGISTRY` which is immutable after init

### 3. Confirmed: `StepVariables` is fresh per execution
- `StepPipelineAlgorithm::execute()` creates `StepVariables::default()` at line 128 of builder.rs
- Each `.apply()` call gets a new `StepVariables` instance

### 4. Confirmed: CSR scratch buffer is properly cleared
- Found `thread_local! { static SCRATCH: RefCell<CsrScratch> ... }` in csr.rs:56
- But `CsrScratch::prepare()` clears all vectors
- `build_csr_from_edges_with_scratch()` also clears pairs after use

### 5. Confirmed: Subgraph topology_cache uses version checking
- CSR cache in subgraph is keyed by `(add_reverse, version)`
- Should invalidate when graph changes
- But we're using fresh graphs in each test!

### 6. Confirmed: Node degrees work correctly
- `debug_step_variable_leak.py` shows node_degree step gives correct results
- Both first and second runs are correct
- So the issue is NOT in basic CSR/degree computation

## Current Hypothesis

The contamination is specific to **iterative algorithms with 20+ iterations**. Possible causes:

1. **Subgraph CSR cache incorrectly shared**: Even though graphs are fresh, maybe the subgraph's `topology_cache` is being reused when it shouldn't be?

2. **Neighbor aggregation caching**: The `neighbor_agg` step builds a CSR each time (line 602 of aggregations.rs), but maybe some intermediate state persists?

3. **Normalization accumulation**: The `normalize_sum` step might be accumulating floating point errors differently on first vs subsequent runs?

4. **Loop unrolling artifact**: With 20 iterations, the builder creates 20 copies of each step. Maybe there's aliasing or variable reuse between iterations that breaks on first run?

## Confirmed Findings

### ✅ Issue reproduces with SINGLE iteration
- `debug_single_iteration_new.py` shows 7.37e-02 error on first run
- Second run passes with 7.45e-09 error (essentially perfect)
- **Rules out**: Iteration accumulation, loop unrolling artifacts

### ✅ Issue is NOT in node degrees
- `debug_step_variable_leak.py` shows node_degree step works correctly
- Both first and second runs give identical, correct results
- **Rules out**: CSR building, basic subgraph operations

### ✅ Issue is CROSS-GRAPH contamination
- Running algo1 on graph1, then algo2 on graph2 triggers the bug
- graph1 and graph2 are separate Graph instances
- **Rules out**: Subgraph-level caching (topology_cache is per-instance)

### ✅ Thread-local CSR scratch is properly cleared
- `CsrScratch::prepare()` zeroes all buffers
- `build_csr_from_edges_with_scratch()` clears pairs after use
- **Rules out**: CSR scratch contamination

### ✅ Python view caching exists but is per-graph
- `python-groggy/src/ffi/api/graph.rs` caches views per Graph instance
- Uses version checking (node_count + edge_count)
- **Rules out**: Cross-graph view caching

## Root Cause Hypothesis

The values on first run are **completely wrong** (not just accumulated error):
- Native PageRank: node 0 = 0.115, node 4 = 0.200
- Builder (1st run): node 0 = 0.189, node 4 = 0.132
- Builder (2nd run): node 0 = 0.115, node 4 = 0.200 ✓

This suggests the **CSR adjacency is being built incorrectly** on first run, then correctly on second run. Possible causes:

1. **CSR version checking bug**: The CSR cache might be returning stale data from graph1 when building CSR for graph2
2. **Subgraph ID collision**: If graph1's subgraph and graph2's subgraph have the same ID due to hashing collision
3. **Global state in neighbor iteration**: Something in how neighbors are iterated is contaminated

## Next Steps (Priority Order)

1. **[P0] Instrument CSR cache operations**
   - Add logging to `csr_cache_get()` and `csr_cache_store()` in subgraph.rs:398
   - Print subgraph_id, version, cache hit/miss for each lookup
   - Run `debug_single_iteration_new.py` with logging to see if cache is misused

2. **[P0] Check subgraph ID uniqueness**
   - Print `self.subgraph_id` for graph1.view() and graph2.view()
   - If they're the same, that's our bug (hash collision)
   - Solution: Use truly unique IDs (atomic counter) instead of content hashing

3. **[P1] Dump CSR structure on first vs second run**
   - Add logging to `NeighborAggregationStep` to dump CSR offsets/neighbors
   - Compare CSR for graph2 on first run (wrong) vs second run (correct)
   - This will show exactly what's different

4. **[P1] Check if view() is the trigger**
   - Modify test to not call `graph1.view()` before `graph2.view()`
   - If that fixes it, the issue is in view creation side effects

5. **[P2] Bisect the algorithm steps**
   - Create minimal repro with just: init + degrees + neighbor_agg
   - Find the exact step where values diverge

## Key Files

- `src/algorithms/builder.rs:128` - StepVariables creation
- `src/algorithms/steps/aggregations.rs:587` - NeighborAggregationStep::apply
- `src/subgraphs/subgraph.rs:405` - CSR cache lookup
- `src/state/topology/csr.rs:56` - Thread-local scratch buffer
- `tests/test_builder_pagerank.py` - Failing tests

## Tolerance Issue

The test uses `assert abs(pr_builder - pr_native) < 1e-6` but failures show ~3e-5 difference. The user mentioned wanting tolerance of `5e-7` which is very tight. This suggests we're not just dealing with floating point accumulation but actual algorithmic difference on first run.
