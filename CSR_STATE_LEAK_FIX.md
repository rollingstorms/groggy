# CSR State Leak Fix - Stopping Point

## Issue Summary

Builder-based PageRank and LPA algorithms were producing incorrect results due to a **cross-run state leak** in the CSR (Compressed Sparse Row) construction. Neighbor aggregation values were being doubled, tripled, or more with each successive algorithm execution.

### Symptoms

- Running multiple builder algorithms sequentially caused neighbor counts to accumulate
- `test_pr_directed_vs_undirected.py` showed:
  - First test: 2 edges → 2 neighbor entries ✓
  - Second test: 3 edges → 5 neighbor entries ✗ (should be 3)
  - Third test: 2 edges → 6 neighbor entries ✗ (should be 2)
- PageRank values diverged significantly from native implementation
- The issue only appeared when running multiple tests in sequence, not in isolation

## Root Cause

The thread-local `CsrScratch` buffer used by `build_csr_from_edges_with_scratch` in `src/state/topology/csr.rs` was not properly clearing the `degree` and `cursor` vectors when resizing them between builds.

### The Bug

In `CsrScratch::prepare()` (lines 38-52):

```rust
fn prepare(&mut self, node_count: usize) {
    if self.degree.len() != node_count {
        self.degree.resize(node_count, 0);  // BUG: resize doesn't clear old values when growing!
    } else {
        self.degree.fill(0);
    }
    // ... same pattern for cursor ...
}
```

**Problem:** When `resize()` grows a vector (e.g., from 3 to 4 elements), it only initializes the *new* elements to 0. Existing elements retain their old values, causing degree counts to accumulate across builds.

### Example of the Bug

1. Build 1 (3 nodes): `degree = [1, 1, 0]`
2. Build 2 (4 nodes): `degree.resize(4, 0)` → `[1, 1, 0, 0]` ← **old values preserved!**
3. Build 2 adds degrees → `[1+new, 1+new, 0+new, new]` ← accumulation!

## The Fix

Changed `CsrScratch::prepare()` to **always** clear before resizing:

```rust
fn prepare(&mut self, node_count: usize) {
    // Always clear and resize to ensure no stale data
    self.degree.clear();
    self.degree.resize(node_count, 0);
    
    self.cursor.clear();
    self.cursor.resize(node_count, 0);

    self.pairs.clear();
}
```

**Location:** `src/state/topology/csr.rs`, lines 38-46

This ensures the scratch buffers are completely reset between CSR builds, regardless of size changes.

## Results After Fix

### Multi-Graph Test
```
Test 1 (3 nodes, 2 edges): neighbor sums = [0.0, 0.333, 0.333] ✓
Test 2 (4 nodes, 3 edges): neighbor sums = [0.0, 0.083, 0.083, 0.083] ✓
Test 3 (3 nodes, 2 edges): neighbor sums = [0.0, 0.333, 0.333] ✓  (matches Test 1!)
```

### PageRank Accuracy (benchmark_builder_vs_native.py)

**Before fix:**
- 50k nodes: max diff ~0.0003
- 200k nodes: max diff ~0.0001+

**After fix:**
- 50k nodes: max diff **0.00006** (5x improvement!)
- 200k nodes: max diff **0.000022** (much better!)

### LPA Results
- Native: 3 communities
- Builder: 5 communities (was worse before, now closer)

## Remaining Issues

1. **PageRank Still Not Exact**
   - Max diff ~0.00006 is better but still above the tight tolerance requirement (0.0000005)
   - The builder algorithm correctly prevents accumulation, but there's still a mathematical difference
   - Need to investigate if the builder PageRank implementation matches all aspects of the native algorithm:
     - Sink node handling
     - Teleport term calculation
     - Normalization approach

2. **Performance**
   - Builder is 240-440x slower than native for PageRank
   - This is expected since we're rebuilding CSR every iteration vs native's optimized path
   - Consider caching CSR across iterations for builder algorithms

3. **LPA Community Count**
   - Still getting different community counts (3 vs 5)
   - The async update mechanism may need review

## Testing

Created and ran multiple validation tests:
- `test_csr_leak_simple.py` - single graph, multiple runs
- `test_multi_graph.py` - multiple different graphs (reproduced the bug)
- `test_pr_directed_vs_undirected.py` - comprehensive PageRank testing

All tests now pass consistently across sequential runs.

## Next Steps

1. **Investigate remaining PageRank divergence**
   - Compare builder vs native algorithm step-by-step
   - Check sink handling, degree weighting, and normalization
   - May need to adjust builder primitives or add missing steps

2. **Debug LPA community detection**
   - Verify async update mechanism
   - Check tie-breaking behavior
   - Ensure label propagation converges properly

3. **Performance optimization**
   - Consider CSR caching across loop iterations
   - Profile the builder pipeline execution
   - Identify other potential bottlenecks

4. **Add regression test**
   - Add `test_multi_graph.py` logic to test suite
   - Ensure CSR state isolation is tested automatically

## Files Modified

- `src/state/topology/csr.rs` - Fixed `CsrScratch::prepare()` to clear vectors before resizing
- `src/algorithms/steps/aggregations.rs` - No changes (diagnostic code removed)

## Commit Message

```
fix: CSR scratch buffer state leak causing neighbor aggregation accumulation

The thread-local CsrScratch buffer wasn't properly clearing degree/cursor
vectors when resizing between builds. Vector::resize() preserves existing
values when growing, causing degree counts to accumulate across successive
CSR builds. Changed prepare() to always clear before resize.

Fixes neighbor aggregation doubling/tripling in builder algorithms when
running multiple pipelines sequentially. PageRank max diff improved from
~0.0003 to ~0.00006 on 50k node graphs.
```
