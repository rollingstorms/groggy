# PageRank Optimization - Current Status

## Session Summary

Began work on optimizing the builder-based PageRank implementation which is currently **232-471x slower** than the native implementation (7.4s vs 0.032s on 50k nodes, 50.8s vs 0.108s on 200k nodes).

## Problem Identified

The builder version creates **1600+ step executions** for a single PageRank run:
- 16 primitive steps per iteration (mul, add, where, neighbor_agg, broadcast_scalar, etc.)
- × 100 iterations
- = 1600+ Python→Rust FFI crossings
- Each `neighbor_agg` rebuilds the CSR (19ms × 100 = ~2 seconds just for CSR construction!)

## Work Completed

### 1. Created PageRankIterStep (Single Iteration Primitive)

**File**: `src/algorithms/steps/centrality.rs`

Created a fused single-iteration step that:
- Takes current ranks and degrees as input
- Builds CSR once per iteration
- Performs power iteration math (sink handling, damping, teleport)
- Returns updated ranks + convergence metrics
- Mirrors native implementation logic

**Status**: ✅ Implemented and compiles successfully

**Registration**: ✅ Added to `src/algorithms/steps/registry.rs` as `core.pagerank_iter`

### 2. Module Integration

- ✅ Added `mod centrality` to `src/algorithms/steps/mod.rs`
- ✅ Exported `PageRankIterStep`
- ✅ Registered step in core registry

### 3. Compilation

✅ Rust library builds successfully:
```bash
cargo build --release --lib
# Finished `release` profile [optimized] target(s) in 24.79s
```

## Next Steps

### Option A: Complete Fused PageRank (Recommended)

Create `PageRankFusedStep` that runs **ALL iterations** inside a single Rust step:

```rust
pub struct PageRankFusedStep {
    rank_source: String,
    degree_source: String,
    rank_target: String,
    damping: f64,
    tolerance: f64,
    max_iter: usize,
}
```

**Benefits**:
- Only 3 total steps: init → degrees → fused_pagerank
- 3 FFI calls instead of 1600+
- CSR built once and reused for all iterations
- Convergence checking inside Rust
- Expected performance: 50-100ms (2-3x native) vs current 7400ms

**Implementation**:
1. Create `PageRankFusedStep` in `centrality.rs`
2. Copy native PageRank iteration logic (lines 282-345 of `pagerank.rs`)
3. Register as `core.pagerank_fused`
4. Update benchmark to use fused version
5. Validate results match native

### Option B: CSR Caching for Current Approach

If we want to keep the iteration flexibility:
- Add CSR caching to `NeighborAggregationStep`
- Detect when same graph/subgraph is used across iterations
- Reuse cached CSR
- Would reduce time from 7400ms → ~800ms (still 25x slower but tolerable)

## Files Modified

1. `src/algorithms/steps/centrality.rs` - NEW (212 lines)
2. `src/algorithms/steps/mod.rs` - Added centrality module + export
3. `src/algorithms/steps/registry.rs` - Added PageRankIterStep registration
4. `PAGERANK_OPTIMIZATION_PLAN.md` - NEW documentation

## Benchmark Results (Current)

From `benchmark_builder_vs_native.py`:

**50k nodes, 250k edges:**
- Native: 0.032s
- Builder: 7.427s
- Ratio: **232x slower**
- Max diff: 0.00006 (acceptable)

**200k nodes, 1M edges:**
- Native: 0.108s
- Builder: 50.769s
- Ratio: **471x slower**  
- Max diff: 0.00002 (acceptable)

## Recommendations

1. **Immediate** (1-2 hours):
   - Implement `PageRankFusedStep` (complete algorithm in one step)
   - Build Python extension: `maturin develop --release`
   - Update `benchmark_builder_vs_native.py` to use fused version
   - Validate performance improvement

2. **Follow-up** (future):
   - Apply same pattern to other iterative algorithms (HITS, eigenvector, LPA iterations)
   - Consider adding `core.power_iteration` as a generic primitive
   - Document when to use primitives vs fused algorithms

## Testing Commands

```bash
# Build Rust library
cargo build --release --lib

# Build Python extension  
cd /Users/michaelroth/Documents/Code/groggy
maturin develop --release

# Run benchmark
python benchmark_builder_vs_native.py

# Profile specific test
python benchmark_pr_profile.py
```

## Related Files

- `src/algorithms/centrality/pagerank.rs` - Native implementation (reference)
- `benchmark_builder_vs_native.py` - Performance benchmark
- `python-groggy/python/groggy/builder.py` - Builder DSL
- `notes/development/STYLE_ALGO.md` - Algorithm implementation patterns

## Key Insight

The builder is fundamentally designed for **composition**, not **performance**. For hot-path algorithms like PageRank, Betweenness, and Community Detection, we need **fused primitives** that execute entire algorithms (or expensive sub-algorithms) in a single Rust step. The fine-grained primitives (mul, add, where, etc.) are great for prototyping and custom logic, but production-quality algorithms should use fused steps.

This mirrors how NumPy works: you *can* write a loop with element-wise operations, but `np.dot()` is orders of magnitude faster because it's a fused primitive in C.
