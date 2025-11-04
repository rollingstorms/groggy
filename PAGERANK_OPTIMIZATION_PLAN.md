# PageRank Builder Optimization Plan

## Current Problem

The builder-based PageRank is **232-471x slower** than native (7.4s vs 0.032s on 50k nodes).

### Root Cause

The builder emits **16 primitive steps per iteration × 100 iterations = 1600 step executions**:
- Each step incurs Python→Rust FFI overhead
- CSR is rebuilt on every `neighbor_agg` call (19ms × 100 = 1900ms just for CSR)
- No convergence checking - always runs full 100 iterations
- Steps don't fuse - intermediate results are materialized

### Native Performance

Native PageRank (src/algorithms/centrality/pagerank.rs) achieves 32ms on 50k nodes by:
1. Building CSR once and caching it
2. Pre-allocating two rank buffers, swapping between them (O(1))
3. Tight inner loop with no allocations
4. Early convergence detection
5. All computation in Rust with no FFI crossings

## Solution: Fused PageRank Primitive

Create `core.pagerank_fused` - a complete PageRank algorithm as a single primitive step.

### Step Signature

```python
core.pagerank_fused(
    rank_source: str,      # Initial ranks (uniform)
    degree_source: str,    # Out-degrees
    rank_target: str,      # Final ranks
    damping: float = 0.85,
    tolerance: float = 1e-6,
    max_iter: int = 100
)
```

### Implementation (Rust)

```rust
pub struct PageRankFusedStep {
    rank_source: String,
    degree_source: String,
    rank_target: String,
    damping: f64,
    tolerance: f64,
    max_iter: usize,
}

impl Step for PageRankFusedStep {
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // 1. Extract input maps
        // 2. Build CSR once (or use cache)
        // 3. Run power iteration loop (all in Rust)
        //    - Buffer swapping
        //    - Convergence check
        //    - Early exit
        // 4. Return final ranks
    }
}
```

### Builder API

```python
def build_pagerank_optimized(damping=0.85, max_iter=100, tolerance=1e-6):
    builder = AlgorithmBuilder("pagerank_opt")
    
    # Initialization
    node_count = builder.graph_node_count()
    ranks = builder.init_nodes(default=1.0)
    inv_n = builder.core.recip(node_count, epsilon=1e-9)
    ranks = builder.core.broadcast_scalar(inv_n, ranks)
    
    # Degrees
    degrees = builder.node_degrees(ranks)
    
    # Fused iteration
    ranks = builder.core.pagerank_fused(
        rank_source=ranks,
        degree_source=degrees,
        damping=damping,
        tolerance=tolerance,
        max_iter=max_iter
    )
    
    # Final normalization
    ranks = builder.core.normalize_sum(ranks)
    builder.attach_as("pagerank", ranks)
    return builder.build()
```

## Expected Performance

With this fused primitive:
- **Setup**: 3 steps (init, degrees, fused iteration) instead of 1600+
- **FFI calls**: 3 instead of 1600+
- **CSR builds**: 1 instead of 100
- **Target**: ~50-100ms (2-3x native) instead of 7400ms

## Implementation Steps

1. ✅ Created `PageRankIterStep` (single iteration primitive)
2. ✅ Registered in step registry
3. ✅ Built and tested compilation
4. ⏭️ Create `PageRankFusedStep` (complete algorithm)
5. ⏭️ Register and test
6. ⏭️ Update benchmark
7. ⏭️ Validate results match native

## Alternative: Keep Iteration But Optimize CSR

If we want to keep the iteration construct for flexibility:
- Cache CSR after first `neighbor_agg`
- Reuse CSR on subsequent calls within same pipeline
- Add convergence detection to builder
- This would get us to maybe 500-800ms (10-25x slowdown vs native)

## Recommendation

Implement the **fused primitive** approach:
- Matches native performance characteristics
- Clean builder API
- Reusable for other iterative algorithms (eigenvector centrality, HITS, etc.)
- Users who need custom iteration logic can still use the individual primitives
