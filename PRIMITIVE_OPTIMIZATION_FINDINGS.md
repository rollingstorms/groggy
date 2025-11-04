# PageRank Primitives Profiling - Key Findings

## Summary

Removed the native PageRank implementation (pagerank.rs) to focus on optimizing the primitive-based builder approach. Detailed profiling reveals clear bottlenecks that need systemic optimization.

## Execution Times

| Graph Size | Nodes   | Edges    | Time (ms) |
|------------|---------|----------|-----------|
| Small      | 50      | 97       | 17.03     |
| Medium     | 1K      | 2,994    | 17.63     |
| Large      | 10K     | 29,994   | 198.98    |
| Very Large | 50K     | 149,994  | 1,008.91  |

## Top Bottlenecks (50K node graph)

### 1. **Neighbor Aggregation** - 703ms (69.7%)
- **Problem**: Each iteration pays CSR construction/lookup cost
- 20 iterations × ~35ms per neighbor_agg = 700ms total
- Dominates all other operations combined

**Optimization targets**:
- Cache CSR across iterations (currently rebuilding)
- Use bulk neighbor operations
- Parallel aggregation for large graphs

### 2. **Normalization** - 82ms (8.2%)
- **Problem**: Two-pass operation (sum + divide) × 20 iterations
- Each normalize: ~4ms

**Optimization targets**:
- Fuse normalize with previous add operation
- Single-pass normalization
- Consider in-place normalization

### 3. **Graph Operations** - 155ms (15.4%)
- node_degrees: 145ms
- broadcast_scalar: 51ms total (20 iterations × ~2.5ms)

**Optimization targets**:
- Compute degrees once, not in every builder pipeline
- Fuse broadcast_scalar with subsequent operations
- Cache degree maps

### 4. **Arithmetic Operations** - 351ms (34.8%)
- mul operations: 231ms (20 iterations × ~11ms)
- where operations: 113ms (20 iterations × ~5.6ms)
- add operations: small overhead

**Optimization targets**:
- Fuse mul + neighbor_agg (weighted aggregation)
- Fuse where + mul (conditional multiply)
- Batch arithmetic operations

## Systemic Issues

### FFI Overhead
- Each primitive crosses Python/Rust boundary
- 228 steps total for 20 iterations ≈ 11 steps per iteration
- Small operations (add, mul, where) pay disproportionate FFI cost
- **Solution**: Composite steps or move iteration loop to Rust

### Loop Unrolling Explosion
- 20 iterations × 11 steps/iter = 220 steps
- Each step has FFI marshalling overhead
- Temporary allocations per step
- **Solution**: Create a `pagerank_iteration` step that fuses the hot path

### Lack of Operation Fusion
- Separate steps for operations that could be combined:
  - `mul(ranks, inv_deg)` + `neighbor_agg` → `weighted_neighbor_agg`
  - `mul(neighbor_sum, damping)` + `add(teleport)` → `damped_add`
  - `add(damped, teleport)` + `normalize` → `add_normalize`

## Recommended Optimizations (Priority Order)

### P0: CSR Caching in Neighbor Aggregation
**Impact**: ~600-700ms savings (69% speedup)
- Cache CSR on first build, reuse across iterations
- Key on (subgraph_id, direction)
- Should reduce from 703ms to ~100ms

### P1: Fused Weighted Neighbor Aggregation
**Impact**: ~250ms savings (combine mul + neighbor_agg)
- New primitive: `core.weighted_neighbor_agg(values, weights, agg='sum')`
- Eliminates temporary `contrib` map
- Fuses two hot operations

### P2: Batch Degree Computation
**Impact**: ~145ms savings on first run, free on subsequent
- Compute and cache degrees at graph level
- Builder accesses cached degrees
- Should be nearly free after first computation

### P3: Fused Conditional Operations  
**Impact**: ~113ms savings (fuse where + mul)
- New primitive: `core.mul_masked(left, right, mask, zero_value=0.0)`
- Eliminates temporary masked arrays

### P4: Composite PageRank Iteration Step
**Impact**: ~500ms+ savings from FFI reduction
- Create `centrality.pagerank_iteration` step
- Takes: ranks, degrees, damping, teleport
- Returns: new_ranks
- Runs entire iteration in Rust (no FFI per operation)

## Expected Results After All Optimizations

| Optimization | Time Savings | Remaining |
|--------------|--------------|-----------|
| Baseline     | -            | 1,009ms   |
| P0: CSR Cache| 600ms        | 409ms     |
| P1: Weighted Agg | 100ms    | 309ms     |
| P2: Degree Cache | 145ms    | 164ms     |
| P3: Fused Ops| 50ms         | 114ms     |
| P4: Iteration Step | 80ms   | **34ms**  |

**Target**: <50ms for 50K nodes (30× speedup)

## Next Steps

1. Instrument CSR cache in `neighbor_agg` - verify cache miss/hit patterns
2. Implement CSR caching with proper invalidation
3. Add `weighted_neighbor_agg` primitive
4. Benchmark after each optimization
5. Consider full `pagerank_iteration` composite step if primitives still too slow

## Files Changed

- **Removed**: `src/algorithms/centrality/pagerank.rs`
- **Modified**: `src/algorithms/centrality/mod.rs` (removed PageRank registration)
- **Modified**: `benches/centrality_algorithms.rs` (removed PageRank benchmark)
- **Created**: `profile_pr_primitives.py` (profiling script)
- **Generated**: `pr_profiling_output.txt` (full profiling data)

## References

- Profiling script: `profile_pr_primitives.py`
- Benchmark comparison: `benchmark_builder_vs_native.py`
- Full profiling output: `pr_profiling_output.txt`
