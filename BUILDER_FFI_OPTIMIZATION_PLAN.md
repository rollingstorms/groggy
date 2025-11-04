# Builder FFI Optimization Plan

## Problem Analysis

From profiling `benchmark_builder_vs_native.py` on 5k nodes:
- **Builder PageRank: ~577ms** (5k nodes) → **51s** (200k nodes) = **467x slower than native**
- **1500+ FFI calls** for 100 iterations (15 steps per iteration)
- Each step is fast (0.2-1.5ms), but FFI overhead dominates
- Native keeps everything in Rust, no boundary crossings

### Per-Iteration Breakdown (15 steps)
```
1. core.mul          (ranks × inv_degrees)
2. core.where        (mask sinks)
3. core.neighbor_agg (sum contributions)  ← 1.5ms, heavier
4. core.mul          (apply damping)
5. core.broadcast_scalar (teleport term)
6. core.mul          (scale teleport)
7. core.where        (extract sink ranks)
8. core.reduce_scalar (sum sinks)
9. core.mul          (scale sink mass)
10. core.mul         (apply damping to sinks)
11. core.add         (combine damped + teleport)
12. core.add         (add sink redistribution)
13+ Multiple init_scalar calls (constants)
```

## Root Cause

**FFI overhead kills performance:**
- Each `sg.apply(step)` crosses Python→Rust boundary
- Data marshalling, GIL management, error handling
- Even at 100ns per call, 1500 calls = 150µs base overhead
- Real overhead is ~0.3-0.5ms per crossing
- Total: 1500 × 0.4ms = **600ms just in FFI overhead**

## Solution: Step Fusion Optimization

### Phase 1: Loop-Level Fusion (Target: 10-20x speedup)

**Goal:** Execute entire iteration body in single Rust call

Instead of:
```python
with builder.iterate(100):
    # 15 step calls per iteration = 1500 FFI calls
```

Optimize to:
```rust
// Single Rust function executes full iteration
fn pagerank_iteration(
    ranks: &NodeMap,
    inv_degrees: &NodeMap,
    is_sink: &NodeMap,
    damping: f64,
    n: f64,
    csr: &CSR,
) -> NodeMap {
    // All 15 steps executed in Rust
    // Returns updated ranks
}
```

**Implementation:**
1. Add `LoopFusionOptimizer` in `src/algorithms/optimizer.rs`
2. Detect common patterns (PageRank iteration, LPA iteration)
3. Replace step sequence with single fused step
4. Keep primitives unchanged, optimization is transparent

### Phase 2: Expression JIT (Target: Additional 2-3x)

**Goal:** Compile expressions to native code

For steps like:
```python
contrib = builder.core.mul(ranks, inv_degrees)
contrib = builder.core.where(is_sink, 0.0, contrib)
```

Generate:
```rust
// JIT-compiled per-node kernel
for (i, &rank) in ranks.iter().enumerate() {
    contrib[i] = if is_sink[i] { 0.0 } else { rank * inv_degrees[i] };
}
```

### Phase 3: Batched Constant Injection (Quick win: 5-10%)

**Problem:** Each iteration creates multiple `init_scalar` steps for constants (damping, 1-damping, etc.)

**Fix:** Inject all constants at pipeline construction time
```python
# Instead of 100× init_scalar per constant
builder.core.mul(x, 0.85)  # creates init_scalar every iteration

# Hoist to:
damping_scalar = builder.const_scalar(0.85)  # created once
builder.core.mul(x, damping_scalar)  # reuses variable
```

## Implementation Priority

### P0: Loop Fusion (Highest Impact)
- [ ] Create `LoopFusionOptimizer` trait
- [ ] Implement PageRank-specific fusion
- [ ] Pattern matcher for common loop bodies
- [ ] Integration tests comparing fused vs unfused
- [ ] Benchmark: expect 200k graph to drop from 51s → ~5s

### P1: Constant Hoisting (Quick Win)
- [ ] Add `builder.const_scalar()` method
- [ ] Detect repeated scalar initialization in loops
- [ ] Rewrite to hoist constants outside loop
- [ ] Benchmark: expect 5-10% improvement

### P2: Generic Step Fusion (Medium Term)
- [ ] Identify fusion candidates automatically
- [ ] Arithmetic chains: `mul → add → mul` → single step
- [ ] Predicate chains: `compare → where` → single masked op
- [ ] Neighbor patterns: `neighbor_agg → mul` → weighted aggregation

### P3: Expression JIT (Future)
- [ ] LLVM backend for expression compilation
- [ ] Or simpler: generate Rust code and `cargo build`
- [ ] Cache compiled kernels per expression pattern

## Success Metrics

Current (200k nodes):
- Native PageRank: **0.11s**
- Builder PageRank: **51.4s** (467x slower)

Target after Phase 1:
- Builder PageRank: **~3-5s** (27-45x slower, acceptable for DSL overhead)

Target after Phases 1+2+3:
- Builder PageRank: **~0.5-1s** (5-10x slower, excellent for generated code)

## Testing Strategy

1. **Correctness First:**
   - All optimizations must pass existing tests
   - `test_builder_pagerank.py` must match native within 1e-6
   - Test with/without optimization flags

2. **Performance Regression Suite:**
   - Benchmark suite with multiple graph sizes
   - Track FFI call count (should drop dramatically)
   - Profile step timing distribution

3. **Pattern Coverage:**
   - PageRank (done)
   - LPA (should benefit from same fusion)
   - Betweenness centrality (if using builder)
   - Custom user algorithms

## Non-Goals

- **Don't** break the primitive-based API
- **Don't** add complexity to user-facing builder code
- **Don't** optimize individual primitives (already fast)
- **Don't** sacrifice correctness for speed

## References

- Profiling output: 1500 steps, ~0.3-1.5ms each
- Native PageRank: `src/algorithms/centrality/pagerank.rs`
- Current builder: `python-groggy/python/groggy/builder.py`
- Step primitives: `src/algorithms/steps/*.rs`
