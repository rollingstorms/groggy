# Builder IR Performance Baseline

**Date**: 2025-11-04  
**Commit**: Builder DSL Refactor Phase 1 Complete  
**Test Graph**: 1000 nodes, 5000 edges

---

## Executive Summary

The Builder DSL is currently functional with domain-separated traits (CoreOps, GraphOps, AttrOps, IterOps) and operator overloading. Performance profiling reveals several optimization opportunities.

**Key Findings**:
- FFI overhead: ~0.25ms per call (baseline for optimization)
- Compilation overhead: Very low (0.04x of execution time)
- Loop overhead: Minimal (~0.18ms for 10 iterations)
- Builder operations complete in 1-9ms range for test graph

---

## Detailed Metrics

### 1. FFI Overhead

```
Baseline (single call):    0.22 ms
Two FFI calls:             0.42 ms (1.9x)
Per-call overhead:         0.25 ms (measured over 100 calls)
```

**Interpretation**: Each FFI crossing costs ~0.25ms. For algorithms with 100+ operations, this adds up to 25ms+ overhead. This is the primary target for batching and fusion optimizations.

---

### 2. Builder Operation Performance

Individual operation timings on 1000-node graph:

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| init_nodes | 8.55 | Includes allocation and initialization |
| scalar_mul | 0.55 | Scalar-vector multiplication |
| degrees | 1.35 | Graph topology query |

**Interpretation**: Operations are reasonably fast, but compound quickly. A 10-iteration PageRank with 10 ops/iteration = 100+ FFI calls = 25ms+ overhead minimum.

---

### 3. Compilation Overhead

```
IR construction time:     0.04 ms
Algorithm execution:      0.99 ms
Ratio (build/exec):       0.04x
```

**Interpretation**: IR construction is negligible compared to execution. We can afford more complex analysis and optimization passes without impacting overall performance.

**Opportunity**: This low overhead makes JIT compilation and optimization passes highly attractive.

---

### 4. Loop Overhead

```
10 iterations (with loop construct):   1.86 ms
10 iterations (unrolled):              1.68 ms
Loop overhead:                         0.18 ms (9.7%)
```

**Interpretation**: The loop construct adds minimal overhead. Not a primary optimization target, but could be further reduced with loop fusion.

---

### 5. Builder vs Native Comparison

```
Builder PageRank (10 iter): 5.29 ms
Native PageRank:            Not available for comparison
```

**Status**: Native implementation not benchmarked. Need to add reference implementation or measure against NetworkX/other libraries.

**Target**: Achieve <2x overhead vs optimized native Rust implementation after fusion and batching optimizations.

---

## Fusion Opportunities

### Identified Patterns

Based on PageRank algorithm structure, common fusable patterns:

1. **Arithmetic chains**: `(a * b + c)` → single fused operation
   - Current: 2 FFI calls
   - After fusion: 1 FFI call
   - Savings: 50%

2. **Conditional arithmetic**: `where(mask, 0.0, a * b)` → fused masked multiply
   - Current: 2 FFI calls
   - After fusion: 1 FFI call
   - Savings: 50%

3. **Map-reduce patterns**: `neighbor_agg(a * b)` → fused map-aggregate
   - Current: 2 FFI calls
   - After fusion: 1 FFI call
   - Savings: 50%

4. **Loop-invariant hoisting**: Computations outside loop executed once
   - Example: `inv_n = 1.0 / n` computed 100x → computed 1x
   - Savings: 99%

### Theoretical Speedup Calculation

For a typical PageRank implementation:
- **Operations per iteration**: ~12
- **Total operations (100 iter)**: ~1200
- **Fusable operations**: ~50% (600)
- **After fusion**: ~600 operations remain
- **Theoretical speedup**: 2.0x

Additional gains from loop invariant hoisting could push this to **3-4x** speedup.

---

## Optimization Roadmap

### Phase 2: Operation Fusion (Priority: HIGH)

**Target**: Reduce operation count by 50%

1. **Arithmetic Fusion** (Days 4-5)
   - Fuse binary operation chains
   - Implement expression tree optimizer
   - Handle scalar and vector operands
   - **Expected gain**: 30-40% operation reduction

2. **Neighbor Aggregation Fusion** (Day 6)
   - Combine pre/post transforms with aggregation
   - Single-pass CSR traversal with inline operations
   - **Expected gain**: 20-30% for graph-heavy algorithms

3. **Loop Fusion & Hoisting** (Day 7)
   - Hoist loop-invariant computations
   - Fuse multiple loops where possible
   - **Expected gain**: 10-20% for iterative algorithms

**Combined expected speedup**: 2-3x

---

### Phase 3: Batched Execution (Priority: MEDIUM)

**Target**: Single FFI call per algorithm

1. **Batch Compilation** (Day 8)
   - Serialize entire algorithm IR in one FFI call
   - Rust-side interpreter for operation sequences
   - **Expected gain**: Eliminate per-op FFI overhead (~25ms for 100 ops)

2. **Parallel Execution** (Day 9)
   - Identify independent operation branches
   - Parallel execution on multi-core systems
   - **Expected gain**: 1.5-3x on multi-core (parallelizable ops only)

3. **Memory Optimization** (Day 10)
   - In-place updates where safe
   - Reuse temporary buffers
   - **Expected gain**: Reduce memory pressure, improve cache locality

**Combined expected speedup**: 3-5x (on top of fusion gains)

---

### Phase 4: JIT Compilation (Priority: FUTURE)

**Target**: Near-native performance (<10% overhead)

1. **Rust Code Generation** (Days 11-12)
   - Emit optimized Rust code from IR
   - Compile and link at runtime (or AOT)
   - **Expected gain**: Eliminate interpreter overhead

2. **Template Library** (Day 12)
   - Pre-compiled kernels for common patterns
   - Pattern matching and template instantiation
   - **Expected gain**: 5-10x for template-matched algorithms

3. **Benchmarking & Validation** (Day 13)
   - Ensure correctness
   - Measure against native implementations
   - **Target**: <10% overhead vs hand-written Rust

**Combined expected speedup**: 5-10x (cumulative with all optimizations)

---

## Success Criteria

### Short-term (Phase 2 - Operation Fusion)
- [ ] Reduce FFI calls by 50% for PageRank
- [ ] Achieve 2x speedup for iterative algorithms
- [ ] All existing tests pass with fused operations

### Medium-term (Phase 3 - Batched Execution)
- [ ] Single FFI call per algorithm execution
- [ ] 3-5x speedup vs baseline
- [ ] Parallel execution for independent operations

### Long-term (Phase 4 - JIT Compilation)
- [ ] <10% overhead vs native Rust
- [ ] Support for custom user algorithms
- [ ] Template library covering 80% of common patterns

---

## Testing Strategy

### Correctness Tests
- All optimizations must pass existing test suite
- Compare optimized vs unoptimized results (numerical equivalence)
- Edge case handling (empty graphs, single nodes, disconnected components)

### Performance Regression Tests
- Automated benchmarking on each optimization
- Track metrics over time
- Alert on performance regressions

### Integration Tests
- Real-world algorithm implementations
- Large-scale graphs (1M+ nodes)
- Production workload simulation

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fusion breaks semantics | Medium | High | Extensive testing, careful dependency analysis |
| Optimization complexity | High | Medium | Incremental approach, modular passes |
| Performance not meeting targets | Medium | Medium | Profiling at each stage, adjust targets |
| Maintenance burden | Medium | Medium | Clean abstraction layers, good documentation |

---

## Next Actions

1. ✅ Complete Day 3 (Performance Profiling) - DONE
2. → Begin Day 4 (Arithmetic Fusion)
   - Implement fusion pattern matcher
   - Add fused arithmetic IR nodes
   - Update Rust backend
3. Continue Phase 2 through Days 5-7
4. Reassess and adjust targets based on measured gains

---

## Appendix: Raw Data

See `benches/builder_ir_baseline.json` for complete profiling data.

**Profile Command**:
```bash
python benches/builder_ir_profile.py
```

**Environment**:
- Platform: Darwin
- Python: 3.x
- Rust: Latest stable
- Graph: 1000 nodes, 5000 edges (random)

---

*Generated: 2025-11-04*  
*Next Update: After Phase 2 completion (Day 7)*
