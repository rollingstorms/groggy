# FFI Optimization Strategy

## The Real Bottleneck

Profiling shows the builder is **467x slower** than native PageRank:
- Native: 0.11s for 200k nodes
- Builder: 51.4s for 200k nodes

**Root cause: FFI overhead, not primitive performance**

### Evidence
- 5k node test: 1500 FFI calls (100 iterations × 15 steps)
- Each step is FAST: 0.2-1.5ms per primitive
- But FFI crossing overhead: ~0.3-0.5ms per call
- Total FFI overhead: 1500 × 0.4ms = **600ms** (>50% of runtime)

### Why FFI is Expensive
1. Python→Rust boundary crossing
2. GIL acquisition/release
3. Data marshalling (PyO3 conversions)
4. Error handling setup/teardown
5. Stack frame overhead

## Solution: Batched Execution

### Current: Per-Step Execution
```
Python                          Rust
------                          ----
for i in range(100):
    result1 = step1.apply(g)    [FFI] → execute → [FFI]
    result2 = step2.apply(g)    [FFI] → execute → [FFI]
    result3 = step3.apply(g)    [FFI] → execute → [FFI]
    ...15 steps...
    # = 1500 FFI calls
```

### Optimized: Batch Execution
```
Python                          Rust
------                          ----
pipeline = [step1, ..., step15]
for i in range(100):
    result = execute_batch(     [FFI] → execute all 15 → [FFI]
        pipeline, state
    )
    # = 100 FFI calls (15x reduction!)
```

## Implementation Phases

### Phase 0: Measure Current State ✅
- [x] Profile builder PageRank on small graph
- [x] Confirm FFI overhead is dominant cost
- [x] Count steps per iteration (15 steps)
- [x] Identify hottest paths (neighbor_agg, mul, add, where)

### Phase 1: Batched Pipeline Execution (Target: 10-15x speedup)

**Goal:** Execute multiple steps in one Rust call

**Changes needed:**

1. **Rust: Add batched executor**
   ```rust
   // src/algorithms/pipeline.rs
   pub fn execute_batch(
       steps: Vec<Box<dyn AlgorithmStep>>,
       initial_state: StepVariables,
       graph: &GraphSpace
   ) -> Result<StepVariables> {
       let mut state = initial_state;
       for step in steps {
           state = step.apply(scope, state)?;
       }
       Ok(state)
   }
   ```

2. **FFI: Expose batched execution**
   ```rust
   // python-groggy/src/ffi/algorithms.rs
   #[pymethod]
   fn apply_batch(
       &self,
       py: Python,
       graph: &PyGraphSpace,
       steps: Vec<PyAlgorithmStep>,
       initial_vars: HashMap<String, PyVariable>
   ) -> PyResult<HashMap<String, PyVariable>> {
       py.allow_threads(|| {
           execute_batch(steps, initial_vars, graph)
       })
   }
   ```

3. **Python: Group steps by iteration**
   ```python
   # python-groggy/python/groggy/builder.py
   class BuiltAlgorithm:
       def optimize_for_ffi(self):
           """Group steps to minimize FFI calls"""
           # Detect loop boundaries
           # Batch steps within loops
           # Return optimized execution plan
   ```

**Expected impact:**
- 200k graph: 51s → ~5-8s (6-10x faster)
- FFI calls: 1500 → 100 (15x reduction)

### Phase 2: Constant Hoisting (Target: 5-10% additional)

**Problem:** Constants recreated every iteration
```python
# Current: 100× init_scalar for each constant
for iter in range(100):
    init_scalar(0.85)      # damping
    init_scalar(0.15)      # 1-damping
    init_scalar(1.0)       # multiplier
    # etc
```

**Fix:** Hoist constants outside loop
```python
# Optimized: constants created once
damping = builder.const_scalar(0.85)
teleport = builder.const_scalar(0.15)
# Reused in loop
```

**Implementation:**
1. Add `builder.const_scalar()` helper
2. Detect repeated `init_scalar` with same value
3. Automatically hoist to pipeline init phase

**Expected impact:**
- 200k graph: 5-8s → 4.5-7s (5-10% faster)
- Reduces step count by ~300 (3 constants × 100 iterations)

### Phase 3: Specialized Fusion Patterns (Target: 2-3x additional)

**Identify common patterns that can be fused:**

1. **Masked arithmetic:** `where(cond, 0, mul(a, b))` → single kernel
2. **Broadcast + arithmetic:** `mul(broadcast_scalar(x, map), y)` → inline
3. **Chain arithmetic:** `add(mul(a, b), c)` → fused multiply-add

**Implementation:**
1. Pattern matcher in optimizer
2. Fused step implementations
3. Transparent replacement

**Expected impact:**
- 200k graph: 4.5-7s → ~2-3s (2x faster)
- Approaches native performance (0.11s) considering DSL overhead

### Phase 4: JIT Compilation (Future / Optional)

Compile expression sequences to native code.

**Approaches:**
- LLVM backend
- Cranelift JIT
- Generate Rust code + runtime compile

**Trade-offs:**
- Major complexity increase
- Compilation time overhead
- Deployment complexity (need compiler)

**Decision:** Defer until Phases 1-3 implemented and measured.

## Success Criteria

### Must Have (Phase 1)
- [ ] Builder PageRank <5s on 200k nodes (10x improvement)
- [ ] All existing tests pass
- [ ] Numerical accuracy maintained (max diff <1e-6)
- [ ] No API changes visible to users

### Nice to Have (Phases 2-3)
- [ ] Builder PageRank <3s on 200k nodes (17x improvement)
- [ ] Automatic optimization (no manual annotation)
- [ ] Works for all builder algorithms (not just PageRank)

### Stretch Goal (Phase 4)
- [ ] Builder PageRank <1s on 200k nodes (50x improvement)
- [ ] Within 5-10x of hand-written native code

## Testing Strategy

1. **Correctness:**
   - Run full test suite with optimization enabled/disabled
   - Compare results between modes (must match exactly)
   - Test edge cases (empty graphs, single node, etc.)

2. **Performance:**
   - Benchmark suite with graphs: 1k, 5k, 10k, 50k, 200k nodes
   - Track FFI call count (should drop 10-15x)
   - Profile to confirm bottleneck has moved
   - Compare to native as baseline

3. **Regression:**
   - CI runs benchmarks on every commit
   - Alert if performance regresses >10%
   - Track optimization effectiveness over time

## References

- Profiling data: `benchmark_builder_vs_native.py` output
- Native impl: `src/algorithms/centrality/pagerank.rs`
- Builder DSL: `python-groggy/python/groggy/builder.py`
- Steps: `src/algorithms/steps/*.rs`
