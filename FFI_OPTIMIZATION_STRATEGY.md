# FFI Optimization Strategy: Builder Primitives Architecture

## Executive Summary

The builder-based PageRank is **51x slower** than native (35.56ms vs 0.68ms for 100 nodes). The bottleneck is **neighbor_agg.total at 8.2%** of total time, but the real issue is **FFI overhead at 31.4%** (pipeline.run / algorithm.builder.step_pipeline). Each primitive step crosses the Python/Rust boundary independently, causing massive overhead.

**Current approach**: Compose 100+ tiny primitives → 100+ FFI crossings → 51x slowdown  
**Goal**: Reduce FFI crossings to near-zero while keeping composability

---

## Current Architecture (Strategy 1: Naive Composition)

### How It Works

**Files involved:**
- `python-groggy/python/groggy/builder.py` - Python DSL that builds step specs
- `src/algorithms/builder.rs:130-192` - `StepPipelineAlgorithm::execute()` loop
- `src/algorithms/steps/*.rs` - Individual step implementations (20+ files)
- `python-groggy/src/ffi/api/algorithms.rs` - FFI entry point

**Execution flow:**
```
Python builder.build()
  → Serializes to JSON spec: {"steps": [{"id": "core.add", ...}, ...]}
  → FFI: create algorithm handle
Rust StepPipelineAlgorithm::try_from_spec()
  → Parse JSON, instantiate each Step
  → Store Vec<Box<dyn Step>>

Python subgraph.apply(algo)
  → FFI: execute algorithm
Rust StepPipelineAlgorithm::execute()
  → For EACH step in pipeline:
      1. Create StepScope (wraps subgraph + variables)
      2. Call step.apply(ctx, scope)
      3. step.apply() reads inputs from StepVariables HashMap
      4. step.apply() writes outputs to StepVariables HashMap
      5. Return to Python (for profiling/cancellation checks)
      6. REPEAT for next step
```

**Example PageRank iteration (1 of 100):**
1. FFI call → core.mul (ranks * inv_degrees)
2. FFI call → core.where (mask sinks)
3. FFI call → core.neighbor_agg (sum neighbor ranks) ← 90% of iteration time
4. FFI call → core.mul (apply damping)
5. FFI call → core.broadcast_scalar (teleport term)
6. FFI call → core.mul (scale teleport)
7. FFI call → core.reduce_scalar (sink mass)
8. FFI call → core.mul (scale sinks)
9. FFI call → core.add (combine terms)
10. FFI call → core.add (add sinks)

**10 FFI crossings per iteration × 100 iterations = 1000+ FFI calls**

### Why It's Slow

1. **FFI overhead dominates** (31.4% of total time)
   - PyO3 argument marshaling (HashMap → Python dict)
   - GIL acquire/release overhead
   - Python object allocation/deallocation
   
2. **HashMap lookups everywhere**
   - StepVariables uses `HashMap<String, AlgorithmParamValue>`
   - Every step reads 1-3 variables, writes 1 variable
   - 1000 steps → 2000-4000 HashMap lookups
   
3. **No data locality**
   - Each step allocates new HashMap for results
   - No reuse of buffers between steps
   - Allocator thrashing

4. **No fusion opportunities**
   - `core.mul(core.add(a, b), c)` → 2 separate loops
   - Native code can fuse into 1 loop
   
5. **Expression system overhead**
   - `core.neighbor_agg` builds CSR from scratch every time
   - CSR cache exists but isn't reused between iterations

### Profiling Evidence

From `profile_pagerank_detailed.py` (100 nodes, 500 edges):
```
Native:    0.68 ms  (median 0.07 ms)
Builder:  35.56 ms
Overhead: 5161.7%

Breakdown:
- pipeline.run:                31.4%  (FFI overhead)
- neighbor_agg.total:           8.2%  (actual work)
- All other primitives:        60.4%  (scattered across 1700+ steps)
```

### When This Strategy Makes Sense

- **Prototyping**: Experimenting with algorithm structure
- **Small graphs**: < 100 nodes where 35ms is acceptable
- **One-shot queries**: Amortized cost doesn't matter
- **Debugging**: Explicit steps make it easy to inspect intermediate values

### When It Breaks Down

- **Iterative algorithms**: PageRank, LPA, community detection (100+ iterations)
- **Large graphs**: 10k+ nodes where native takes <100ms but builder takes seconds
- **Production workloads**: Need predictable performance

---

## Alternative Strategies

### Strategy 2: Fused Loops (Low Risk, High Impact)

**Concept**: Detect common patterns and emit fused step implementations.

**Example:**
```python
# Before (3 steps):
a = builder.core.add(x, y)
b = builder.core.mul(a, z)
c = builder.core.normalize_sum(b)

# After (1 fused step):
c = builder._fused_add_mul_norm(x, y, z)
```

**Implementation:**
- Add pattern recognition pass in `builder.py:_encode_step()`
- Register fused steps in Rust: `FusedAddMulNormStep`
- Fused step does 1 loop instead of 3

**Files to modify:**
- `python-groggy/python/groggy/builder.py` - Add fusion pass
- `src/algorithms/steps/composition.rs` - Add fused step types
- `src/algorithms/steps/registry.rs` - Register fused steps

**Pros:**
- Backward compatible (fallback to unfused)
- Incremental adoption (fuse hottest patterns first)
- Keeps composability for uncommon cases

**Cons:**
- Combinatorial explosion (need many fused variants)
- Still has FFI overhead (1 call vs 3, but not 0)
- Maintenance burden (more step types)

**Expected speedup:** 3-5x (reduce FFI calls by fusing 3-5 step chains)

---

### Strategy 3: JIT Compilation (Medium Risk, Very High Impact)

**Concept**: Compile the step pipeline to native code at runtime.

**Example:**
```python
algo = builder.build()  # Generates LLVM IR or Rust code
compiled = algo.compile()  # JIT compile to native function
result = compiled.execute(subgraph)  # Single FFI call
```

**Implementation approaches:**

**3a. LLVM via Cranelift:**
- Generate LLVM IR from step specs
- Use cranelift to JIT compile
- Single FFI call executes entire pipeline

**3b. Rust codegen + dynamic linking:**
- Generate Rust source from step specs
- Compile to .so/.dylib
- dlopen and call via FFI

**3c. WebAssembly:**
- Compile steps to WASM
- Execute in WASM runtime (wasmer/wasmtime)
- Zero-copy data sharing with Rust

**Files to add:**
- `src/algorithms/jit/` - JIT compilation infrastructure
- `src/algorithms/jit/codegen.rs` - IR generation
- `src/algorithms/jit/runtime.rs` - JIT execution engine

**Pros:**
- Near-native performance (1-2x slowdown vs handwritten)
- Full optimization (loop fusion, SIMD, vectorization)
- Zero FFI overhead (single call)

**Cons:**
- Complex implementation (weeks of work)
- Compilation overhead (amortized over many runs)
- Harder to debug (compiled code vs interpretable steps)
- Security concerns (code generation)

**Expected speedup:** 30-50x (approaches native performance)

---

### Strategy 4: Batched Execution (Low Risk, Medium Impact)

**Concept**: Execute multiple steps in Rust before returning to Python.

**Current:**
```python
for step in pipeline:
    execute_step(step)  # FFI call per step
```

**Batched:**
```python
execute_batch(pipeline, batch_size=10)  # FFI call per 10 steps
```

**Implementation:**
- Modify `StepPipelineAlgorithm::execute()` to batch steps
- Add profiling hook points between batches (not per-step)
- Python side requests batches instead of single steps

**Files to modify:**
- `src/algorithms/builder.rs:130-192` - Batching logic
- `python-groggy/src/ffi/api/algorithms.rs` - Batched execute API

**Pros:**
- Trivial to implement (few hours)
- Backward compatible
- Immediate 5-10x speedup

**Cons:**
- Still has some FFI overhead
- Loses per-step profiling granularity
- Doesn't solve HashMap lookup overhead

**Expected speedup:** 5-10x (reduce FFI calls by batch_size)

---

### Strategy 5: Pre-compiled Algorithm Templates (Low Risk, High Impact for Common Cases)

**Concept**: Pre-implement hot algorithms with optimal code, expose builder-like API.

**Example:**
```python
# Still looks like builder API
algo = gg.algorithms.templates.PageRank(
    damping=0.85,
    max_iter=100,
    teleport_mode="uniform"
)

# But internally it's native PageRank, not primitives
result = subgraph.apply(algo)
```

**Implementation:**
- Keep native implementations (src/algorithms/centrality/pagerank.rs)
- Add builder-style factory functions in Python
- Detect when builder matches template, substitute native version

**Files to modify:**
- `python-groggy/python/groggy/algorithms/templates.py` - Template API
- `python-groggy/python/groggy/builder.py` - Template matching
- Keep existing native implementations

**Pros:**
- Zero overhead for common algorithms
- Easy to add new templates incrementally
- Builder API still available for custom algorithms

**Cons:**
- Only helps for pre-known algorithms
- Doesn't solve general problem
- Code duplication (native + builder versions)

**Expected speedup:** 50x for templated algorithms, 1x for custom

---

### Strategy 6: Dataflow Graph Optimization (High Risk, Very High Impact)

**Concept**: Build IR, apply compiler-style optimizations, then execute.

**Pipeline:**
```
Builder DSL
  → Build dataflow IR (like MLIR or similar)
  → Optimization passes:
      - Dead code elimination
      - Common subexpression elimination
      - Loop fusion
      - Constant folding
      - Algebraic simplification
  → Lowering to optimized step sequence or native code
  → Execute
```

**Example optimizations:**
```python
# Before:
a = builder.core.mul(x, 0.85)
b = builder.core.mul(x, 0.85)  # CSE: reuse 'a'

# After:
a = builder.core.mul(x, 0.85)
b = a  # Alias, no computation

# Before:
with builder.iterate(100):  # Loop invariant code motion
    inv_n = builder.core.recip(n)  # Same every iteration
    ...

# After:
inv_n = builder.core.recip(n)  # Hoist outside loop
with builder.iterate(100):
    ...
```

**Files to add:**
- `src/algorithms/ir/` - IR data structures
- `src/algorithms/ir/optimize.rs` - Optimization passes
- `src/algorithms/ir/lower.rs` - Lowering to execution

**Pros:**
- Principled approach (follows compiler design)
- Catches optimization opportunities humans miss
- Extensible (add new passes incrementally)

**Cons:**
- Extremely complex (months of work)
- Hard to debug (optimizations hide intent)
- Needs sophisticated IR design

**Expected speedup:** 20-40x (depends on optimization quality)

---

## Recommended Approach: Hybrid Strategy

**Phase 1: Quick wins (1-2 days)**
- Implement Strategy 4 (batched execution): 5-10x speedup immediately
- Fix HashMap overhead: reuse buffers in StepVariables
- Profile again to find next bottleneck

**Phase 2: Medium term (1-2 weeks)**
- Implement Strategy 2 (fused loops) for hottest patterns:
  - Fused arithmetic chains (add/mul/div sequences)
  - Fused neighbor_agg + arithmetic
  - Fused iteration loops (PageRank, LPA)
- Expected: 10-20x total speedup

**Phase 3: Long term (1-3 months)**
- Choose between Strategy 3 (JIT) or Strategy 6 (IR optimizer)
- If JIT: Start with simple codegen, expand coverage
- If IR: Build minimal IR, add optimization passes incrementally
- Expected: 30-50x total speedup (near-native)

**Fallback: Strategy 5 (templates)**
- Always have native implementations for critical algorithms
- Use builder for prototyping, native for production
- This is the safest approach but doesn't solve the general problem

---

## Systemic Issues to Address

### 1. CSR Cache Not Working

**Problem:** `neighbor_agg` rebuilds CSR every iteration (see aggregations.rs:609-623)

**Current code:**
```rust
// Line 609: Build CSR from scratch
let _build_time = build_csr_from_edges_with_scratch(
    &mut csr,
    nodes.len(),
    edges.iter().copied(),
    ...
);
```

**Fix:** Cache CSR at subgraph level, keyed by (node_set, edge_set)

**Expected impact:** 5-10x speedup for neighbor_agg

### 2. HashMap Allocation Overhead

**Problem:** Every step allocates new HashMap for results (see StepVariables)

**Current:**
```rust
// Every step does this:
let mut result = HashMap::with_capacity(nodes.len());
for node in nodes {
    result.insert(node, computed_value);
}
scope.variables_mut().set_node_map("output", result);
```

**Fix:** Reuse buffers
```rust
// StepVariables maintains buffer pool
let mut buffer = scope.variables_mut().get_buffer_for("output");
buffer.clear();
for node in nodes {
    buffer.insert(node, computed_value);
}
// buffer stays in StepVariables, not reallocated
```

**Expected impact:** 2-3x speedup overall

### 3. No SIMD / Vectorization

**Problem:** Arithmetic steps process one node at a time

**Current:**
```rust
for (node, value) in source_map.iter() {
    let result = value * scalar;
    result_map.insert(*node, result);
}
```

**Fix:** Use columnar layout + SIMD
```rust
// Store values as Vec<f64> instead of HashMap<NodeId, f64>
let results: Vec<f64> = source_values
    .iter()
    .map(|&v| v * scalar)
    .collect();
// Auto-vectorizes to SIMD
```

**Expected impact:** 2-4x speedup for arithmetic

### 4. State Leak Between Runs

**Problem:** StepVariables not cleared between algorithm runs (see tracking notes)

**Current:** Global caches accumulate state

**Fix:** Reset caches at algorithm start:
```rust
impl StepPipelineAlgorithm {
    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        // Clear any cached state
        subgraph.clear_csr_cache();
        
        let mut variables = StepVariables::default();
        // ... rest of execution
    }
}
```

**Expected impact:** Correctness fix, minimal performance impact

---

## Decision Matrix

| Strategy | Implementation Time | Risk | Expected Speedup | Maintains Composability |
|----------|-------------------|------|------------------|------------------------|
| 1. Current (naive) | - | - | 1x (baseline) | ✅ Full |
| 2. Fused loops | 1-2 weeks | Low | 3-5x | ✅ Yes (with fallback) |
| 3. JIT compilation | 1-3 months | Medium | 30-50x | ✅ Yes |
| 4. Batched execution | 1-2 days | Low | 5-10x | ✅ Full |
| 5. Templates | Ongoing | Low | 50x (for templates) | ⚠️ Partial |
| 6. IR optimizer | 3-6 months | High | 20-40x | ✅ Yes |

## Questions for Agent Handoff

1. **What performance target?**
   - Quick fix: 5-10x speedup → Use Strategy 4 (batched execution)
   - Production ready: 30-50x → Use Strategy 3 (JIT) or 6 (IR)
   - Pragmatic: Case by case → Use Strategy 5 (templates)

2. **How much engineering time available?**
   - Days → Strategy 4
   - Weeks → Strategy 2 + 4
   - Months → Strategy 3 or 6

3. **Priority: Generality vs Performance?**
   - Generality (any algorithm) → Strategy 3 or 6
   - Performance (known algorithms) → Strategy 5

4. **Willing to maintain JIT infrastructure?**
   - Yes → Strategy 3
   - No → Strategy 2 + 4 + 5

## Next Immediate Steps

1. **Profile CSR cache effectiveness**
   - Add instrumentation to track cache hits/misses
   - Verify CSR is actually being rebuilt every time
   
2. **Implement batched execution (Strategy 4)**
   - Modify StepPipelineAlgorithm::execute to batch N steps
   - Reduces FFI overhead immediately
   
3. **Fix HashMap allocation**
   - Add buffer pool to StepVariables
   - Reuse allocations between steps
   
4. **Benchmark after each fix**
   - Target: builder within 5-10x of native (achievable with Strategies 2+4)
   - Re-profile to find next bottleneck

---

**Current Status:** Using Strategy 1 (naive composition). Performance gap: 51x.  
**Recommendation:** Implement Strategy 4 (batching) first, then Strategy 2 (fusion) for hot paths.  
**Long-term:** Evaluate Strategy 3 (JIT) vs keeping native implementations (Strategy 5).
