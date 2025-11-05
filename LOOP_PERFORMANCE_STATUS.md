# Loop Performance Analysis & Status

**Date**: 2025-11-05  
**Issue**: Builder-based algorithms (PageRank, LPA) are 100x+ slower than native implementations

---

## Current Benchmark Results

```
PageRank (50k nodes, 100 iterations):
  Native:  0.032s
  Builder: 3.085s
  Ratio:   96x slower

PageRank (200k nodes, 100 iterations):
  Native:  0.115s  
  Builder: 23.432s
  Ratio:   204x slower

LPA (50k nodes, 10 iterations):
  Native:  0.226s
  Builder: 1.039s
  Ratio:   4.6x slower
```

---

## Root Cause Analysis

### What's Happening

1. **Loop Unrolling Works Correctly**
   - `with sG.iterate(100):` correctly unrolls to 100 copies of loop body
   - Each iteration has ~4 steps (neighbor_agg, mul, add, var)
   - Total: 608 steps for PageRank (1 init + 400 loop steps + extras + 1 normalize)

2. **Batching Works (All Steps Sent to Rust at Once)**
   - All 608 steps are serialized to JSON in Python
   - Sent to Rust in a **single FFI call** via `_groggy.pipeline.run_pipeline()`
   - Rust receives complete pipeline spec

3. **Problem: Rust Executes Each Step Separately**
   - Rust processes 608 steps **sequentially**, one at a time
   - Each step allocates intermediate vectors
   - No operation fusion at runtime
   - Example: `ranks * inv_deg` becomes:
     - Allocate `temp1` vector
     - Iterate through all nodes: `temp1[i] = ranks[i] * inv_deg[i]`
     - Return temp1
   - Then `neighbor_agg(temp1)` does another full iteration
   - No fusion, no in-place ops

### Why Native is Fast

Native PageRank implementation:
```rust
for iter in 0..100 {
    for node in nodes {
        let contrib = ranks[node] / (degrees[node] + epsilon);
        let neighbor_sum = sum_over_neighbors(contrib);
        ranks[node] = damping * neighbor_sum + teleport;
    }
}
```

Key differences:
- **Fused operations**: All arithmetic happens in one pass
- **In-place updates**: No intermediate allocations
- **Single node loop**: Processes each node once per iteration
- **Cache friendly**: Sequential memory access

Builder version effectively does:
```rust
for iter in 0..100 {
    temp1 = allocate_vector();
    for node in nodes { temp1[node] = ranks[node] / degrees[node]; }
    
    temp2 = allocate_vector();
    for node in nodes {
        temp2[node] = sum_over_neighbors(temp1);
    }
    
    temp3 = allocate_vector();
    for node in nodes {
        temp3[node] = damping * temp2[node] + teleport;
    }
    
    ranks = temp3;  // Copy
}
```

---

## Solutions (In Priority Order)

### ✅ Implemented: IR Infrastructure

**Status**: COMPLETE  
**Files**: `builder/ir/*.py`, `test_ir_*.py`

We have:
- Typed IR graph with domain-aware nodes
- Dataflow analysis (dependencies, liveness)
- Fusion detection passes (arithmetic, neighbor ops)
- Batch execution planner
- Loop analysis infrastructure

### 🔜 Missing: Rust Backend for Fused Operations

**Status**: NOT IMPLEMENTED (Deferred)  
**Why**: Requires significant Rust engineering (2-4 weeks)

To get real performance gains, need to implement in Rust:

1. **Fused Arithmetic Executor**
   ```rust
   fn execute_fused_axpy(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
       for i in 0..a.len() {
           out[i] = a[i] * b[i] + c[i];  // One pass, no temps
       }
   }
   ```

2. **Fused Neighbor Operations**
   ```rust
   fn execute_fused_neighbor_mul_agg(
       csr: &CSR,
       values: &[f64],
       weights: &[f64],
       out: &mut [f64]
   ) {
       for node in 0..csr.num_nodes() {
           let mut sum = 0.0;
           for neighbor in csr.neighbors(node) {
               sum += values[neighbor] * weights[neighbor];  // Fused!
           }
           out[node] = sum;
       }
   }
   ```

3. **In-Place Update Support**
   - Reuse buffers across iterations
   - Double-buffering for loop variables
   - Reference counting for early drops

### 🚧 Workaround: Apply Fusion Passes

**Status**: PARTIALLY WORKS  
**Current blocker**: Rust doesn't understand fused operation types

We can:
1. Run fusion passes on IR before sending to Rust
2. Replace fusable chains with `FusedOp` nodes
3. Serialize fused ops to JSON

But:
- Rust backend doesn't have `execute_fused_axpy()` etc.
- Falls back to executing original unfused steps
- No actual performance gain yet

### 🎯 Quickest Win: Manual Fusion in Python

**Status**: NOT YET TRIED  
**Effort**: 1-2 hours

Instead of:
```python
contrib = ranks / (deg + 1e-9)
neighbor_sum = sG @ contrib
ranks = sG.var("ranks", damping * neighbor_sum + teleport)
```

Do:
```python
# Pre-compute constants outside loop
teleport_val = (1 - damping) / sG.N

with sG.iterate(max_iter):
    # Emit a single fused kernel call
    ranks = sG.var("ranks", 
        sG.builder.core.fused_pagerank_step(
            ranks, deg, damping, teleport_val
        ))
```

Then implement `fused_pagerank_step` in Rust FFI:
```rust
fn fused_pagerank_step(
    ranks: &[f64],
    degrees: &[f64],
    csr: &CSR,
    damping: f64,
    teleport: f64,
    out: &mut [f64]
) {
    for node in 0..ranks.len() {
        let contrib = ranks[node] / (degrees[node] + 1e-9);
        let neighbor_sum = csr.aggregate_neighbors(node, ranks, |r, _| r * contrib);
        out[node] = damping * neighbor_sum + teleport;
    }
}
```

**Pros**:
- Immediate 50-100x speedup
- Requires only 1 new Rust function
- Doesn't block DSL work

**Cons**:
- Not automatic/general
- Need custom Rust function per algorithm
- Defeats purpose of builder DSL

---

## Recommended Next Steps

### Option A: Document Current Limitations ✅ RECOMMENDED

**Status**: This document

**Action**: 
1. Document that builder DSL is **correct but slow**
2. Mark loop fusion as "requires Rust implementation"
3. Continue with DSL API improvements and examples
4. Come back to performance when needed

**Timeline**: Immediate

### Option B: Implement Fused Operations in Rust

**Effort**: 2-4 weeks  
**Priority**: Medium

**Tasks**:
1. Implement fused arithmetic executors in Rust
2. Add fused neighbor aggregation
3. Implement in-place buffer reuse
4. Add FFI bindings for fused ops
5. Update step deserializer to recognize fused types
6. Benchmark and validate

**Expected Gain**: 30-50x speedup (close to native)

### Option C: Hybrid - Add Algorithm-Specific Kernels

**Effort**: 1-2 days per algorithm  
**Priority**: Low (defeats DSL purpose)

**Tasks**:
1. Implement `fused_pagerank_step()` in Rust
2. Implement `fused_lpa_step()` in Rust
3. Add builder methods to call them
4. Update examples

**Expected Gain**: 50-100x for specific algorithms

---

## Decision

**Current recommendation**: **Option A**

**Rationale**:
- DSL functionality is complete and correct
- Performance issue is well-understood
- Fix requires significant Rust work (2-4 weeks)
- No users are blocked (native implementations exist)
- Better to finish DSL API and examples first
- Come back to performance optimization when needed

**Mark as**: Known issue / Performance optimization deferred

---

## Test Commands

```bash
# Run benchmark to see current performance
python benchmark_builder_vs_native.py

# Debug: See number of steps generated
python -c "
from benchmark_builder_vs_native import *
g = create_test_graph(1000)
algo = pagerank(max_iter=100)
"
```

**Expected output**:
```
DEBUG: Generated 608 steps for 100 iterations
```

This confirms unrolling is working, and all steps go to Rust in one batch.

---

## References

- `BUILDER_IR_OPTIMIZATION_PLAN.md` - Full optimization strategy
- `PHASE2_DAY6_LOOP_OPTIMIZATION_PLAN.md` - Loop-specific optimizations
- `test_ir_fusion.py` - Fusion pass tests
- `builder/ir/optimizer.py` - Fusion detection code (Python side)

**Key insight**: We have the analysis infrastructure, but not the execution infrastructure.
