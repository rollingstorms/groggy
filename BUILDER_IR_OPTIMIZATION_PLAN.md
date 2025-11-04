# Builder IR Optimization & Validation Plan

**Goal**: Transform the builder DSL from a simple step recorder into a high-performance IR compilation system that eliminates FFI overhead, fuses operations, and achieves near-native performance.

**Success Metrics**:
- [ ] Reduce FFI crossings by 90%+ for typical algorithms
- [ ] Achieve <10% overhead vs hand-optimized Rust for PageRank, LPA, Connected Components
- [ ] Support automatic loop fusion and operation batching
- [ ] Maintain readable Python DSL syntax
- [ ] Enable future JIT compilation and autograd

---

## Phase 1: IR Foundation & Analysis (Days 1-3)

### Day 1: IR Type System & Representation ✅ COMPLETE

**Objective**: Define a typed, domain-aware IR that replaces the current JSON step list.

#### Tasks:
- [x] **Create `builder/ir.py` module**
  - Define `IRNode` base class with `domain`, `op_type`, `inputs`, `outputs`, `metadata`
  - Define domain-specific node types:
    - `CoreIRNode` (arithmetic, reductions, conditionals)
    - `GraphIRNode` (topology operations, neighbor aggregation)
    - `AttrIRNode` (attribute load/store)
    - `ControlIRNode` (loops, convergence checks)
  - Implement node serialization to/from JSON for FFI

- [x] **Add IR visualization**
  - Implement `ir.to_dot()` for Graphviz visualization
  - Create `ir.pretty_print()` for debugging
  - Add `ir.stats()` to show op counts by domain

- [x] **Update AlgorithmBuilder to emit typed IR**
  - Added `self.ir_graph: IRGraph` alongside `self.steps` list
  - Created `_add_ir_node()` helper method
  - Added `get_ir_stats()` and `visualize_ir()` methods
  - Maintained full backward compatibility with current JSON serialization

**Validation**:
```python
b = AlgorithmBuilder("test")
x = b.init_nodes(1.0)
y = x * 2.0 + 1.0
print(b.ir_graph.pretty_print())
# Should show typed nodes with domains
```

**Status**: ✅ All tests passing. Created:
- `builder/ir/nodes.py` - Typed IR node classes
- `builder/ir/graph.py` - IR graph structure with dependency tracking
- `test_ir_foundation.py` - Comprehensive test suite

**Next**: Day 2 - Dataflow Analysis

---

### Day 2: Dataflow Analysis ✅ COMPLETE

**Objective**: Build analysis passes to understand dependencies, lifetimes, and optimization opportunities.

#### Tasks:
- [x] **Implement dataflow graph construction**
  - Build dependency DAG from IR nodes
  - Identify read-after-write, write-after-read dependencies
  - Detect independent computation branches

- [x] **Add liveness analysis**
  - Track which variables are live at each point
  - Identify variables that can be dropped early
  - Find opportunities for in-place updates

- [x] **Implement loop analysis**
  - Detect loop-invariant computations (can be hoisted)
  - Find loop-carried dependencies
  - Identify reduction patterns

- [x] **Create analysis visualization**
  - Show dependency chains
  - Highlight optimization opportunities
  - Display critical path analysis

**Validation**:
```python
# PageRank example
# Should identify:
# - uniform, inv_deg are loop-invariant (hoist)
# - ranks has loop-carried dependency (special handling)
# - contrib is temporary (can be fused)
```

**Status**: ✅ All tests passing. Created:
- `builder/ir/analysis.py` - Complete dataflow analysis implementation
- `test_ir_dataflow.py` - Comprehensive test suite (8 tests, all passing)

**Key Features Implemented**:
- RAW/WAR/WAW dependency classification
- Backward liveness analysis with fixed-point iteration
- Dead code detection
- Fusion chain detection for arithmetic operations
- Critical path computation
- Complete analysis reporting with `print_analysis()`

**Next**: Day 3 - Performance Profiling Infrastructure

---

### Day 3: Performance Profiling Infrastructure ✅ COMPLETE

**Objective**: Measure current performance and establish baselines for optimization.

#### Tasks:
- [x] **Create `benches/builder_ir_profile.py`**
  - Micro-benchmarks for each primitive operation
  - Measure FFI crossing overhead
  - Profile memory allocation patterns
  - Track compilation time vs execution time

- [x] **Implement IR-level profiling hooks**
  - Count operations by domain
  - Measure theoretical vs actual FFI calls
  - Track fusion opportunities missed

- [x] **Add comparative benchmarks**
  - Builder DSL vs hand-written Python FFI
  - Builder DSL vs pure Rust
  - Current unoptimized vs optimized targets

- [x] **Create performance regression tests**
  - Set baseline performance metrics
  - Auto-fail if optimization regresses
  - Track improvements over time

**Deliverable**: `BUILDER_PERFORMANCE_BASELINE.md` with current metrics

**Status**: ✅ All profiling infrastructure complete. Created:
- `benches/builder_ir_profile.py` - Comprehensive profiling suite
- `BUILDER_PERFORMANCE_BASELINE.md` - Baseline metrics and optimization roadmap
- `benches/builder_ir_baseline.json` - Raw profiling data

**Key Findings**:
- FFI overhead: ~0.25ms per call
- Compilation overhead: Negligible (0.04x of execution)
- Loop overhead: Minimal (~0.18ms for 10 iterations)
- Primary optimization target: FFI call reduction through fusion
- Theoretical 2-4x speedup possible with fusion and hoisting

**Next**: Day 4 - Core Optimization Passes

---

### Day 4: Core Optimization Passes ✅ COMPLETE

**Objective**: Implement fundamental compiler optimization passes to reduce IR size and eliminate redundancy.

#### Tasks:
- [x] **Implement Dead Code Elimination (DCE)**
  - Mark-and-sweep algorithm for unused computations
  - Backward reachability from side-effecting operations
  - Safe removal of unreachable nodes

- [x] **Implement Constant Folding**
  - Evaluate constant expressions at compile time
  - Fold arithmetic operations on known values
  - Reduce IR size and enable further optimizations

- [x] **Implement Common Subexpression Elimination (CSE)**
  - Detect duplicate computations
  - Reuse existing results instead of recomputing
  - Update all uses to reference the surviving computation

- [x] **Create optimization framework**
  - `IROptimizer` class with pluggable passes
  - `optimize_ir()` convenience function
  - Iterative fixed-point optimization

**Validation**:
```python
# Test all three passes together
ir = build_test_ir()
optimize_ir(ir, passes=['constant_fold', 'cse', 'dce'])
# Should reduce node count and preserve semantics
```

**Status**: ✅ All tests passing. Created:
- `builder/ir/optimizer.py` - Complete optimization framework
- `test_ir_optimizer.py` - Comprehensive test suite (5 tests, all passing)

**Key Features**:
- DCE removes unused operations based on side-effect analysis
- Constant folding evaluates arithmetic on constants at compile time
- CSE eliminates duplicate operations with same inputs
- Framework supports iterative optimization to fixed point
- All optimizations preserve program semantics

**Results**:
- Test cases demonstrate 1-4 node reductions per optimization
- Combined passes work together (constant folding enables CSE)
- Semantic preservation validated across all tests

**Next**: Day 5-7 - Advanced Fusion Passes

---

## Phase 2: Operation Fusion (Days 5-7)

### Day 5: Arithmetic Fusion

**Objective**: Fuse chains of arithmetic operations into single FFI calls.

#### Tasks:
- [ ] **Implement fusion pattern matcher**
  - Detect chains like `(a * b + c) / d`
  - Match binary op trees
  - Handle scalar and vector operands

- [ ] **Add fused arithmetic IR nodes**
  - `FusedArithmetic` node type
  - Expression tree representation
  - Support for common patterns (AXPY, fused multiply-add, etc.)

- [ ] **Update Rust backend for fused ops**
  - Add `execute_fused_arithmetic` in FFI
  - Implement expression evaluator in Rust
  - Use SIMD where applicable

- [ ] **Add fusion optimization pass**
  - `passes/fuse_arithmetic.py`
  - Walk IR graph and replace fusable chains
  - Preserve semantics (floating point order, etc.)

**Test Case**:
```python
# Before: 4 FFI calls
contrib = ranks * inv_deg
contrib = is_sink.where(0.0, contrib)
# After: 1 FFI call (fused where(is_sink, 0.0, ranks * inv_deg))
```

---

### Day 5: Neighbor Aggregation Fusion

**Objective**: Combine graph operations with pre/post arithmetic.

#### Tasks:
- [ ] **Detect map-reduce patterns**
  - Find `transform → neighbor_agg → transform` chains
  - Match patterns like `G @ (values * weights)`
  - Identify reduction with post-processing

- [ ] **Implement fused neighbor operations**
  - Add `FusedNeighborOp` IR node
  - Support pre-aggregation transform (map phase)
  - Support post-aggregation transform (reduce phase)
  - Handle weighted aggregation

- [ ] **Update Rust implementation**
  - Add `execute_fused_neighbor_agg` in FFI
  - Implement single-pass CSR traversal with inline transforms
  - Optimize for cache locality

- [ ] **Add test suite**
  - Verify correctness vs unfused
  - Benchmark performance gains
  - Test edge cases (empty neighborhoods, zero weights)

**Target Pattern**:
```python
# Before: 3 FFI calls
contrib = ranks * inv_deg
contrib = is_sink.where(0.0, contrib)
neighbor_sum = sG.neighbor_agg(contrib, "sum")

# After: 1 FFI call
neighbor_sum = sG.fused_neighbor_agg(
    ranks, 
    pre_transform=lambda r: where(is_sink, 0.0, r * inv_deg),
    agg="sum"
)
```

---

### Day 6: Loop Fusion & Hoisting

**Objective**: Optimize loops by eliminating redundant computation and merging iterations.

#### Tasks:
- [ ] **Implement loop-invariant code motion (LICM)**
  - Identify expressions that don't change in loop
  - Hoist them outside loop body
  - Update variable dependencies

- [ ] **Add loop fusion pass**
  - Merge consecutive loops with same iteration count
  - Fuse independent update operations
  - Preserve loop-carried dependencies

- [ ] **Implement loop unrolling**
  - Detect small fixed-iteration loops
  - Unroll to eliminate loop overhead
  - Balance code size vs performance

- [ ] **Add convergence optimization**
  - Special handling for `until_converged` loops
  - Early exit detection
  - Delta computation fusion

**Example Transformation**:
```python
# Before:
uniform = 1.0 / n  # Computed every iteration (bad!)
for i in range(100):
    ranks = 0.85 * (sG @ contrib) + 0.15 * uniform

# After (hoisted):
uniform = 1.0 / n  # Computed once outside loop
for i in range(100):
    ranks = 0.85 * (sG @ contrib) + 0.15 * uniform
```

---

### Day 7: Integration & Testing

**Objective**: Integrate all fusion passes and validate correctness.

#### Tasks:
- [ ] **Create optimization pipeline**
  - Define pass order: arithmetic → neighbor → loop
  - Add pass orchestration in `builder/optimizer.py`
  - Allow user to enable/disable passes

- [ ] **Add correctness validation**
  - Property-based testing (Hypothesis)
  - Compare optimized vs unoptimized results
  - Test numerical stability

- [ ] **Benchmark suite**
  - Run PageRank with 0-4 optimization levels
  - Measure FFI call reduction
  - Compare wall-clock performance

- [ ] **Document optimization passes**
  - Add docstrings explaining each pass
  - Provide before/after examples
  - Document when passes are safe to apply

**Deliverable**: `OPTIMIZATION_PASSES.md` documentation

---

## Phase 3: Batched Execution (Days 8-10)

### Day 8: Batch Compilation

**Objective**: Compile entire algorithm IR into single batched execution plan.

#### Tasks:
- [ ] **Implement batch plan generator**
  - Create `BatchExecutionPlan` class
  - Pack multiple operations into single FFI payload
  - Handle variable lifetime and memory layout

- [ ] **Add execution plan serialization**
  - Define compact binary format for plans
  - Include operation opcodes, operand indices, result slots
  - Optimize for cache-friendly layout

- [ ] **Update FFI interface**
  - Add `execute_batch_plan(plan: bytes) → results`
  - Implement batch interpreter in Rust
  - Support streaming large results

- [ ] **Add plan caching**
  - Cache compiled plans by algorithm signature
  - Invalidate on graph structure change
  - Warm up common patterns

---

### Day 9: Parallel Execution

**Objective**: Execute independent operations in parallel.

#### Tasks:
- [ ] **Detect parallelizable operations**
  - Find independent branches in dataflow graph
  - Identify data-parallel operations (map, element-wise)
  - Mark thread-safe operations

- [ ] **Implement parallel batch executor**
  - Use Rayon for work-stealing parallelism
  - Schedule independent ops to different threads
  - Handle synchronization at merge points

- [ ] **Add parallel execution controls**
  - User-configurable thread count
  - Grain size tuning for parallel loops
  - Fallback to sequential for small inputs

- [ ] **Benchmark parallel speedup**
  - Measure scalability (1-16 threads)
  - Identify parallelization overhead
  - Find optimal thresholds

---

### Day 10: Memory Optimization

**Objective**: Minimize allocations and enable in-place updates.

#### Tasks:
- [ ] **Implement memory reuse analysis**
  - Detect when output can overwrite input
  - Find opportunities for buffer reuse
  - Track allocation sizes

- [ ] **Add in-place operation support**
  - Mark operations that can be in-place
  - Update IR to track mutability
  - Implement in Rust backend

- [ ] **Implement memory pooling**
  - Reuse buffers across algorithm runs
  - Pre-allocate common sizes
  - Profile memory high-water mark

- [ ] **Add memory profiling**
  - Track peak memory usage
  - Measure allocation counts
  - Compare to theoretical minimum

---

## Phase 4: JIT Compilation Foundation (Days 11-13)

### Day 11: Rust Code Generation

**Objective**: Generate specialized Rust code from IR for maximum performance.

#### Tasks:
- [ ] **Implement IR → Rust codegen**
  - Create `codegen/rust_emitter.py`
  - Generate function from IR graph
  - Handle all operation types

- [ ] **Add compilation infrastructure**
  - Generate temporary Rust crate
  - Compile with optimization flags
  - Load as dynamic library

- [ ] **Implement type inference**
  - Infer Rust types from operations
  - Handle generic operations
  - Generate type-specialized code

- [ ] **Add compilation caching**
  - Cache by IR hash
  - Store compiled artifacts
  - Version management

**Example Output**:
```rust
// Generated from PageRank IR
pub fn pagerank_generated(graph: &CSRGraph, max_iter: usize) -> Vec<f64> {
    let n = graph.node_count();
    let inv_n = 1.0 / n as f64;
    let mut ranks = vec![1.0; n];
    
    for _ in 0..max_iter {
        // Fused: contrib = where(is_sink, 0.0, ranks / (deg + 1e-9))
        // Then: neighbor_sum = sum of contrib from neighbors
        // Then: ranks = 0.85 * neighbor_sum + 0.15 * inv_n + sink_mass
        // All in single pass!
    }
    ranks
}
```

---

### Day 12: Template Library

**Objective**: Pre-compile common patterns for instant execution.

#### Tasks:
- [ ] **Define algorithm templates**
  - PageRank, LPA, Connected Components, BFS, etc.
  - Parameterize by graph type, value type
  - Include multiple implementations (sync, async, parallel)

- [ ] **Implement template registry**
  - Match IR pattern to template
  - Substitute parameters
  - Dispatch to pre-compiled code

- [ ] **Add template generation tools**
  - Tool to generate template from IR
  - Validation suite for templates
  - Performance benchmarks

- [ ] **Update builder to use templates**
  - Detect when IR matches template
  - Fall back to batch execution for novel patterns
  - Log template hits/misses

---

### Day 13: Benchmarking & Validation

**Objective**: Validate JIT performance and correctness.

#### Tasks:
- [ ] **Comprehensive benchmark suite**
  - Compare JIT vs batch vs native
  - Test across graph sizes (1K - 10M nodes)
  - Measure compilation overhead

- [ ] **Correctness validation**
  - Verify JIT results match native
  - Test edge cases and corner cases
  - Property-based testing

- [ ] **Performance regression tests**
  - Set performance SLOs
  - Auto-fail on regression
  - Track improvements

- [ ] **Documentation**
  - User guide for JIT system
  - Performance tuning guide
  - Troubleshooting common issues

**Deliverable**: `BUILDER_JIT_GUIDE.md`

---

## Phase 5: Advanced Features (Days 14-16)

### Day 14: Dataflow Optimizations

**Objective**: Implement graph-level optimization passes.

#### Tasks:
- [ ] **Common subexpression elimination (CSE)**
  - Detect duplicate computations
  - Share results
  - Update all uses

- [ ] **Dead code elimination (DCE)**
  - Remove unused variables
  - Eliminate no-op operations
  - Simplify control flow

- [ ] **Constant folding**
  - Evaluate constant expressions at compile time
  - Propagate constants through operations
  - Simplify conditional branches

- [ ] **Algebraic simplification**
  - Apply identities (x*1 = x, x+0 = x)
  - Simplify expressions (x/x = 1)
  - Canonicalize operation order

---

### Day 15: Profiling & Debugging Tools

**Objective**: Help users understand and optimize their algorithms.

#### Tasks:
- [ ] **Add IR visualization tools**
  - Interactive visualization of optimization passes
  - Show before/after transformations
  - Highlight performance hotspots

- [ ] **Implement execution profiler**
  - Time each operation
  - Show critical path
  - Identify bottlenecks

- [ ] **Add debugging mode**
  - Step through IR execution
  - Inspect intermediate values
  - Validate invariants

- [ ] **Create optimization guide**
  - Common patterns and anti-patterns
  - How to write fusion-friendly code
  - Performance tips

**Deliverable**: `BUILDER_OPTIMIZATION_GUIDE.md`

---

### Day 16: Future-Proofing

**Objective**: Prepare for gradients, autograd, and differentiable programming.

#### Tasks:
- [ ] **Design gradient IR nodes**
  - Forward and backward pass representation
  - Gradient accumulation
  - Checkpointing for memory efficiency

- [ ] **Implement basic autograd**
  - Automatic differentiation for arithmetic ops
  - Chain rule through neighbor aggregation
  - Gradient of custom operations

- [ ] **Add differentiable matrix view**
  - Convert graph to differentiable matrix ops
  - Integrate with existing matrix view
  - Support optimization workflows

- [ ] **Documentation and examples**
  - Tutorial on differentiable graph algorithms
  - Example: differentiable PageRank for link prediction
  - Integration guide for ML frameworks

**Note**: This is foundational work; full implementation may extend beyond this phase.

---

## Validation & Testing Strategy

### Correctness Testing
- [ ] **Unit tests for each optimization pass**
  - Test pass correctness in isolation
  - Verify semantics preserved
  - Test edge cases

- [ ] **Integration tests**
  - Full algorithm pipelines
  - Compare optimized vs unoptimized
  - Test with various graph types/sizes

- [ ] **Property-based tests (Hypothesis)**
  - Generate random IR graphs
  - Verify optimization equivalence
  - Stress test edge cases

### Performance Testing
- [ ] **Micro-benchmarks**
  - Each primitive operation
  - Fusion patterns
  - Memory allocation

- [ ] **Algorithm benchmarks**
  - PageRank, LPA, CC, BFS, etc.
  - Compare to baseline (hand-optimized Rust)
  - Track over time

- [ ] **Scaling tests**
  - Graph sizes: 1K, 10K, 100K, 1M, 10M nodes
  - Thread counts: 1, 2, 4, 8, 16
  - Various graph densities

### Regression Testing
- [ ] **Performance regression suite**
  - Run on every commit
  - Alert on >5% regression
  - Track performance trends

- [ ] **Correctness regression suite**
  - Snapshot tests for deterministic algorithms
  - Tolerance-based tests for iterative algorithms
  - Compare to reference implementations

---

## Success Criteria

### Phase 1 (IR Foundation)
- ✅ Typed IR with domain awareness
- ✅ Dataflow analysis working
- ✅ Performance baselines established

### Phase 2 (Fusion)
- ✅ Arithmetic fusion reduces FFI calls by 50%+
- ✅ Neighbor aggregation fusion working
- ✅ Loop optimization reduces redundant computation

### Phase 3 (Batching)
- ✅ Single FFI call per algorithm execution
- ✅ Parallel execution shows 2-4x speedup on 4+ cores
- ✅ Memory usage within 2x of theoretical minimum

### Phase 4 (JIT)
- ✅ JIT compilation produces near-native performance (<10% overhead)
- ✅ Template library covers common algorithms
- ✅ Compilation overhead <1s for typical algorithms

### Phase 5 (Advanced)
- ✅ Optimization passes reduce computation by 20%+
- ✅ Profiling tools help users find bottlenecks
- ✅ Autograd foundation ready for ML integration

---

## Timeline Summary

| Phase | Days | Focus | Key Deliverable |
|-------|------|-------|-----------------|
| 1 | 1-3 | IR Foundation | Typed IR system with analysis |
| 2 | 4-7 | Operation Fusion | Fused execution passes |
| 3 | 8-10 | Batched Execution | Single FFI call per algorithm |
| 4 | 11-13 | JIT Compilation | Code generation and templates |
| 5 | 14-16 | Advanced Features | Dataflow optimization & autograd prep |

**Total: ~16 days of focused implementation**

---

## Risk Mitigation

### Technical Risks
- **Risk**: Optimization breaks correctness
  - **Mitigation**: Extensive testing, property-based tests, validation mode

- **Risk**: JIT compilation too slow
  - **Mitigation**: Template library for common patterns, incremental compilation

- **Risk**: Fusion heuristics too conservative or too aggressive
  - **Mitigation**: Tunable optimization levels, profiling feedback

### Scope Risks
- **Risk**: Features creep beyond 16 days
  - **Mitigation**: Core features in Phases 1-3 (10 days), Phases 4-5 are enhancement

- **Risk**: Compatibility breaks existing code
  - **Mitigation**: Maintain compatibility layer, gradual migration path

---

## Next Steps

1. **Review and approve this plan**
2. **Set up development branch: `feature/ir-optimization`**
3. **Begin Phase 1, Day 1: IR Type System**
4. **Daily progress updates to this document**
5. **Weekly review of metrics and adjustments**

---

## Progress Tracking

### Phase 1: IR Foundation (Days 1-3)
- [x] Day 1: IR Type System & Representation ✅
- [x] Day 2: Dataflow Analysis ✅
- [x] Day 3: Performance Profiling Infrastructure ✅

### Phase 2: Operation Fusion (Days 4-7)
- [ ] Day 4: Arithmetic Fusion
- [ ] Day 5: Neighbor Aggregation Fusion
- [ ] Day 6: Loop Fusion & Hoisting
- [ ] Day 7: Integration & Testing

### Phase 3: Batched Execution (Days 8-10)
- [ ] Day 8: Batch Compilation
- [ ] Day 9: Parallel Execution
- [ ] Day 10: Memory Optimization

### Phase 4: JIT Compilation (Days 11-13)
- [ ] Day 11: Rust Code Generation
- [ ] Day 12: Template Library
- [ ] Day 13: Benchmarking & Validation

### Phase 5: Advanced Features (Days 14-16)
- [ ] Day 14: Dataflow Optimizations
- [ ] Day 15: Profiling & Debugging Tools
- [ ] Day 16: Future-Proofing (Autograd)

---

## Notes

This plan builds incrementally:
- Each phase delivers value independently
- Core functionality (Phases 1-3) provides 90% of performance gains
- Later phases (4-5) add polish and future capabilities
- Testing integrated throughout, not deferred to end

The IR optimization work is the heart of making the builder DSL production-ready. It transforms groggy from "nice syntax" to "high-performance graph computation engine."

