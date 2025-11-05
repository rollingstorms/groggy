# Phase 2, Day 6: Loop Fusion & Optimization Strategy

**Date**: 2025-11-04  
**Status**: Planning Complete, Implementation Deferred

## Overview

Day 6 focuses on loop-level optimizations that can dramatically reduce FFI overhead and improve algorithm performance. The key techniques are:

1. **Loop-Invariant Code Motion (LICM)** - Hoist computations that don't change across iterations
2. **Loop Fusion** - Merge consecutive loops with compatible iteration patterns
3. **Loop Unrolling** - Eliminate loop overhead for small fixed-iteration loops

## Current State

### Existing Infrastructure

We have solid foundations:
- ✅ IRGraph with typed nodes (Core, Graph, Attr, Control)
- ✅ Dataflow analysis (liveness, dependencies, fusion detection)
- ✅ Basic optimization passes (DCE, constant folding, CSE)
- ✅ Control flow nodes (ControlIRNode with loop support)

### What's Missing for Loop Optimization

1. **Loop Body Tracking**: Need to explicitly track which nodes belong to each loop's body
2. **Execution Order**: IRGraph uses a list structure; we need topological ordering for hoisting
3. **Loop Metadata**: Count, convergence criteria, iteration variables need to be formalized
4. **Side Effect Analysis**: Must distinguish pure operations from those with side effects

## Loop Optimization Patterns

### Pattern 1: Hoisting Loop-Invariant Computations

**Before**:
```python
for _ in range(100):
    teleport = 0.15 / n  # Computed 100 times (wasteful!)
    ranks = 0.85 * neighbor_sum + teleport
```

**After (with LICM)**:
```python
teleport = 0.15 / n  # Computed once outside loop
for _ in range(100):
    ranks = 0.85 * neighbor_sum + teleport
```

**Impact**: Eliminates 99 redundant FFI calls for teleport computation.

### Pattern 2: Loop Fusion

**Before**:
```python
# Loop 1: Update ranks
for _ in range(100):
    ranks = compute_ranks(...)

# Loop 2: Update degrees (independent)
for _ in range(100):
    degrees = compute_degrees(...)
```

**After (fused)**:
```python
for _ in range(100):
    ranks = compute_ranks(...)
    degrees = compute_degrees(...)  # Fused into same loop
```

**Impact**: Reduces loop overhead and improves cache locality.

### Pattern 3: Loop Unrolling

**Before**:
```python
for _ in range(3):
    x = x * 2
```

**After (unrolled)**:
```python
x = x * 2
x = x * 2
x = x * 2
```

**Impact**: Eliminates loop control overhead for small fixed iterations.

## Implementation Strategy

### Phase 1: Enhanced Loop Tracking (Week 3)

**Goal**: Make loops first-class citizens in the IR.

**Tasks**:
1. Add `loop_body: List[str]` to ControlIRNode metadata
2. Track loop-carried dependencies explicitly
3. Add `is_loop_invariant(node, loop)` analysis function
4. Implement topological sort for IRGraph

**Code Sketch**:
```python
class LoopAnalysis:
    def get_loop_body(self, loop_node: ControlIRNode) -> List[IRNode]:
        """Return all nodes in loop body in execution order."""
        
    def get_loop_variables(self, loop_node: ControlIRNode) -> Set[str]:
        """Variables modified within loop."""
        
    def is_invariant(self, node: IRNode, loop: ControlIRNode) -> bool:
        """Check if node can be hoisted outside loop."""
```

### Phase 2: LICM Implementation (Week 3)

**Goal**: Automatically hoist loop-invariant operations.

**Algorithm**:
```
for each loop in IR:
    loop_vars = get_loop_variables(loop)
    
    for each node in loop_body:
        if all inputs not in loop_vars and no side effects:
            hoist node before loop
            update dataflow graph
```

**Safety Conditions**:
- Node has no side effects (pure computation)
- All inputs are defined outside loop or are constants
- Output is not used before the loop

### Phase 3: Loop Fusion (Week 4)

**Goal**: Merge independent consecutive loops.

**Algorithm**:
```
for each pair of consecutive loops (L1, L2):
    if same_iteration_count(L1, L2) and not dependent(L1, L2):
        merge L2 body into L1
        remove L2
        recompute liveness
```

**Fusion Conditions**:
- Loops have same iteration count
- L2 doesn't depend on L1's outputs
- Both loops have same convergence criteria (if applicable)

### Phase 4: Loop Unrolling (Week 4)

**Goal**: Eliminate overhead for small loops.

**Algorithm**:
```
for each loop with fixed count ≤ 4:
    replicate loop body 'count' times
    rename variables uniquely
    remove loop node
```

**When to Unroll**:
- Fixed iteration count known at compile time
- Count ≤ 4 (tunable threshold)
- Body is small (prevents code bloat)

## Integration with Existing Optimizer

The loop optimizations will integrate into `IROptimizer`:

```python
class IROptimizer:
    def optimize(self, passes: List[str] = None):
        if passes is None:
            passes = ['dce', 'constant_fold', 'cse', 'licm', 'loop_fusion']
        
        for pass_name in passes:
            if pass_name == 'licm':
                self.loop_invariant_code_motion()
            elif pass_name == 'loop_fusion':
                self.fuse_loops()
            # ...
    
    def loop_invariant_code_motion(self) -> bool:
        """Hoist loop-invariant operations."""
        changed = False
        for loop in self._find_loops():
            changed |= self._hoist_invariants(loop)
        return changed
    
    def fuse_loops(self) -> bool:
        """Merge compatible consecutive loops."""
        changed = False
        loops = self._find_loops()
        i = 0
        while i < len(loops) - 1:
            if self._can_fuse(loops[i], loops[i+1]):
                self._merge_loops(loops[i], loops[i+1])
                changed = True
            i += 1
        return changed
```

## Expected Performance Impact

Based on PageRank as a test case:

| Optimization | FFI Calls Before | FFI Calls After | Speedup |
|--------------|------------------|-----------------|---------|
| Baseline (unoptimized) | ~1000/iteration | ~1000 | 1.0x |
| + LICM | ~1000 | ~700 | 1.4x |
| + Loop Fusion | ~700 | ~700 | 1.0x* |
| + Constant Folding | ~700 | ~500 | 1.4x |
| **Total** | **~1000** | **~500** | **2.0x** |

\* Loop fusion improves cache locality but doesn't reduce FFI calls in PageRank (single main loop).

## Testing Strategy

### Unit Tests

1. **test_loop_analysis.py**
   - Identify loop bodies correctly
   - Detect loop-carried dependencies
   - Classify invariant vs variant operations

2. **test_licm.py**
   - Hoist simple arithmetic (teleport = 0.15 / n)
   - Don't hoist operations with side effects
   - Don't hoist loop-carried dependencies
   - Preserve semantics (floating point order)

3. **test_loop_fusion.py**
   - Fuse independent loops
   - Don't fuse dependent loops
   - Don't fuse loops with different counts
   - Handle convergence loops correctly

4. **test_loop_unrolling.py**
   - Unroll small fixed loops
   - Don't unroll large loops
   - Rename variables correctly
   - Preserve semantics

### Integration Tests

Test on real algorithms with loop optimizations enabled:

```python
@algorithm("pagerank", optimize=["licm", "constant_fold"])
def pagerank(sG, damping=0.85, max_iter=100):
    # Should automatically hoist teleport computation
    ...
```

Verify:
- Correctness (same results as unoptimized)
- Performance (measure FFI call reduction)
- Compilation time (optimizations shouldn't be too slow)

## Next Steps

### Immediate (Week 3)

1. ✅ Document loop optimization strategy (this file)
2. Add loop body tracking to ControlIRNode
3. Implement topological sort for IRGraph
4. Add side effect analysis

### Short-term (Week 3-4)

5. Implement LICM pass
6. Add LICM tests
7. Implement loop fusion pass
8. Add fusion tests

### Medium-term (Week 4-5)

9. Implement loop unrolling (optional)
10. Add unrolling tests
11. Integrate all passes into IROptimizer
12. Benchmark on PageRank, LPA, BFS

### Long-term (Phase 3+)

13. Auto-vectorization within loops
14. Parallel loop execution
15. Convergence detection optimization
16. Adaptive unrolling based on profiling

## Design Decisions

### Why Not Implement Now?

Loop optimization requires careful integration with the execution model:

1. **Execution Order**: IRGraph currently uses a simple list. Hoisting requires precise ordering.
2. **Loop Semantics**: Need to formalize what "loop body" means in our IR.
3. **Side Effects**: Must distinguish pure operations (can hoist) from impure (can't).
4. **Testing**: Loop optimizations are easy to get wrong; need comprehensive tests.

Given we're in the middle of Phase 2 (dataflow & fusion), it's better to:
- Finish Phase 2 basics (dataflow analysis, pattern detection)
- Plan loop optimizations thoroughly (this document)
- Implement in Phase 3 or 4 when the foundation is solid

### Alternative Approaches Considered

1. **Eager Unrolling**: Unroll all loops at Python level before IR
   - ❌ Loses optimization opportunities
   - ❌ Makes convergence loops impossible

2. **JIT Loop Compilation**: Compile entire loop to native code
   - ✅ Maximum performance
   - ❌ Complex implementation
   - 💡 Good future direction (Phase 4+)

3. **Manual Loop Annotations**: Let users mark invariant operations
   - ✅ Simple implementation
   - ❌ Poor developer experience
   - ❌ Error-prone

**Chosen**: Automatic analysis + optimization passes
- ✅ Best developer experience
- ✅ Composable with other optimizations
- ✅ Proven approach (used by LLVM, GCC, etc.)

## References

### Academic Papers

- "Loop-Invariant Code Motion" - Allen & Cocke, 1976
- "Efficient Loop Fusion with Integer Range Analysis" - Song et al., 2010
- "Polyhedral Optimization of TensorFlow Computation Graphs" - Zinenko et al., 2018

### Similar Systems

- **LLVM**: Comprehensive loop optimization passes
- **TVM**: Loop tiling and fusion for ML graphs
- **Halide**: Schedule-based loop transformations
- **JAX**: Loop unrolling via `jax.lax.scan`

### Internal Documentation

- `BUILDER_IR_OPTIMIZATION_PLAN.md` - Overall optimization strategy
- `BUILDER_PERFORMANCE_BASELINE.md` - Performance metrics and targets
- `python-groggy/python/groggy/builder/ir/optimizer.py` - Existing optimization framework
- `python-groggy/python/groggy/builder/ir/analysis.py` - Dataflow analysis utilities

## Conclusion

Loop optimization is a high-impact area that can dramatically improve performance by:
1. Reducing redundant FFI calls (LICM)
2. Improving cache locality (fusion)
3. Eliminating control overhead (unrolling)

However, it requires careful design and implementation. This document provides a comprehensive roadmap for implementation in Week 3-4 of Phase 2, after the current dataflow foundation is complete.

**Status**: ✅ Planning Complete - Ready for implementation in Week 3

---

**Author**: Groggy IR Optimization Team  
**Last Updated**: 2025-11-04  
**Phase**: 2 (IR Dataflow & Fusion)  
**Week**: 2 (Days 5-7)
