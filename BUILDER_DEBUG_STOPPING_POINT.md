# Builder Debugging Stopping Point

**Date:** 2025-11-03  
**Status:** Paused for systematic debugging

## Current State

We've been working on debugging the builder-based PageRank and LPA implementations in `benchmark_builder_vs_native.py`. Both algorithms are showing incorrect results despite having all necessary primitives implemented.

### Issues Identified

1. **PageRank Accuracy**
   - Current tolerance: Max diff ~0.000003 (should be within 0.0000005)
   - Discrepancy suggests a subtle algorithmic issue, not just rounding
   - All primitives appear to work individually

2. **LPA Collapsing to Single Community**
   - Native implementation maintains multiple communities (e.g., 5 communities)
   - Builder implementation collapses to 1 community
   - Warning messages about variable redefinition:
     ```
     UserWarning: Pipeline validation: Step 3 (core.update_in_place): redefines variable 'nodes_0'
     UserWarning: Pipeline validation: Step 6 (core.update_in_place): redefines variable 'nodes_0_iter0'
     ```
   - Suggests the update_in_place mechanism is not working as intended

### Primitives Implemented (Phases 1-3 Complete)

**Phase 1: Arithmetic & Comparison**
- ✅ `core.recip` - Element-wise reciprocal with epsilon
- ✅ `core.compare` - Comparison operations (eq, gt, lt, etc.)
- ✅ `core.where` - Conditional element-wise selection
- ✅ `core.reduce_scalar` - Reduce node map to scalar
- ✅ `core.broadcast_scalar` - Broadcast scalar to node map

**Phase 2: Neighbor Operations & LPA Components**
- ✅ `core.neighbor_agg` - Weighted neighbor aggregation
- ✅ `core.collect_neighbor_values` - Collect neighbor values
- ✅ `core.mode` - Find most common value with tie-breaking
- ✅ `core.update_in_place` - Update map in place with ordering
- ✅ `core.histogram` - Create histogram from values

**Phase 3: Utilities & Graph Constants**
- ✅ `core.clip` - Clip values to range
- ✅ Auto-detect scalar in existing ops (`core.add`, `core.mul`)
- ✅ Graph constants (`core.graph.node_count`, `core.graph.edge_count`)

### Current Benchmark Setup

File: `benchmark_builder_vs_native.py`

**PageRank Builder Implementation:**
```python
def build_pagerank_algorithm(n, damping=0.85, max_iter=20):
    # Initialize, compute degrees, safe reciprocal
    # Identify sinks with core.compare
    # Loop: compute contributions, handle sinks, apply damping
    # Normalize with core.normalize_sum
```

**LPA Builder Implementation:**
```python
def build_lpa_algorithm(max_iter=10):
    # Initialize with unique labels
    # Loop: collect neighbor labels, find mode, update in place
```

## What We Know

1. **All individual primitives work** - Unit tests pass for each step
2. **Composition is failing** - Something about how primitives interact is wrong
3. **Possible culprits:**
   - Variable aliasing/resolution in loops
   - update_in_place not preserving iteration-specific state
   - Scalar handling in arithmetic operations
   - Normalization not being applied correctly
   - Sink handling in PageRank

## What We Don't Know

1. Which specific primitive or combination is causing the PageRank error?
2. Why does update_in_place trigger redefinition warnings?
3. Is the issue in the Python builder DSL or the Rust execution?
4. Are the iteration mechanics working correctly?

## Removed Work

- Deleted `src/algorithms/steps/async_lpa.rs` - was an attempt to fix LPA async behavior
- Removed references from `src/algorithms/steps/mod.rs`
- This approach was abandoned because the problem is more fundamental: **making any algorithm work correctly from primitives**

## Next Steps (When Resuming)

1. **Systematic Primitive Testing**
   - Create minimal test cases for each primitive in isolation
   - Test primitive combinations (e.g., recip → mul → neighbor_agg)
   - Verify scalar handling in arithmetic operations

2. **Debug PageRank Step-by-Step**
   - Add instrumentation to print intermediate values
   - Compare each step's output with native implementation
   - Identify exactly where divergence occurs

3. **Debug LPA Variable Management**
   - Investigate why update_in_place redefines variables
   - Check if loop unrolling creates proper variable chains
   - Verify that ordered=True actually enforces deterministic ordering

4. **Consider Simpler Test Case**
   - Create a tiny 5-node graph with known results
   - Manually trace what each primitive should produce
   - Compare with actual outputs

5. **Review Variable Resolution**
   - Examine `_resolve_operand` and `_encode_step` in builder.py
   - Verify that loop variables are properly scoped
   - Check if aliases are being collapsed incorrectly

## Files to Focus On

**Python Side:**
- `python-groggy/python/groggy/builder.py` - Variable resolution, loop handling
- `benchmark_builder_vs_native.py` - Algorithm implementations

**Rust Side:**
- `src/algorithms/steps/transformations.rs` - update_in_place implementation
- `src/algorithms/steps/arithmetic.rs` - Scalar handling in operations
- `src/algorithms/steps/normalization.rs` - Normalization methods
- `src/algorithms/builder.rs` - Pipeline execution

## Test Commands

```bash
# Run the benchmark
python benchmark_builder_vs_native.py

# Run builder tests
pytest tests/test_builder_*.py -v

# Run Rust tests for steps
cargo test --test builder_validation_integration
```

## Key Insight

The async_lpa approach was wrong because it tried to fix symptoms (LPA not working) rather than root causes (primitives not composing correctly). The real problem is that **one or more primitives is subtly broken**, or **the composition mechanism has bugs**. We need to debug methodically, not add more complexity.

## Context for Future Debugging

When you return to this:
1. Start with a minimal reproducible case
2. Add print statements to trace execution
3. Compare intermediate values with native implementation
4. Don't assume any primitive works correctly just because its unit test passes
5. The composition and variable management systems are suspect

The goal is not to make PageRank or LPA work specifically, but to **make the primitive composition system reliable** so any algorithm can be built from them.
