# Pipeline Builder Completion - Executive Summary

**Objective**: Enable the builder DSL to construct PageRank and LPA algorithms  
**Timeline**: 2-3 weeks  
**Current Status**: Planning complete, ready to implement

---

## What's Missing

### Today's Builder (4 methods)
```python
builder = AlgorithmBuilder("simple")
nodes = builder.init_nodes(default=0.0)      # ✅ Works
degrees = builder.node_degrees(nodes)        # ✅ Works  
normalized = builder.normalize(degrees)      # ✅ Works
builder.attach_as("result", normalized)      # ✅ Works
```

### Target Builder (Full PageRank)
```python
builder = AlgorithmBuilder("pagerank")

ranks = builder.init_nodes(default=1.0)      # ✅ Works

with builder.iterate(20):                    # ❌ Missing
    neighbor_sums = builder.map_nodes(       # ❌ Missing
        "sum(ranks[neighbors(node)])",       # ❌ Missing
        inputs={"ranks": ranks}
    )
    damped = builder.core.mul(               # ❌ Missing
        neighbor_sums, 0.85
    )
    ranks = builder.var("ranks",             # ❌ Missing
        builder.core.add(damped, 0.15)       # ❌ Missing
    )
    ranks = builder.core.normalize_sum(ranks) # ❌ Missing

builder.attach_as("pagerank", ranks)         # ✅ Works
```

**Gap**: Need 5 new features to enable iterative algorithms

---

## Implementation Plan

### Week 1: Core Features ⚡

**Phase 1: Add Core Operations** (Days 1-3)
- `builder.core.add()`, `.sub()`, `.mul()`, `.div()`
- `builder.core.normalize_sum()`
- `builder.var()` for variable reassignment

**Phase 2: Add Neighbor Aggregation** (Day 4)
- `builder.map_nodes(fn, inputs)` for neighbor operations
- Support `sum()`, `mean()`, `mode()` functions
- Support `neighbors(node)` accessor

**Phase 3: Add Iteration** (Day 5)
- `builder.iterate(count)` context manager
- Loop unrolling in pipeline generation

**Milestone**: Can write PageRank skeleton ✓

---

### Week 2: Examples & Integration 🎯

**Phase 4: Build PageRank Example** (Days 1-2)
- Implement full PageRank with builder
- Verify results match native algorithm
- Test on various graph types

**Phase 5: Build LPA Example** (Days 3-4)  
- Implement Label Propagation with builder
- Verify community structure matches
- Test convergence behavior

**Phase 6: Integrate Validation** (Day 5)
- Connect Rust validation to Python builder
- Auto-validate on `build()`
- Rich error messages

**Milestone**: Both examples working ✓

---

### Week 3: Testing & Polish ✨

**Phase 7: Comprehensive Testing** (Days 1-2)
- Unit tests for all new methods
- Integration tests for examples
- 90%+ code coverage

**Phase 8: FFI Polish** (Days 3-5)
- Release GIL during pipeline execution
- Rich error translation (Rust → Python)
- Handle lifecycle management

**Milestone**: Production ready ✓

---

## Technical Approach

### Loop Implementation: Unrolling

**Decision**: Unroll loops in Python, generate repeated steps

**Why**: 
- Simpler implementation
- No Rust changes needed
- Works for fixed iteration counts
- Can optimize later if needed

**Example**:
```python
# Python DSL
with builder.iterate(3):
    x = builder.core.add(x, 1)

# Generates Rust steps
[
    {"id": "core.add", "left": "x", "right": 1, "target": "x_iter0"},
    {"id": "core.add", "left": "x_iter0", "right": 1, "target": "x_iter1"},
    {"id": "core.add", "left": "x_iter1", "right": 1, "target": "x_iter2"},
    {"id": "core.alias", "source": "x_iter2", "target": "x"},  # Final value
]
```

---

### Map Operations: Expression Delegation

**Decision**: Delegate expression parsing to existing Rust `map_nodes` step

**Why**:
- Expression parser already exists in Rust
- Supports complex operations
- Type-safe at execution time
- Python just passes string + context

**Example**:
```python
# Python
neighbor_sums = builder.map_nodes(
    "sum(ranks[neighbors(node)])",
    inputs={"ranks": ranks_var}
)

# Generates Rust step
{
    "id": "core.map_nodes",
    "params": {
        "expr": {"sum": {"index": {"var": "ranks"}, ...}},
        "target": "neighbor_sums_0"
    }
}
```

---

## File Changes Required

### Python Changes (Main Work)

**`python-groggy/python/groggy/builder.py`** (~300 new lines)
- Add `CoreOps` class (11 methods)
- Add `LoopContext` class
- Add `var()`, `map_nodes()`, `iterate()` methods
- Update `_encode_step()` for new operations
- Loop unrolling logic

**`python-groggy/python/groggy/builder.pyi`** (new file, ~150 lines)
- Type hints for all public APIs
- Better IDE support

**`tests/builder/test_*.py`** (new files, ~500 lines)
- `test_pagerank.py` - PageRank example
- `test_lpa.py` - LPA example  
- `test_builder_core.py` - Unit tests

### Rust Changes (Minimal)

**`python-groggy/src/ffi/api/pipeline.rs`** (~50 new lines)
- Expose `validate_pipeline` to Python
- Add GIL release around execution
- Better error translation

**No changes needed to**:
- Step primitives (all exist)
- Expression parser (works as-is)
- Pipeline executor (works as-is)

---

## Success Criteria

### Functional ✅
- [ ] PageRank example runs successfully
- [ ] Results match native PageRank (< 1e-6 error)
- [ ] LPA example runs successfully
- [ ] Communities similar to native LPA

### Performance ⚡
- [ ] Builder PageRank ≈ same speed as native
- [ ] GIL released during execution
- [ ] No memory leaks

### Quality 📊
- [ ] All tests passing (90%+ coverage)
- [ ] Type hints complete
- [ ] Documentation clear
- [ ] Error messages helpful

---

## Example: Before & After

### Before (Limited)
```python
# Can only do simple node operations
builder = AlgorithmBuilder("degree_norm")
degrees = builder.node_degrees(builder.init_nodes())
builder.attach_as("result", builder.normalize(degrees))
```

### After (Full Algorithm)
```python
# Can implement PageRank!
builder = AlgorithmBuilder("pagerank")
ranks = builder.init_nodes(1.0)

with builder.iterate(20):
    # Aggregate neighbor ranks
    sums = builder.map_nodes(
        "sum(ranks[neighbors(node)])", 
        inputs={"ranks": ranks}
    )
    
    # Apply damping factor
    damped = builder.core.mul(sums, 0.85)
    ranks = builder.var("ranks", 
        builder.core.add(damped, 0.15)
    )
    
    # Normalize for stability
    ranks = builder.core.normalize_sum(ranks)

builder.attach_as("pagerank", ranks)
```

**Impact**: From toy examples to production algorithms! 🚀

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Loop complexity | High | Low | Use unrolling, test incrementally |
| Expression bugs | Medium | Medium | Leverage existing parser, extensive tests |
| Type mismatches | Medium | Low | Validation catches before execution |
| Performance issues | Low | Low | Same Rust backend as native algorithms |

---

## Dependencies

### Already Complete ✅
- Rust step primitives (48+ steps)
- Schema registry system
- Validation framework
- Expression parser
- Pipeline executor

### Need to Build
- Python DSL extensions
- Loop handling (unrolling)
- FFI validation bridge

### External
- None

---

## Next Steps

1. **Today**: Review and approve plan
2. **Day 1**: Setup test infrastructure, add `CoreOps` skeleton
3. **Day 2-3**: Implement core operations
4. **Day 4**: Add `map_nodes`
5. **Day 5**: Add `iterate`
6. **Week 2**: Build examples
7. **Week 3**: Polish and test

---

## Questions?

**Q: Why unroll loops instead of a loop primitive?**  
A: Simpler to implement, works for fixed iterations, can optimize later

**Q: What about convergence-based loops?**  
A: Phase 2 feature, defer until basic loops work

**Q: Will this be slower than native algorithms?**  
A: No - generates the same Rust steps, should be equivalent performance

**Q: What if expression parsing fails?**  
A: Validation catches it before execution, provides clear error

**Q: Can we nest loops?**  
A: Not in v1, but the architecture supports it for v2

---

## Summary

**What**: Complete the pipeline builder to support PageRank and LPA  
**Why**: Enable users to build custom algorithms without Rust  
**How**: Add 5 core features (core ops, map_nodes, iterate, var, validation)  
**When**: 2-3 weeks  
**Risk**: Low (building on existing infrastructure)  
**Impact**: High (unlocks iterative algorithm construction)

**Status**: Ready to implement! 🎯
