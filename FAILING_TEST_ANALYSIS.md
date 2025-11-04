# Failing Test Analysis: test_builder_node_degrees_directed_chain

## Status: Pre-existing Issue (Not caused by refactor)

### Test Details

**Test**: `tests/test_builder_core.py::test_builder_node_degrees_directed_chain`

**Graph Structure**:
```
n0 -> n1 -> n2
```

**Expected Degrees** (by test):
- n0: degree = 1 (one outgoing edge)
- n1: degree = 1 (one outgoing edge)
- n2: degree = 0 (no outgoing edges)

**Actual Degrees**:
- n0: degree = 1
- n1: degree = 2  ❌ (test expects 1)
- n2: degree = 1  ❌ (test expects 0)

### Root Cause

The test expects **directed out-degree** behavior, but the graph implementation returns **total degree** (treating edges as undirected or counting both in+out edges).

For node n1:
- In directed graph (out-degree only): 1 outgoing edge to n2 → degree = 1
- In undirected graph: 1 edge to n0 + 1 edge to n2 → degree = 2
- **Actual**: degree = 2 ✓ (undirected behavior)

### Evidence This Predates Refactor

1. **Tested with original builder.py**: Same failure
   ```
   Original builder results:
     deg0 = 1 (expected: 1) ✓
     deg1 = 2 (expected: 1) ✗
     deg2 = 1 (expected: 0) ✗
   ```

2. **Git history**: Recent commits mention "directed/undirected graph implementation"
   - Suggests this is an area under active development
   - Test may be aspirational (testing desired future behavior)

3. **Refactor impact**: Zero
   - Phase 1 (operator overloading): No changes to degree calculation
   - Phase 2 (trait migration): Moved methods but kept same implementation
   - 37/38 tests passing (same ratio before and after)

### Recommendation

**Short term**: Mark test as `@pytest.mark.xfail` with reason:
```python
@pytest.mark.xfail(reason="Awaiting directed graph degree implementation")
def test_builder_node_degrees_directed_chain():
    ...
```

**Long term**: Implement proper directed/undirected graph support:
1. Add `graph.add_directed_edge()` and `graph.add_undirected_edge()` APIs
2. Implement `node.out_degree()`, `node.in_degree()`, `node.degree()` methods
3. Update step primitives to handle directed vs undirected correctly
4. Re-enable test once directed graph support is complete

### Impact on Refactor

**None** - This is a pre-existing graph implementation issue unrelated to the builder DSL refactor.

The refactor is proceeding successfully:
- ✅ Phase 1 complete (infrastructure + operators)
- ✅ Phase 2 complete (trait migration)
- ✅ 37/38 tests passing (97.4%)
- ✅ Zero regressions introduced

---

**Conclusion**: Safe to continue with Phase 3. This test failure is a known graph implementation issue, not a refactor bug.
