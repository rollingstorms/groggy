# Next Steps - Current Priorities

## 🚨 CRITICAL ISSUES

### 🔥 Performance Issue - O(n²) Algorithmic Scaling
**URGENT**: NetworkX is 83x faster than Groggy at 250K node scale

**Problem**: Direct comparison shows severe algorithmic scaling issue:
- NetworkX: O(n) performance (~12ns/node at 250K scale)  
- Groggy: O(n²) performance (~1000ns/node at 250K scale)
- **Result**: NetworkX completes 250K node filtering in 0.003s, Groggy takes 0.25s

**Root Cause**: Node filtering appears to be using inefficient bulk attribute access patterns in Rust core. Edge filtering works efficiently, but node filtering has a bottleneck.

**Investigation Results**:
- Edge filtering: Fast, efficient individual access pattern
- Node filtering: Slow, appears to be calling bulk attribute access for entire columns
- Issue is in the core Rust filtering algorithms, not Python bindings

**Action Required**:
1. Profile the `filter_nodes` implementation in Rust core
2. Compare with `filter_edges` efficient implementation  
3. Fix the O(n²) algorithmic complexity
4. Target: Match or exceed NetworkX performance at 250K+ scale

**Files to investigate**:
- `src/core/query.rs` - filtering algorithms
- `src/api/graph.rs` - graph filter methods
- Focus on attribute access patterns during filtering

---

## 🎯 HIGH PRIORITY FEATURES

### 1. GraphTable Integration - DataFrame Views
**Status**: Architecture ready, implementation in progress

**Goal**: Enable pandas-like data analysis workflows with graph data
```python
# Create table views of graph data
node_table = g.table()  # All nodes with all attributes
edge_table = g.edges.table()  # All edges with source/target + attributes

# Subgraph table views
engineers = g.filter_nodes('dept == "Engineering"')  
eng_table = engineers.table()  # Only engineering nodes

# Export capabilities
node_table.to_pandas()  # Convert to pandas DataFrame
node_table.to_csv('data.csv')  # Direct export
```

**Current Status**: 
- ✅ Core PySubgraph architecture refactored with proper graph references
- ✅ GraphTable class exists and is functional
- 🔄 Need to re-enable `table()` methods on PyGraph and PySubgraph
- 🔄 Need to test integration with new subgraph architecture

**Next Actions**:
1. Re-implement `table()` method on PyGraph class
2. Re-implement `table()` method on PySubgraph class  
3. Test GraphTable creation from both graphs and subgraphs
4. Validate pandas export functionality

### 2. Enhanced Query Parser - 3+ Term Logic
**Goal**: Support complex logical expressions in string queries
```python
# Current: 2-term expressions work
g.filter_nodes("age > 30 AND dept == 'Engineering'")

# Missing: 3+ term expressions  
g.filter_nodes("age > 30 AND dept == 'Engineering' AND salary < 150000")
g.filter_nodes("(age < 25 OR age > 65) AND active == true")
```

**Implementation**: Extend the query parser to handle:
- Parentheses for grouping: `(age < 25 OR age > 65)`
- 3+ term expressions: `A AND B AND C`
- Mixed operators: `A AND (B OR C)`

### 3. Phase 2.2 - Subgraph Graph Reference Architecture  
**Goal**: Enable complex subgraph operations that require graph references

**Current Limitation**: Some subgraph operations fail because they can't access the parent graph:
```python
components = g.connected_components()
components[0].set(team_id=1)  # May fail without proper graph reference
```

**Solution**: This is partially solved with the `Rc<RefCell<Graph>>` refactor, but needs validation and testing.

---

## 🔄 IMMEDIATE ACTIONS

### Phase 1: Fix Critical Performance Issue
1. **Profile node filtering performance** - identify the O(n²) bottleneck
2. **Compare efficient edge filtering** - understand why edges are fast
3. **Implement algorithmic fix** - achieve O(n) scaling for node operations
4. **Validate with benchmarks** - ensure Groggy matches/exceeds NetworkX performance

### Phase 2: Complete GraphTable Integration  
1. **Re-enable table() methods** - implement with new PySubgraph architecture
2. **Test subgraph table views** - validate GraphTable works with filtered data
3. **Validate exports** - ensure to_pandas(), to_csv() work correctly
4. **Performance test** - ensure table creation is efficient

### Phase 3: Enhanced Query Parsing
1. **Extend parser grammar** - support 3+ terms and parentheses
2. **Update query tests** - validate complex expressions  
3. **Performance optimize** - ensure parsing doesn't add overhead

---

## 📋 VALIDATION CHECKLIST

### Performance Validation
- [ ] Node filtering achieves O(n) scaling 
- [ ] 250K node filtering completes in < 50ms (target: match NetworkX)
- [ ] Memory usage remains efficient at scale
- [ ] No regression in edge filtering performance

### Feature Validation  
- [ ] `g.table()` returns proper GraphTable instance
- [ ] `subgraph.table()` works with filtered data
- [ ] GraphTable exports work: `to_pandas()`, `to_csv()`, `to_json()`
- [ ] Complex queries parse correctly: `"A AND B AND C"`, `"(A OR B) AND C"`
- [ ] Subgraph operations work: `components[0].set()`, `subgraph['attr']`

### Integration Testing
- [ ] All existing tests pass after performance fixes
- [ ] GraphTable integrates correctly with existing workflows  
- [ ] Query parser handles edge cases gracefully
- [ ] Memory usage remains stable under load

---

## 🎯 SUCCESS CRITERIA

1. **Performance**: Groggy matches or exceeds NetworkX performance at 250K+ node scale
2. **Usability**: GraphTable enables seamless data science workflows  
3. **Completeness**: Query parser handles complex logical expressions
4. **Reliability**: All operations work consistently across graphs and subgraphs

The focus is on **fixing the critical performance issue first**, then completing the GraphTable integration to deliver a powerful, fast graph analysis tool.