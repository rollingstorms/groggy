# GroupBy Refactoring Plan

## Overview

This document outlines the plan to refactor the groupby functionality in Groggy by:
1. **Removing** groupby methods from the main Graph class
2. **Adding** groupby methods to Subgraph and accessor classes (NodesAccessor, EdgesAccessor)
3. **Implementing** proper return types (SubgraphArray for graph data, TableArray for table data)

## Current State Analysis

### Current GroupBy Implementation

**Graph-level groupby methods:**
- `Graph::group_nodes_by_attribute()` in `src/api/graph.rs` (line 2276)
- `PyGraph::group_by()` in `python-groggy/src/ffi/api/graph.rs` (line 963)
- `PyGraph::group_nodes_by_attribute()` in `python-groggy/src/ffi/api/graph.rs` (line 938)

**Table-level groupby methods:**
- `GraphTable::group_by()` in `src/storage/table/graph_table.rs` (line 992)
- `NodesTable::group_by()` in `src/storage/table/nodes.rs` (line 320)
- `EdgesTable::group_by()` in `src/storage/table/edges.rs` (line 586)
- `BaseTable::group_by()` in `src/storage/table/base.rs` (line 560)
- `PyGraphTable::group_by()` in `python-groggy/src/ffi/core/table.rs` (line 704)

### Current Accessor Structure

**NodesAccessor:**
- Located: `python-groggy/src/ffi/core/accessors.rs` and `python-groggy/src/ffi/storage/accessors.rs`
- Has: `table()` method returning GraphTable
- Current pattern: `g.nodes.table()` → GraphTable

**EdgesAccessor:**
- Located: `python-groggy/src/ffi/core/accessors.rs` and `python-groggy/src/ffi/storage/accessors.rs`
- Has: `table()` method returning GraphTable  
- Current pattern: `g.edges.table()` → GraphTable

**Subgraph:**
- Located: `src/subgraphs/subgraph.rs` and `python-groggy/src/ffi/subgraphs/subgraph.rs`
- Has: `table()` method returning NodesTable (node-focused)
- Has: `nodes_table()` and `edges_table()` methods

### Current Array Types

**SubgraphArray:**
- Located: `python-groggy/src/ffi/storage/subgraph_array.rs`
- Has: `table()` method returning TableArray
- Supports method chaining via iterator pattern

**TableArray:**
- Located: `python-groggy/src/ffi/storage/table_array.rs` (referenced but not shown in search)
- Used for collections of table objects

## Proposed New Architecture

### Target API Design

```python
# === CURRENT (to be removed) ===
result = g.group_by('attr_name', 'agg_attr', 'operation')  # Returns dict

# === PROPOSED (new functionality) ===

# 1. Subgraph groupby → SubgraphArray
subgraph_groups = g.view().group_by('attr_name', 'nodes')  # or 'edges'
subgraph_groups = subgraph.group_by('attr_name', 'nodes')

# 2. Accessor groupby → SubgraphArray  
node_groups = g.nodes.group_by('node_attr')
edge_groups = g.edges.group_by('edge_attr')

# 3. Table groupby → TableArray
table_groups = g.nodes.table().group_by('node_attr') 
table_groups = g.edges.table().group_by('edge_attr')
table_groups = subgraph.table().group_by('attr_name')
```

### Return Type Logic

| Source | Method | Return Type | Rationale |
|--------|--------|-------------|-----------|
| `Subgraph` | `group_by('attr', 'nodes')` | `SubgraphArray` | Groups create new subgraphs |
| `Subgraph` | `group_by('attr', 'edges')` | `SubgraphArray` | Groups create new subgraphs |
| `NodesAccessor` | `group_by('attr')` | `SubgraphArray` | Node groups become subgraphs |
| `EdgesAccessor` | `group_by('attr')` | `SubgraphArray` | Edge groups become subgraphs |
| `GraphTable` | `group_by('attr')` | `TableArray` | Table groups remain tables |
| `NodesTable` | `group_by('attr')` | `TableArray` | Table groups remain tables |
| `EdgesTable` | `group_by('attr')` | `TableArray` | Table groups remain tables |

## Implementation Plan

### Phase 1: Add GroupBy to Subgraph

**Files to modify:**
- `src/subgraphs/subgraph.rs`
- `python-groggy/src/ffi/subgraphs/subgraph.rs`

**New methods:**
```rust
impl Subgraph {
    /// Group subgraph by node attribute, returning SubgraphArray
    pub fn group_by_nodes(&self, attr_name: &AttrName) -> GraphResult<Vec<Subgraph>> {
        // Implementation: group nodes by attribute value, create subgraph for each group
    }
    
    /// Group subgraph by edge attribute, returning SubgraphArray  
    pub fn group_by_edges(&self, attr_name: &AttrName) -> GraphResult<Vec<Subgraph>> {
        // Implementation: group edges by attribute value, create subgraph for each group
    }
}
```

**Python wrapper:**
```rust
#[pymethods]
impl PySubgraph {
    /// Group subgraph by attribute
    /// Args:
    ///     attr_name: Attribute to group by
    ///     element_type: 'nodes' or 'edges'
    pub fn group_by(&self, attr_name: String, element_type: String) -> PyResult<PySubgraphArray> {
        // Delegate to appropriate Rust method based on element_type
    }
}
```

### Phase 2: Add GroupBy to Accessors

**Files to modify:**
- `python-groggy/src/ffi/core/accessors.rs`
- `python-groggy/src/ffi/storage/accessors.rs`

**NodesAccessor methods:**
```rust
#[pymethods]
impl PyNodesAccessor {
    /// Group nodes by attribute, returning SubgraphArray
    pub fn group_by(&self, attr_name: String) -> PyResult<PySubgraphArray> {
        // Implementation: 
        // 1. Get attribute values for all constrained nodes
        // 2. Group nodes by attribute value
        // 3. Create subgraph for each group (with induced edges)
        // 4. Return as SubgraphArray
    }
}
```

**EdgesAccessor methods:**
```rust
#[pymethods]
impl PyEdgesAccessor {
    /// Group edges by attribute, returning SubgraphArray
    pub fn group_by(&self, attr_name: String) -> PyResult<PySubgraphArray> {
        // Implementation:
        // 1. Get attribute values for all constrained edges  
        // 2. Group edges by attribute value
        // 3. Create subgraph for each group (with connected nodes)
        // 4. Return as SubgraphArray
    }
}
```

### Phase 3: Enhance View Integration

**Files to modify:**
- `python-groggy/src/ffi/api/graph.rs` (view() method)

**Enhanced view pattern:**
```python
# Enable: g.view().group_by('attr_name', 'nodes')
view = g.view()  # Returns PySubgraph  
groups = view.group_by('department', 'nodes')  # Returns SubgraphArray
```

### Phase 4: Remove Old GroupBy from Graph

**Files to modify:**
- `src/api/graph.rs` - Remove `group_nodes_by_attribute()`
- `python-groggy/src/ffi/api/graph.rs` - Remove `group_by()` and `group_nodes_by_attribute()`

**Migration strategy:**
1. Add deprecation warnings first
2. Update all internal usage to new API
3. Update tests and documentation
4. Remove deprecated methods

### Phase 5: Enhanced Array Types

**Ensure SubgraphArray and TableArray support:**

**SubgraphArray methods needed:**
```rust
#[pymethods]
impl PySubgraphArray {
    /// Apply group_by to all subgraphs
    pub fn group_by(&self, attr_name: String, element_type: String) -> PyResult<PySubgraphArray> {
        // Flatten results from grouping each subgraph
    }
    
    /// Convert to TableArray via table() method
    pub fn table(&self) -> PyResult<PyTableArray> {
        // Already implemented
    }
}
```

**TableArray methods needed:**
```rust  
#[pymethods]
impl PyTableArray {
    /// Apply group_by to all tables  
    pub fn group_by(&self, attr_name: String) -> PyResult<PyTableArray> {
        // Group each table and flatten results
    }
    
    /// Convert back to SubgraphArray (if possible)
    pub fn to_subgraphs(&self) -> PyResult<PySubgraphArray> {
        // Convert tables back to subgraphs where possible
    }
}
```

## Files Requiring Changes

### Core Rust Files
1. `src/subgraphs/subgraph.rs` - Add groupby methods
2. `src/api/graph.rs` - Remove old groupby methods
3. `src/errors.rs` - Add new error types if needed

### Python FFI Files  
4. `python-groggy/src/ffi/subgraphs/subgraph.rs` - Add groupby wrapper
5. `python-groggy/src/ffi/core/accessors.rs` - Add accessor groupby methods
6. `python-groggy/src/ffi/storage/accessors.rs` - Add accessor groupby methods  
7. `python-groggy/src/ffi/api/graph.rs` - Remove old groupby methods
8. `python-groggy/src/ffi/storage/subgraph_array.rs` - Enhance array methods
9. `python-groggy/src/ffi/storage/table_array.rs` - Enhance array methods

### Test Files
10. Update all test files that use old groupby API
11. Add comprehensive tests for new groupby functionality

### Documentation Files
12. Update API documentation
13. Update examples and tutorials
14. Update README if needed

## Breaking Changes

### API Changes
- `g.group_by()` method removed → Use `g.view().group_by()` or `g.nodes.group_by()`
- `g.group_nodes_by_attribute()` removed → Use `g.nodes.group_by()`
- Return types changed from `dict` to `SubgraphArray`

### Migration Path
```python
# OLD
groups_dict = g.group_by('dept', 'salary', 'mean')

# NEW  
node_groups = g.nodes.group_by('dept')  # SubgraphArray
# To get aggregation, apply to each subgraph:
salaries = [subgraph.nodes.table()['salary'].mean() for subgraph in node_groups]
```

## Implementation Priority

1. **High Priority**: Subgraph.group_by() - Core new functionality
2. **High Priority**: NodesAccessor.group_by() - Most common use case  
3. **Medium Priority**: EdgesAccessor.group_by() - Complete the accessor API
4. **Medium Priority**: Enhanced array chaining - Better UX
5. **Low Priority**: Remove old Graph.group_by() - Breaking change, do last

## Testing Strategy

### Unit Tests
- Test each new groupby method independently
- Test return type correctness (SubgraphArray vs TableArray)
- Test edge cases (empty groups, missing attributes, etc.)

### Integration Tests  
- Test method chaining: `g.nodes.group_by().table()`
- Test view integration: `g.view().group_by()`
- Test array operations: `groups.filter().sample()`

### Migration Tests
- Ensure old functionality can be replicated with new API
- Performance comparison between old and new implementations
- Backwards compatibility during transition period

## Performance Considerations

### Efficiency Goals
- **SubgraphArray creation**: Minimize memory allocation, reuse graph references
- **Attribute grouping**: Use bulk attribute retrieval (like current implementation)
- **Subgraph construction**: Efficient induced edge computation

### Memory Management
- Share graph references across subgraphs in SubgraphArray
- Use Arc/Rc for efficient cloning
- Lazy computation where possible

## Risk Assessment

### High Risk
- **Breaking changes**: Existing code will need updates
- **Complex implementation**: Groupby logic with subgraph creation is non-trivial

### Medium Risk  
- **Performance regression**: New implementation must be as fast as current
- **API consistency**: New patterns should feel natural and consistent

### Low Risk
- **Table groupby unchanged**: Existing table.group_by() can remain the same
- **Internal graph operations**: Core graph functionality unaffected

## Future Enhancements

### Potential Additions
- **Multi-attribute groupby**: Group by multiple attributes simultaneously
- **Hierarchical groupby**: Nested grouping structures  
- **Aggregation integration**: Built-in aggregation methods on SubgraphArray
- **Lazy evaluation**: Defer subgraph creation until needed

### API Extensions
```python
# Multi-attribute groupby
groups = g.nodes.group_by(['dept', 'level'])

# Built-in aggregation
salary_by_dept = g.nodes.group_by('dept').agg({'salary': 'mean'})

# Hierarchical grouping
nested_groups = g.nodes.group_by('dept').group_by('level')
```

This refactoring will make the groupby functionality more consistent with the rest of the Groggy API while providing more powerful and flexible grouping capabilities.

## Detailed File Dependencies and Impact Analysis

### Current GroupBy Usage Locations

**Core Rust Implementation Files:**
1. `src/api/graph.rs:2276` - `Graph::group_nodes_by_attribute()` - **HIGH IMPACT** 
2. `python-groggy/src/ffi/api/graph.rs:963` - `PyGraph::group_by()` - **HIGH IMPACT**
3. `python-groggy/src/ffi/api/graph.rs:938` - `PyGraph::group_nodes_by_attribute()` - **HIGH IMPACT**

**Table GroupBy Implementations (KEEP):**
4. `src/storage/table/graph_table.rs:992` - `GraphTable::group_by()` - **NO CHANGE**
5. `src/storage/table/nodes.rs:320` - `NodesTable::group_by()` - **NO CHANGE**
6. `src/storage/table/edges.rs:586` - `EdgesTable::group_by()` - **NO CHANGE**
7. `src/storage/table/base.rs:560` - `BaseTable::group_by()` - **NO CHANGE**
8. `python-groggy/src/ffi/core/table.rs:704` - `PyGraphTable::group_by()` - **NO CHANGE**

**Array Iterator Methods (KEEP):**
9. `src/storage/array/iterator.rs:314` - `group_by_source()` - **NO CHANGE**
10. `src/storage/array/lazy_iterator.rs:572` - `group_by_source()` - **NO CHANGE**

### Test Files Requiring Updates

**Generated Test Files:**
1. `documentation/testing/generated_tests/test_graph.py` - Contains tests for:
   - `test_group_by()` function (line 1289)
   - `test_group_nodes_by_attribute()` function (line 1323)
   - Main test runner calls (lines 2758-2760)

2. `documentation/testing/generated_tests/test_nodestable.py` - Contains:
   - `test_group_by()` for NodesTable (line 370) - **KEEP, table groupby unchanged**

3. `documentation/testing/generated_tests/test_basetable.py` - Contains:
   - `test_group_by()` for BaseTable (line 336) - **KEEP, table groupby unchanged**

4. `documentation/testing/generated_tests/test_edgestable.py` - Contains:
   - `test_group_by()` for EdgesTable (line 507) - **KEEP, table groupby unchanged**

### Documentation Files Using GroupBy

**API Documentation:**
1. `documentation/meta_api_discovery/api_meta_summary.json` - API metadata for:
   - `group_by` method (line 551)
   - `group_nodes_by_attribute` method (line 578)

2. `documentation/meta_api_discovery/meta_api_test_results_enhanced_v2.json` - Test results for:
   - Multiple `group_by` tests (lines 146, 430, 773, 1617)
   - `group_nodes_by_attribute` test (line 1627)

3. `documentation/meta_api_discovery/api_discovery_results_enhanced_v2.json` - Discovery results

**Example Documentation:**
4. `documentation/examples/data-analysis-workflow.md` - **IMPORTANT**: Contains working examples:
   - `users_table.group_by('location').agg({...})` (line 81)
   - `users_table.group_by('occupation').agg({...})` (line 89)
   - `interactions_table.group_by('interaction_type').agg({...})` (line 127)
   - `users_with_performance.group_by('location').agg({...})` (line 189)
   - Multiple other `group_by` calls on tables

### Benchmark and Usage Files

**Performance Testing:**
1. `documentation/development/benchmark_graph_libraries.py:364` - Uses `self.graph.group_by()`
   - **BREAKING CHANGE** - Needs migration to new API

**Phase Examples:**
2. `documentation/development/PHASE_1_2_USAGE_EXAMPLE.py` - Contains:
   - `users_table.group_by(['department'])` (line 133)
   - `large_table.group_by(['department'])` (line 431)
   - **NO CHANGE** - Table groupby remains the same

### Key Library Reference
3. `src/lib.rs:120` - Contains example using `.group_by("department")`
   - **NEEDS UPDATE** - Main library documentation example

### Missing Array Infrastructure

**TableArray Implementation:**
- `python-groggy/src/ffi/storage/table_array.rs` - Referenced but not fully shown
- **NEEDS ENHANCEMENT** - Must support new groupby functionality

**SubgraphArray Enhancement:**
- `python-groggy/src/ffi/storage/subgraph_array.rs` - Exists but needs groupby methods
- **NEEDS ENHANCEMENT** - Add groupby delegation methods

### Migration Impact Assessment

**HIGH RISK CHANGES:**
1. **Graph.group_by() removal** - Direct API breaking change
2. **Benchmark code** - Performance testing code needs updating  
3. **Library examples** - Core documentation examples need updating

**MEDIUM RISK CHANGES:**
1. **Generated tests** - Automated tests need regeneration
2. **API metadata** - Discovery and metadata files need updates
3. **Documentation examples** - Some examples show graph.group_by usage

**LOW RISK CHANGES:**
1. **Table groupby** - All table-level groupby methods remain unchanged
2. **Array iterators** - Iterator groupby methods are unrelated
3. **Most documentation** - Many examples use table.group_by which is unchanged

### Implementation Order for Safety

**Phase A: Add New Functionality (Non-Breaking)**
1. Add `Subgraph::group_by()` methods
2. Add `NodesAccessor::group_by()` and `EdgesAccessor::group_by()` 
3. Enhance `SubgraphArray` and `TableArray` with new methods
4. Add comprehensive tests for new functionality

**Phase B: Update Examples and Documentation (Non-Breaking)**
1. Add new API examples alongside old ones
2. Update documentation to show both patterns
3. Create migration guides

**Phase C: Deprecation Phase (Soft Breaking)**
1. Add deprecation warnings to old `Graph::group_by()` methods
2. Update internal usage to new API
3. Update benchmark code to use new API

**Phase D: Final Removal (Hard Breaking)**
1. Remove deprecated `Graph::group_by()` methods
2. Update all remaining documentation
3. Release as major version bump

This staged approach minimizes risk and provides a clear migration path for users.