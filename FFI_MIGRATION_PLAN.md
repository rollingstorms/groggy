# FFI Migration Plan - Final Cleanup

## Overview

This is the final migration from FFI layer business logic to pure delegation wrappers. Based on comprehensive analysis, we have identified which methods need migration vs simple wrapping.

## Comprehensive FFI Method Analysis

### 📁 `api/graph.rs` (2459 lines) - **NEEDS MAJOR CLEANUP**

#### ✅ Basic Operations (Core exists - needs wrapping)
- `add_node` ✓
- `add_nodes` ✓ 
- `add_edge` ✓
- `add_edges` ✓
- `node_ids` ✓
- `edge_ids` ✓

#### ⚠️ Attribute Operations (Mixed status)
- `set_node_attrs` ✅ (fixed bulk operation)
- `get_node_attrs` ✅ (fixed bulk operation) 
- `set_edge_attrs` ✓ (needs bulk edge support)

#### ✅ Business Logic - **CORE EXISTS** (just needs FFI wrapper)
- `filter_nodes` → ✅ **CORE HAS**: `find_nodes` + `filter_nodes` in query.rs  
- `filter_edges` → ✅ **CORE HAS**: `find_edges` + `filter_edges` in query.rs
- `group_by` → ✅ **CORE HAS**: table.rs:623
- `shortest_path` → ✅ **CORE HAS**: graph.rs:1311 + traversal.rs:323  
- `aggregate` → ✅ **CORE HAS**: graph.rs:1545,1630 + table.rs:649
- `connected_components` → ✅ **CORE HAS**: graph.rs:1335 + traversal.rs:468

#### ✅ Matrix Operations - **CORE EXISTS** (just needs FFI wrapper)
- `adjacency_matrix` → ✅ **CORE HAS**: graph.rs:1783
- `weighted_adjacency_matrix` → ✅ **CORE HAS**: graph.rs:1788
- `dense_adjacency_matrix` → ✅ **CORE HAS**: graph.rs:1797
- `sparse_adjacency_matrix` → ✅ **CORE HAS**: graph.rs:1804
- `laplacian_matrix` → ✅ **CORE HAS**: graph.rs:1811

#### ✅ Graph Analysis - **CORE EXISTS** (just needs FFI wrapper)  
- `neighbors` → ✅ **CORE HAS**: graph.rs:866
- `degree` → ✅ **CORE HAS**: graph.rs:845
- `neighborhood` → ✅ **CORE HAS**: graph.rs:1366

#### 🔴 Missing from Core (needs implementation first)
- `resolve_string_id_to_node` → ❌ **MISSING** - implement in core
- `group_nodes_by_attribute` → ❌ **MISSING** - probably not needed
- `adjacency` → ❌ **MISSING** - implement in core  
- `transition_matrix` → ❌ **MISSING** - implement in core
- `in_degree` → ❌ **MISSING** - implement in core
- `out_degree` → ❌ **MISSING** - implement in core  
- `add_graph` → ❌ **MISSING** - implement in core
- `adjacency_matrix_to_graph_matrix` → ❌ **HELPER** - possibly remove

### 📁 `api/graph_query.rs` - **DUPLICATE METHODS**
- `filter_nodes` **DUPLICATE** → should call rust core, not reimplement
- `filter_edges` **DUPLICATE** → should call rust core, not reimplement
- `filter_subgraph_nodes` → not needed, remove
- `aggregate` **DUPLICATE** → consolidate with main graph.rs
- `execute` → query execution logic
- `aggregate_custom_nodes` → migrate to core

### 📁 `api/graph_version.rs` - **NEEDS CORE IMPLEMENTATION**
- `restore_snapshot` → ❌ **MISSING** - needs implementation in core first
- `get_history` → ❌ **MISSING** - needs core support

### 📁 `core/accessors.rs` - **SUBGRAPH OPERATIONS** ✅
**Note**: These are all subgraph operations, not main graph - WORKING
- `attributes` → ✅ subgraph attribute access - WORKING
- `table` → ✅ subgraph table conversion - WORKING
- `all` → ✅ subgraph "select all" - WORKING  
- `_get_node_attribute_column` → ✅ helper method - WORKING
- `_get_edge_attribute_column` → ✅ helper method - WORKING

### 📁 `core/table.rs` - **TABLE CONVERSION** ✅  
- `from_graph_nodes` → ✅ **CORE HAS**: table.rs:225
- `from_graph_edges` → ✅ **CORE HAS**: table.rs:253

## Implementation Strategy

### Phase 1: File Reorganization ✅ PARTIALLY COMPLETE
Current modular structure already exists:

```
python-groggy/src/ffi/api/
├── graph.rs (main class + basic ops) - 2459 lines - NEEDS BREAKUP
├── graph_attributes.rs (attribute operations) - ✅ CREATED  
├── graph_version.rs (version control) - ✅ EXISTS (354 lines)
├── graph_query.rs (query/filter operations) - NEEDS CREATION
├── graph_analysis.rs (algorithms) - NEEDS CREATION  
├── graph_io.rs (serialization/import/export) - NEEDS CREATION
```

**Priority**: Extract attribute methods from main `graph.rs` to `graph_attributes.rs` module.

### Phase 2: Core Method Migration
Move business logic from FFI to core, following the pattern:
1. Implement algorithm in `src/core/` or `src/api/graph.rs`
2. Replace FFI business logic with delegation call
3. Use `py.allow_threads()` for CPU-intensive operations
4. Proper error conversion with `map_err(graph_error_to_py_err)`

### Phase 3: Pure Wrapper Creation
For methods that already exist in core, create simple wrappers:
```rust
fn some_method(&self, py: Python, args: Type) -> PyResult<ReturnType> {
    let result = py.allow_threads(|| {
        self.inner.borrow().some_method(args).map_err(graph_error_to_py_err)
    })?;
    Ok(result.into())  // Convert to Python type
}
```

## Priority Order - **CORE IMPLEMENTATION STATUS**

### 🚨 **IMMEDIATE** - Remove Duplicates (Core exists)  
1. **✅ filter_nodes/filter_edges** - Remove FFI duplicates, use core implementations
2. **✅ connected_components** - Remove FFI version, use core (graph.rs:1335)  
3. **✅ aggregate** - Consolidate duplicates, use core implementations

### 🔥 **HIGH PRIORITY** - Create FFI Wrappers (Core exists)
**Matrix Operations** (5 methods - all have core implementations):
1. **✅ adjacency_matrix** - Core: graph.rs:1783 
2. **✅ weighted_adjacency_matrix** - Core: graph.rs:1788  
3. **✅ dense_adjacency_matrix** - Core: graph.rs:1797
4. **✅ sparse_adjacency_matrix** - Core: graph.rs:1804
5. **✅ laplacian_matrix** - Core: graph.rs:1811

**Graph Analysis** (3 methods - core exists):
1. **✅ neighbors** - Core: graph.rs:866
2. **✅ degree** - Core: graph.rs:845  
3. **✅ neighborhood** - Core: graph.rs:1366

**Query/Path Operations** (2 methods - core exists):
1. **✅ shortest_path** - Core: graph.rs:1311 + traversal.rs:323
2. **✅ group_by** - Core: table.rs:623

### ✅ **MEDIUM PRIORITY** - Recently Implemented in Core!
1. **✅ in_degree** - **JUST IMPLEMENTED** in core (graph.rs:866)
2. **✅ out_degree** - **JUST IMPLEMENTED** in core (graph.rs:887)  
3. **✅ adjacency** - **JUST IMPLEMENTED** in core (graph.rs:1878) - alias for adjacency_matrix
4. **✅ add_graph** - **JUST IMPLEMENTED** in core (graph.rs:348)

### ❌ **LOW PRIORITY** - Still Missing from Core  
1. **❌ transition_matrix** - Missing from core, implement first 
2. **❌ resolve_string_id_to_node** - Missing from core, implement first

### 🔻 **LOW PRIORITY** - Version Control & Helpers
1. **❌ restore_snapshot** - Missing core implementation
2. **❌ get_history** - Missing core implementation  
3. **✅ Table operations** - Already working (from_graph_nodes/edges)
4. **Remove unnecessary** - `group_nodes_by_attribute`, `adjacency_matrix_to_graph_matrix`

## **UPDATED IMPACT ESTIMATE** (After Recent Core Implementations)
- **✅ Ready for FFI wrapping**: **19 methods** (core exists) ⬆️ +4 methods!
- **❌ Need core implementation**: **4 methods** (missing from core) ⬇️ -4 methods!
- **🗑️ Remove/consolidate**: **6 methods** (duplicates + unnecessary)  
- **Effort split**: **83% wrapping**, **17% core implementation** ⬆️ Much better ratio!

### **🎉 MAJOR PROGRESS**: 
We just implemented 4 critical missing methods in core:
- `in_degree`, `out_degree` - Essential degree operations
- `adjacency` - Simple matrix alias  
- `add_graph` - Graph merging functionality

**Only 4 methods still missing from core** vs 8 before!

## File Breakdown Target

Current `graph.rs`: ~2500 lines → Target: ~7 files of 300-400 lines each

This will make the codebase much more maintainable and easier to understand, with clear separation of concerns following the three-tier architecture principle.

## Success Metrics

- [x] **Critical FFI Fix**: Resolved AttrValue match pattern error - bulk operations now working (94.2% test success)
- [x] **Method Classification**: Completed analysis of 40+ FFI methods (21 need wrapping, 19 need migration)
- [x] **Migration Plan**: Created comprehensive roadmap with priority ordering
- [ ] All FFI methods are pure delegation (no business logic)
- [ ] Core test coverage maintained at 95%+
- [ ] No performance regressions in bulk operations
- [ ] Clear module boundaries with single responsibility
- [ ] Comprehensive test suite maintains 94%+ success rate

## Current Status (2024-08-28)

✅ **COMPLETED**:
- Fixed critical FFI bulk operations architecture
- Comprehensive method classification and analysis  
- Migration plan document with priority ordering
- Identified existing modular file structure

🔄 **IN PROGRESS**:
- File reorganization (graph_attributes.rs created)
- Version control wrapper methods  

⏳ **NEXT STEPS**:
1. Extract attribute methods from main graph.rs (reduce from 2459 lines)
2. Create pure wrapper methods for version control operations
3. Migrate query/filter business logic to core implementations