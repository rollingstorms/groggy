# Storage View Unification Plan

## Overview

This document outlines the comprehensive plan to unify the three main storage view classes (GraphArray, GraphMatrix, and GraphTable) with consistent architecture and API design.

## Current State Analysis

### Issues Identified
1. **GraphArray**: ✅ Well-designed, lives in core, has statistical operations
2. **GraphMatrix**: ❌ Currently specialized for adjacency matrices (square), not general collection
3. **GraphTable**: ❌ Lives in FFI, should be in core for consistency  
4. **Architecture**: ❌ Mixed core/FFI responsibilities, hard to debug/maintain

### Current File Locations
- `src/core/array.rs` - GraphArray (✅ good)
- `src/core/adjacency.rs` - GraphMatrix (❌ should be `src/core/matrix.rs`)
- `src/ffi/core/table.rs` - GraphTable (❌ should be `src/core/table.rs`)

## Unified Architecture Design

### Core Hierarchy (All in `src/core/`)

```rust
// 1. Base: GraphArray (single column, typed)
pub struct GraphArray {
    values: Vec<AttrValue>,
    name: Option<String>,           // Column name
    cached_stats: RefCell<Stats>,   // Lazy stats
}

// 2. Collection: GraphMatrix (homogeneous columns)  
pub struct GraphMatrix {
    columns: Vec<GraphArray>,       // All same type
    dtype: AttrValueType,           // Enforced type
    properties: MatrixProperties,   // is_square, is_numeric, etc.
}

// 3. Collection: GraphTable (heterogeneous columns)
pub struct GraphTable {  
    columns: Vec<GraphArray>,       // Mixed types allowed
    index: Option<GraphArray>,      // Row index/labels
    metadata: TableMetadata,        // Source info, etc.
}
```

### Key Design Principles

1. **Unified Column Storage**: Both Matrix and Table are collections of GraphArrays
2. **Type Safety**: Matrix enforces homogeneous types, Table allows mixed
3. **Lazy Evaluation**: Stats computed on demand, cached intelligently
4. **View vs Copy**: Clear semantics for performance-critical operations
5. **Integration**: Native integration with graph operations and attribute system

## Critical Missing Pieces

### Memory Management & Ownership
- How do arrays share data with the graph pool?
- When do we copy vs create views?
- Reference counting for shared columns?

### Type Coercion Strategy
- What happens when mixing int64 and float64 in operations?
- Automatic promotion rules?
- Error vs warning vs silent conversion?

### Indexing Strategy
- Row-major vs column-major storage?
- How do you efficiently get both row[i] and column[j]?
- Sparse vs dense representations?

### Integration Points
- How do tables connect to graph attribute storage?
- Lazy loading from graph pool?
- Dirty tracking for cached operations?

### Iterator Patterns
- Row iterators, column iterators, cell iterators
- Parallel iteration support?
- Memory-efficient streaming for large tables?

### Error Handling Strategy
- Consistent error types across all three storage views
- Graceful degradation for missing data
- Validation at construction vs operation time?

## Comprehensive API Specification

### GraphArray Core API
```rust
impl GraphArray {
    // Construction
    pub fn from_vec(values: Vec<AttrValue>) -> Self
    pub fn from_graph_attribute(graph: &Graph, attr: &str, entities: &[Id]) -> Self
    pub fn with_name(self, name: String) -> Self
    
    // Access & Indexing  
    pub fn get(&self, index: usize) -> Option<&AttrValue>
    pub fn len(&self) -> usize
    pub fn is_empty(&self) -> bool
    pub fn iter(&self) -> impl Iterator<Item = &AttrValue>
    
    // Statistics (cached)
    pub fn mean(&self) -> Option<f64>
    pub fn median(&self) -> Option<f64> 
    pub fn std(&self) -> Option<f64>
    pub fn min(&self) -> Option<&AttrValue>
    pub fn max(&self) -> Option<&AttrValue>
    pub fn sum(&self) -> Option<f64>
    pub fn count(&self) -> usize
    pub fn unique(&self) -> GraphArray
    pub fn value_counts(&self) -> HashMap<AttrValue, usize>
    
    // Operations
    pub fn filter(&self, predicate: impl Fn(&AttrValue) -> bool) -> GraphArray
    pub fn map(&self, transform: impl Fn(&AttrValue) -> AttrValue) -> GraphArray
    pub fn sort(&self) -> GraphArray
    pub fn reverse(&self) -> GraphArray
    
    // Type Operations
    pub fn dtype(&self) -> AttrValueType
    pub fn can_convert_to(&self, target_type: AttrValueType) -> bool
    pub fn convert_to(&self, target_type: AttrValueType) -> Result<GraphArray>
}
```

### GraphMatrix API
```rust
impl GraphMatrix {
    // Construction
    pub fn from_arrays(arrays: Vec<GraphArray>) -> Result<Self>  // Type checking
    pub fn from_graph_attributes(graph: &Graph, attrs: &[&str], entities: &[Id]) -> Result<Self>
    pub fn zeros(rows: usize, cols: usize, dtype: AttrValueType) -> Self
    pub fn identity(size: usize) -> Self
    
    // Properties
    pub fn shape(&self) -> (usize, usize)
    pub fn dtype(&self) -> AttrValueType
    pub fn is_square(&self) -> bool
    pub fn is_symmetric(&self) -> bool
    pub fn is_numeric(&self) -> bool
    
    // Access  
    pub fn get(&self, row: usize, col: usize) -> Option<&AttrValue>
    pub fn get_row(&self, row: usize) -> Option<&GraphArray>
    pub fn get_column(&self, col: usize) -> Option<&GraphArray>
    pub fn iter_rows(&self) -> impl Iterator<Item = &GraphArray>
    pub fn iter_columns(&self) -> impl Iterator<Item = &GraphArray>
    
    // Linear Algebra
    pub fn transpose(&self) -> GraphMatrix
    pub fn multiply(&self, other: &GraphMatrix) -> Result<GraphMatrix> // Phase 5
    pub fn inverse(&self) -> Result<GraphMatrix> // Phase 5
    pub fn determinant(&self) -> Option<f64> // Phase 5
    pub fn eigen(&self) -> Vec<f64>  // If numeric // Phase 5 // todo: eigenvalues and eigenvectors // could be complex to implement from scratch, lanczos?
    pub fn rank(&self) -> usize // Phase 5
    pub fn svd(&self) -> (GraphMatrix, GraphMatrix, GraphMatrix) // Phase 5 // todo: singular value decomposition
    pub fn qr(&self) -> (GraphMatrix, GraphMatrix) // Phase 5 // todo: QR decomposition
    pub fn cholesky(&self) -> GraphMatrix // Phase 5 // todo: Cholesky decomposition
    
    // Statistical Operations (inherit from GraphArray)
    pub fn sum_axis(&self, axis: Axis) -> GraphArray  // Sum rows or columns
    pub fn mean_axis(&self, axis: Axis) -> GraphArray
    pub fn std_axis(&self, axis: Axis) -> GraphArray
}
```

### GraphTable API (Pandas-like)
```rust
impl GraphTable {
    // Construction
    pub fn from_arrays(arrays: Vec<GraphArray>, columns: Option<Vec<String>>) -> Self
    pub fn from_graph_nodes(graph: &Graph, nodes: &[NodeId], attrs: Option<&[&str]>) -> Self
    pub fn from_graph_edges(graph: &Graph, edges: &[EdgeId], attrs: Option<&[&str]>) -> Self
    pub fn from_subgraph(subgraph: &Subgraph) -> Self
    
    // Properties
    pub fn shape(&self) -> (usize, usize)
    pub fn columns(&self) -> &[String]
    pub fn dtypes(&self) -> HashMap<String, AttrValueType>
    pub fn index(&self) -> Option<&GraphArray>
    pub fn memory_usage(&self) -> usize
    
    // Access & Indexing (like pandas)
    pub fn get_column(&self, name: &str) -> Option<&GraphArray>
    pub fn get_row(&self, index: usize) -> Option<HashMap<String, &AttrValue>>
    pub fn iloc(&self, row: usize) -> Option<HashMap<String, &AttrValue>>  // Position-based
    pub fn loc(&self, label: &AttrValue) -> Option<HashMap<String, &AttrValue>>  // Label-based
    
    // Slicing & Selection  
    pub fn select(&self, columns: &[&str]) -> GraphTable
    pub fn filter_rows(&self, predicate: impl Fn(&HashMap<String, &AttrValue>) -> bool) -> GraphTable
    pub fn head(&self, n: usize) -> GraphTable
    pub fn tail(&self, n: usize) -> GraphTable
    pub fn sample(&self, n: usize) -> GraphTable
    
    // Statistical Operations
    pub fn describe(&self) -> GraphTable  // Summary statistics
    pub fn group_by(&self, column: &str) -> GroupBy
    pub fn aggregate(&self, ops: HashMap<String, AggregateOp>) -> HashMap<String, AttrValue>
    pub fn pivot_table(&self, index: &str, columns: &str, values: &str) -> GraphTable
    
    // Data Manipulation
    pub fn sort_by(&self, column: &str, ascending: bool) -> GraphTable  // todo: multi-column sort
    pub fn drop_duplicates(&self) -> GraphTable
    pub fn fillnans(&self, value: AttrValue) -> GraphTable
    pub fn dropna(&self) -> GraphTable
    pub fn merge(&self, other: &GraphTable, on: &str, how: JoinType) -> GraphTable
    
    // Export/Import
    pub fn to_csv(&self, path: &str) -> Result<()>
    pub fn to_json(&self) -> String
    pub fn to_dict(&self) -> HashMap<String, Vec<AttrValue>>
    pub fn to_arrays(&self) -> Vec<GraphArray>  // Convert back to arrays
    
    // Integration with Graph Operations
    pub fn apply_to_graph(&self, graph: &mut Graph, entity_col: &str) -> Result<()>  // Write back
    pub fn join_graph_data(&self, graph: &Graph, on: &str) -> GraphTable  // Enrich from graph
}
```

## Implementation Strategy

### Phase 1: Core Foundation ✅
1. **Create `src/core/matrix.rs`** - Move GraphMatrix from adjacency.rs, refactor as collection of GraphArrays ✅
2. **Create `src/core/table.rs`** - Move GraphTable from FFI to core ✅
3. **Unify Error Handling** - Consistent error types across all storage views ✅
4. **Add Missing GraphArray methods** - Fill gaps identified in API spec ✅

### Phase 2: Integration Layer ✅ **COMPLETED**
1. **Create `python-groggy/src/ffi/core/matrix.rs`** - ✅ Created dedicated Python bindings for GraphMatrix with full API 
2. **Fix Compilation Issues** - ✅ Resolved AdjacencyMatrix enum changes and Python binding errors
3. **Build System** - ✅ `maturin develop --release` builds successfully with warnings only
4. **Unified Builder Patterns** - ✅ Implemented `gr.array()`, `gr.table()`, `gr.matrix()` constructors with full Python integration
5. **Statistical Operations** - ✅ Added working `sum_axis()`, `mean_axis()`, `std_axis()` methods to PyGraphMatrix
6. **Graph Pool Integration** - ✅ Sophisticated columnar storage with AttributeColumn and memory pooling already integrated
7. **Memory Management** - ✅ AttributeMemoryPool with string/float/byte pool reuse already implemented
8. **Caching Strategy** - ✅ CachedStats with smart invalidation already working across operations

### Phase 3: FFI Wrapper Layer
1. **Python Bindings** - Thin wrappers around core functionality
2. **Display Integration** - Consistent `__repr__` and `_repr_html_` - implemented in the py legacy code graph_table_legacy.py - for Array, Matrix, and Table
3. **Indexing Operations** - Python-style `[]` operator support
4. **Iterator Protocol** - Python iteration support

### Phase 4: Rich API Implementation
1. **Statistical Operations** - Full pandas-like statistical API
2. **Data Manipulation** - Sorting, filtering, grouping, joining
3. **Linear Algebra** - Matrix operations for GraphMatrix
4. **Export/Import** - CSV, JSON, integration with external formats

### Phase 5: Advanced Linear Algebra
1. **Linear Algebra** - Matrix operations for GraphMatrix
2. **Advanced Linear Algebra** - Advanced linear algebra operations for GraphMatrix
3. **Advanced Graph Operations** - Advanced graph operations for GraphMatrix

### Phase 6: Advanced Visualization Module - Viz
1. **Visualization** - Advanced visualization for graphs built in rust and export as JS for embedding in HTML
2. **Scaled Up Visualization** - Advanced visualization for graphs built in rust and export as JS for embedding in HTML



## File Migration Plan

### Current Structure
```
src/
  core/
    array.rs          ✅ Keep
    adjacency.rs      ❌ Rename to matrix.rs, refactor
  ffi/
    core/
      table.rs        ❌ Move to src/core/table.rs
```

### Target Structure  
```
src/
  core/
    array.rs          ✅ GraphArray (enhanced)
    matrix.rs         🆕 GraphMatrix (collection of arrays)
    table.rs          🆕 GraphTable (collection of arrays) 
    adjacency.rs      ✅ Keep for specialized adjacency operations
  ffi/
    core/
      array.rs        ✅ PyGraphArray (thin wrapper)
      matrix.rs       🆕 PyGraphMatrix (thin wrapper)
      table.rs        🆕 PyGraphTable (thin wrapper)
```

## Next Steps (Phase 2 Continuation)

1. **Complete PyGraphMatrix Statistical Operations** - Implement sum_axis, mean_axis, std_axis methods in core GraphMatrix
2. **Implement Builder Patterns** - Create `gr.array()`, `gr.table()`, `gr.matrix()` unified constructors
3. **Graph Pool Integration** - Efficient attribute column loading from graph storage
4. **Memory Management** - Reference counting and copy vs view semantics
5. **Comprehensive Testing** - Ensure all three storage views work together seamlessly

## Current Status (August 2025)

✅ **Completed:**
- Core GraphMatrix architecture in `src/core/matrix.rs`
- Core GraphTable architecture in `src/core/table.rs` 
- Python bindings for GraphMatrix in `python-groggy/src/ffi/core/matrix.rs`
- Build system successfully compiling with `maturin develop --release`
- AdjacencyMatrix compatibility layer (temporarily disabled pending Phase 1 completion)

🔧 **In Progress:**
- Statistical operations implementation in core GraphMatrix
- Complete Python API for PyGraphMatrix methods

📋 **Pending:**
- Graph pool integration for efficient data loading
- Unified builder patterns
- Memory management improvements
- Advanced linear algebra operations (Phase 5)

---

*This document serves as the blueprint for the storage view unification project. Update as implementation progresses.*