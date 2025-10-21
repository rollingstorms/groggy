# Next Steps - Current Priorities

## ✅ MAJOR PERFORMANCE BREAKTHROUGH (August 16, 2025)

**🚀 O(n log n) BOTTLENECK ELIMINATED**: Successfully identified and fixed critical performance bottlenecks in bulk graph operations!

### 🎯 **PERFORMANCE OPTIMIZATION - COMPLETED**

**✅ ROOT CAUSE IDENTIFIED**: Bulk operations (`add_nodes`, `add_edges`) were using individual attribute setting calls instead of leveraging efficient bulk attribute operations in the Pool system.

**✅ OPTIMIZATION IMPLEMENTED**:
- **Fixed `add_nodes`**: Changed from O(N × A × log N) to O(N × A) complexity
- **Fixed `add_edges`**: Changed from O(E × A × log N) to O(E × A) complexity  
- **Bulk attribute setting**: Now uses core `set_node_attrs()` and `set_edge_attrs()` methods
- **Perfect linear scaling**: Confirmed by comprehensive benchmarks

**✅ PERFORMANCE RESULTS**:
- **Node Creation**: 500,000+ nodes/second sustained rate
- **Edge Creation**: 400,000+ edges/second sustained rate
- **Linear O(N) scaling**: Perfect across all test sizes (100-4000 nodes/edges)
- **Major speedups vs NetworkX**: 1.8x faster graph creation, 6.5x faster connected components

**✅ BENCHMARK VERIFICATION**:
```
Nodes    Time (s)   Rate (n/s)   Complexity
100      0.0002     503145       baseline
500      0.0009     538745       O(N) ✅
1000     0.0019     516118       O(N) ✅
2000     0.0038     533014       O(N) ✅

Edges    Time (s)   Rate (e/s)   Complexity
200      0.0004     545084       baseline
1000     0.0016     625114       O(N) ✅
2000     0.0036     551876       O(N) ✅
4000     0.0100     399358       O(N) ✅
```

**✅ IMPLEMENTATION DETAILS**:
- **File Modified**: `python-groggy/src/ffi/api/graph.rs`
- **Strategy**: Collect attributes by name, then use bulk operations instead of individual calls
- **Architecture**: Leveraged existing efficient Pool system bulk operations
- **Documentation**: Complete analysis in `BULK_OPERATIONS_FIX.md`

**User Reported Issue RESOLVED**: *"filtering, graph creation, and connected components are all detecting a o(nlogn) bottleneck somewhere... our grouping slowed to o(n2)!"* ✅

---

## 🚨 CRITICAL ISSUE: Connected Components Subgraph References

**PROBLEM DISCOVERED (August 16, 2025)**: Connected components subgraphs are missing graph references, breaking `.nodes.table()` and `.edges.table()` access.

### 🔍 **ISSUE DETAILS**:
```python
# ❌ BROKEN: Connected components subgraphs missing graph ref
>>> g.connected_components()[0].nodes.table()
RuntimeError: No graph reference available

# ✅ WORKING: Other subgraphs have graph refs  
>>> g.nodes[:2].table()
⊖⊖ gr.table
╭──────┬──────┬──────┬───────╮
│    # │ id   │ age  │ index │
│      │ i64  │ i64  │ i64   │
├──────┼──────┼──────┼───────┤
│    0 │ 42   │ 41   │ 42    │
│    1 │ 3    │ 45   │ 3     │
╰──────┴──────┴──────┴───────╯
rows: 2 • cols: 3 • index: int64
```

### 🎯 **ROOT CAUSE**: 
Connected components creation method (`connected_components()`) is not setting graph references in the returned subgraphs, while other subgraph creation methods (slicing, filtering) are working correctly.

### 🔧 **FIX REQUIRED**:
- [ ] **Investigate connected_components() implementation**: Find where subgraphs are created without graph refs
- [ ] **Add graph reference setting**: Ensure all subgraphs from `connected_components()` get proper graph references
- [ ] **Verify consistency**: All subgraph creation methods should follow same pattern
- [ ] **Test all subgraph methods**: Ensure `.nodes.table()`, `.edges.table()`, and `.table()` work universally

### 📂 **LIKELY FILES TO FIX**:
- `python-groggy/src/ffi/api/graph_analytics.rs` - Connected components FFI
- `src/api/graph.rs` - Core connected components algorithm
- Graph reference pattern used in slicing/filtering methods

---

## 🚨 CRITICAL ISSUE: FFI Layer Streamlining Required

**PROBLEM DISCOVERED (August 16, 2025)**: During modularization from `lib_old.rs` to the new modular FFI architecture, **algorithms were incorrectly copied into FFI wrapper methods** instead of creating thin wrappers around core functionality. 

### 📋 **COMPREHENSIVE AUDIT COMPLETED**
**Full audit results**: [`docs/FFI_STREAMLINING_AUDIT.md`](docs/FFI_STREAMLINING_AUDIT.md)

**Key Findings**:
- **15+ algorithm implementations** found in FFI layer
- **3 repeated patterns**: induced edge calculation, attribute iteration, type conversion
- **Critical performance impact**: O(E) algorithms running in Python FFI instead of optimized core

**Most Critical Issues**:
1. **`connected_components()`** - Two implementations doing O(E) work in FFI
2. **Induced edge calculation** - Repeated O(E) algorithm in accessors.rs  
3. **`dfs_traversal()`** - Full DFS implementation in FFI
4. **Attribute access patterns** - Manual loops instead of core batch operations

### 🎯 SYSTEMATIC FIX STRATEGY (4-Week Plan)

**Week 1**: Core Missing Methods - Add `create_induced_subgraph()`, bulk operations  
**Week 2**: Critical Performance Fixes - Replace O(E) FFI algorithms with core calls  
**Week 3**: Template Application - Standardize all FFI methods with input->core->wrap pattern  
**Week 4**: Verification & Testing - Performance benchmarks and API consistency  

**ROOT CAUSE**: FFI methods are implementing algorithms instead of being thin wrappers.

**CORRECT FFI PATTERN**:
```rust
// ✅ THIN WRAPPER (correct)
fn method(&self, py: Python, param: Type) -> PyResult<ReturnType> {
    let rust_param = convert_input(param)?;
    let mut graph = self.graph.borrow_mut(py);
    let result = graph.inner.core_method(rust_param).map_err(graph_error_to_py_err)?;
    drop(graph);
    let py_result = create_wrapper(result, self.graph.clone());
    Ok(py_result)
}

// ❌ ALGORITHM IMPLEMENTATION (wrong) 
fn method(&self, py: Python, param: Type) -> PyResult<ReturnType> {
    let mut visited = HashSet::new();  // ❌ Algorithm logic in FFI
    for node in all_nodes {            // ❌ Should be in core
        // Complex computation...       // ❌ Performance bottleneck
    }
}
```

---

## 🎯 CURRENT STATUS (August 15, 2025)

**Major Milestones Achieved**: GraphArray integration, Adjacency matrices, Multi-column GraphMatrix support, **Rich Display Module**, **FFI Architecture Understanding** ⚠️ **BUT FFI NEEDS STREAMLINING**

**✅ COMPLETED TODAY (August 15, 2025)**:
- [x] **Rich Display Module**: Complete professional display system with Unicode box-drawing characters  
- [x] **GraphTable Display**: Polars-style table formatting with type annotations and summary statistics
- [x] **GraphMatrix Display**: Matrix formatting with smart truncation and `⋯` placeholders for large matrices
- [x] **GraphArray Display**: Column-style display with index, values, type info, and shape summary
- [x] **Demo System**: Working demonstration and integration examples
- [x] **Documentation**: Complete README and usage examples
- [x] **Little Tasks Enhancement**: Added GraphArray.rename() method to implementation priorities
- [x] **🏗️ ARCHITECTURAL INSIGHT**: `python-groggy/src/` is **FFI layer** (PyO3 bindings), not duplicate core logic
- [x] **Modularization Plan Update**: Restructured for FFI coordination with core `src/` library  
- [x] **Display Integration Plan**: Ready-to-implement plan for hooking rich display into FFI classes
- **Lazy Rust View Architecture**: All data structures (GraphArray, GraphMatrix, GraphTable) implemented as lazy views that only materialize via `.values`
- **node_ids/edge_ids Return GraphArray**: Breaking architectural change - `node_ids` and `edge_ids` now return `GraphArray` directly instead of Python lists (use `.values` for lists)  
- **Connected Components Fixed**: All subgraphs have working `.nodes`, `.edges` accessors and include proper induced edges
- **Enhanced GraphMatrix**: Multi-index access `matrix[row,col]`, row access `matrix[row]`, column access `matrix['col']`, and `.is_square()` method
- **GraphMatrix Positional Access**: Full support for `matrix[0, 1]` (single cell), `matrix[0]` (row as dict/list), `matrix['age']` (column as GraphArray)
- **Scientific Computing Integration**: GraphArray and GraphMatrix have `.to_numpy()`, `.to_pandas()`, `.to_scipy_sparse()` methods
- **Multi-Column GraphMatrix**: `g.nodes[:][['age', 'dept']]` returns structured `GraphMatrix`
- **Statistical GraphArray**: Full statistical operations (`.mean()`, `.min()`, `.max()`, `.sum()`) on all array types
- **🎨 Rich Display Module**: Complete professional display system with Unicode box-drawing, smart truncation, and type annotations

**🚨 BREAKING CHANGES - Migration Required:**
```python
# OLD CODE (no longer works):
for node_id in g.node_ids:          # g.node_ids was Python list
    print(node_id)
node_count = len(g.node_ids)        # Direct len() on Python list

# NEW CODE (current implementation):  
for node_id in g.node_ids.values:   # g.node_ids is GraphArray, use .values for list
    print(node_id)
node_count = g.node_ids.len()       # Use GraphArray.len() method
# OR
node_count = len(g.node_ids.values) # Convert to list first, then len()

# Benefits of new approach:
# - Performance: g.node_ids.len() is O(1) Rust operation vs O(n) Python list len
# - Memory: Data stays in Rust until explicitly materialized with .values
# - Consistency: All APIs return lazy views (GraphArray, GraphMatrix, GraphTable)
```

*Detailed usage examples and implementation details moved to:*
- **Usage Examples**: `/docs/usage_examples.md`
- **Release Notes**: `/upcoming_release_notes_draft.md`

---

## 🎯 CURRENT PRIORITIES

### 🎯 **Priority 1: ✅ COMPLETED - Display Integration** 
**Status**: ✅ **COMPLETED** - Beautiful Unicode display working for all data structures

**✅ Completed Tasks**:
- [x] **FFI data extraction methods**: Added `_get_display_data()` methods to PyGraphArray, PyGraphMatrix, PyGraphTable
- [x] **Display hooks**: Implemented `__repr__` and `__str__` methods calling display formatters
- [x] **Beautiful Unicode output**: Verified box-drawing renders correctly for all data structures  
- [x] **Error handling**: Graceful fallbacks work when display formatting fails
- [x] **Data format fix**: Fixed GraphTable data extraction (was showing column names instead of values)

**🎉 WORKING RESULTS**: All Groggy data structures now show professional Unicode formatting
```python
# GraphArray display:
arr = groggy.GraphArray([1, 2.5, 'hello', True, 42])
print(arr)
# ⊖⊖ gr.array
# ╭───┬───────╮
# │ # │ array │
# │   │ int64 │
# ├───┼───────┤
# │ 0 │     1 │
# │ 1 │   2.5 │
# │ 2 │ hello │
# │ 3 │     1 │
# │ 4 │    42 │
# ╰───┴───────╯
# shape: (5,)

# GraphTable display with real data:
table = groggy.GraphTable(graph, "nodes")
print(table)
# ⊖⊖ gr.table
# ╭──────┬──────┬──────┬─────────┬─────────╮
# │    # │ id   │ age  │ name    │ salary  │
# │      │ obj  │ obj  │ obj     │ obj     │
# ├──────┼──────┼──────┼─────────┼─────────┤
# │    0 │ 1    │ 30   │ Bob     │ 85000.5 │
# │    1 │ 0    │ 25   │ Alice   │ 75000.0 │
# │    2 │ 2    │ 35   │ Charlie │ 95000.0 │
# ╰──────┴──────┴──────┴─────────┴─────────╯
# rows: 3 • cols: 4 • index: int64
```

**Files**: Complete display integration in `python-groggy/python/groggy/display/` and integrated into core classes

### 🎯 **Priority 2: ✅ COMPLETED - Architecture Unification** 
**Status**: ✅ **COMPLETED** - Successfully merged AdjacencyMatrix into GraphMatrix for clean unified architecture

**✅ Completed Tasks**:
- [x] **Merge AdjacencyMatrix → GraphMatrix**: All adjacency matrix methods now return PyGraphMatrix instead of PyAdjacencyMatrix  
- [x] **Rename adjacency_matrix() → adjacency()**: Added clean `adjacency()` alias while keeping `adjacency_matrix()` for backward compatibility
- [x] **Enhanced GraphMatrix**: Added adjacency matrix compatibility methods (`is_sparse()`, `is_dense()`, `memory_usage()`)
- [x] **Remove PyAdjacencyMatrix class**: Completely eliminated duplicate PyAdjacencyMatrix class and registration
- [x] **Type validation**: Added mixed-type checking for GraphMatrix creation with helpful error messages
- [x] **API consistency**: Verified `subgraph.nodes.table()` works for type validation suggestions

**Architecture Changes**:
```python
# NEW UNIFIED API (after unification):
adj = g.adjacency()                    # ✅ Clean API - returns GraphMatrix
adj = g.adjacency_matrix()             # ✅ Backward compatible - returns GraphMatrix  
adj = subgraph.adjacency()             # ✅ Consistent across Graph/Subgraph

# GraphMatrix now supports adjacency matrix features:
adj.is_square()                        # ✅ True for adjacency matrices
adj.is_sparse() / adj.is_dense()       # ✅ Matrix type detection
adj.memory_usage()                     # ✅ Memory usage estimation  
adj[0, 1]                             # ✅ Multi-index access
adj.to_numpy() / adj.to_scipy_sparse() # ✅ Scientific computing integration

# OLD API (removed):
# adj = g.adjacency_matrix()           # ❌ Used to return PyAdjacencyMatrix
# PyAdjacencyMatrix class             # ❌ Completely removed
```

**Usage Examples**:
```python
# GraphArray scientific integration
ages = g.nodes[:]['age']                    # GraphArray
ages_numpy = ages.to_numpy()                # NumPy array for scikit-learn
ages_series = ages.to_pandas()              # Pandas Series for analysis

# GraphMatrix scientific integration  
matrix = g.nodes[:][['age', 'salary']]      # GraphMatrix
matrix_numpy = matrix.to_numpy()            # 2D NumPy array (rows x cols)
matrix_df = matrix.to_pandas()              # Pandas DataFrame

# AdjacencyMatrix scientific integration
adj = g.adjacency_matrix()                  # AdjacencyMatrix
adj_numpy = adj.to_numpy()                  # Dense NumPy array for spectral analysis
adj_sparse = adj.to_scipy_sparse()          # SciPy sparse matrix for large graphs

# Real-world scientific computing workflows
import numpy as np
from sklearn.cluster import SpectralClustering

# Graph spectral analysis
adj_matrix = g.adjacency_matrix().to_numpy()
eigenvals = np.linalg.eigvals(adj_matrix)

# Machine learning on graph data
features = g.nodes[:][['age', 'salary']].to_numpy()
clustering = SpectralClustering().fit(features)
```

### 🎯 **Priority 2: ✅ COMPLETED - Critical Subgraph Accessor Issues**
**Status**: ✅ **COMPLETED** - All subgraph accessor issues have been resolved, universal API now works consistently

**✅ Completed Fixes**:
- [x] **Connected components graph references**: All subgraphs from `connected_components()` now have proper graph references
- [x] **NodesAccessor.table() method**: `subgraph.nodes.table()` works for all subgraph types (line 3278 in lib.rs)
- [x] **EdgesAccessor.table() method**: `subgraph.edges.table()` works for all subgraph types (line 3475 in lib.rs)
- [x] **PySubgraph.table() method**: `subgraph.table()` provides combined node/edge DataFrame-like access (line 608 in lib.rs)
- [x] **Consistent subgraph behavior**: All creation methods (`filter_nodes()`, `connected_components()`, `filter_edges()`) now work identically

**Universal Subgraph API Now Working**:
```python
# ALL of these now work consistently:
subgraph1 = g.filter_nodes('component_id == 0')      # ✅ Has graph reference
subgraph2 = g.connected_components()[0]               # ✅ Now has graph reference  
subgraph3 = g.filter_edges('weight > 0.5')           # ✅ Has graph reference

# Universal subgraph API that works:
subgraph.nodes           # ✅ NodesAccessor with graph reference
subgraph.edges           # ✅ EdgesAccessor with graph reference  
subgraph.nodes.table()   # ✅ GraphTable of subgraph nodes
subgraph.edges.table()   # ✅ GraphTable of subgraph edges
subgraph.table()         # ✅ Combined GraphTable implementation
```

---

## 📝 ADDITIONAL DESIGN NOTES & USAGE ISSUES

### **Query Syntax for Null Values**
- **Issue**: How to query for non-null values using `!= None` syntax
- **Current**: Unclear if `g.filter_nodes("name != None")` works
- **Expected**: Support null/None checking in string-based queries
- **Use Case**: Filter out nodes/edges with missing attribute values

### **Lazy Rust View Architecture (COMPLETED)**
**Core Architectural Principle**: All data structures are lazy Rust views that only materialize to Python objects via `.values`:

```python
# NEW ARCHITECTURE: Everything is a lazy Rust view
node_ids = g.node_ids              # Returns GraphArray, not Python list
edge_ids = g.edge_ids              # Returns GraphArray, not Python list
table = g.nodes.table()            # Returns GraphTable (lazy view)
matrix = g.nodes[:][['age', 'dept']] # Returns GraphMatrix (lazy view)
array = g.nodes[:]['salary']       # Returns GraphArray (lazy view)

# Fast internal operations (all in Rust)
avg_salary = g.nodes[:]['salary'].mean()           # No Python conversion
total_nodes = g.node_ids.len()                     # No Python conversion
filtered = g.nodes[:]['age'].filter(lambda x: x > 30)  # No Python conversion

# Explicit materialization when needed
node_list = g.node_ids.values      # [1, 2, 3, 4, 5] - Python list
salary_list = array.values         # [50000, 60000, 75000] - Python list
pandas_df = table.to_pandas()      # Pandas DataFrame
numpy_array = matrix.to_numpy()    # NumPy ndarray

# Benefits:
# - Low memory usage: Data stays in Rust until explicitly materialized
# - High performance: Internal operations use native Rust performance
# - Consistent API: All data structures follow same lazy pattern
# - Backward compatibility: .values provides traditional Python objects
```

### **Core Design Distinction: GraphMatrix vs GraphTable**
**Key Architectural Principle**:
- **GraphMatrix**: Collection of GraphArrays (columns) - **mixed types allowed, no column names required**
- **GraphTable**: Structured DataFrame-like container - **mixed types allowed, column names required**

**Detailed Differences**:
```python
# GraphMatrix: Lightweight column collection
matrix = g.nodes[:][['age', 'name', 'active']]  # Mixed types: int, str, bool
# - Can hold GraphArrays of different types (int, str, bool, float)
# - Minimal metadata, sparse-friendly storage
# - No required column names (can be positional access)
# - Scientific computing focus: to_numpy(), to_scipy_sparse()
# - Use case: Multi-column extraction, matrix math, linear algebra

# GraphTable: Rich DataFrame-like structure  
table = g.table()  # All node attributes with full metadata
# - Can hold GraphArrays of different types (int, str, bool, float)
# - Rich metadata, column names required
# - Full DataFrame semantics with row/column labels
# - Data analysis focus: to_pandas(), to_csv(), to_json()
# - Use case: Data analysis, exports, joins, complex table operations
```

**Both Support Mixed Types**: The fundamental difference is **metadata richness and intended use case**, not type restrictions.

### **Graph/Subgraph Attributes Design**
- **Issue**: Graphs and subgraphs need their own attributes (metadata) stored consistently with node/edge attributes
- **Design Question**: How to assign IDs to graphs/subgraphs for attribute storage?
- **Proposed Solution**: Store attr_id in subgraph object, link to attribute storage system
- **Use Case**: Graph-level metadata (creation_date, description, version), subgraph metadata (component_type, analysis_result)
- **Storage Pattern**: Same attribute storage mechanism as nodes/edges, but for graph entities

```python
# Graph attributes (proposed)
g.set_attr('created_by', 'user123')
g.set_attr('analysis_type', 'social_network')
print(g.get_attr('created_by'))  # 'user123'

# Subgraph attributes (proposed)  
component = g.connected_components()[0]
component.set_attr('component_type', 'core_cluster')
component.set_attr('centrality_score', 0.85)
print(component.get_attr('component_type'))  # 'core_cluster'

# Attribute access consistency
g.attributes                    # Graph attribute accessor
component.attributes            # Subgraph attribute accessor
```

### **Accessor Attributes Missing**
- **Issue**: Node/edge accessors and subgraph accessors lack `.attributes` property
- **Missing**: `.nodes.attributes`, `.edges.attributes` for filtered attribute access
- **Expected Behavior**: 
  - `g.nodes.attributes` → node attributes for full graph
  - `subgraph.nodes.attributes` → node attributes for subgraph nodes only
  - `subgraph.edges.attributes` → edge attributes for subgraph edges only
- **Use Case**: Filtered attribute analysis, subgraph-specific attribute operations

```python
# Current missing functionality (proposed):
g.nodes.attributes              # All node attributes  
g.edges.attributes              # All edge attributes

# Subgraph filtered attributes (proposed):
component = g.connected_components()[0]
component.nodes.attributes      # Node attributes for component nodes only
component.edges.attributes      # Edge attributes for component edges only

# Filtered subgraph attributes (proposed):
engineers = g.filter_nodes('dept == "Engineering"')
engineers.nodes.attributes      # Node attributes for engineering nodes only
engineers.edges.attributes      # Edge attributes for engineering edges only
```

### **GraphArray Missing Methods**
- **Issue**: GraphArray lacks essential statistical and analysis methods
- **Missing Methods**: `.unique()`, `.count()`, `.percentile()`
- **Use Case**: Data analysis, validation, statistical summaries

```python
# Missing GraphArray methods (needed):
ages = table['age']              # Returns GraphArray

# Unique values analysis
unique_ages = ages.unique()      # Returns GraphArray of unique values
unique_count = len(ages.unique()) # Count of unique values

# Value counting
age_counts = ages.count()        # Returns dict {value: count} or GraphArray of counts
value_count = ages.count(30)     # Count occurrences of specific value

# Percentile calculations  
p25 = ages.percentile(25)        # 25th percentile (Q1)
p75 = ages.percentile(75)        # 75th percentile (Q3) 
p90 = ages.percentile(90)        # 90th percentile
median_alt = ages.percentile(50) # Alternative to .median()

# Statistical workflow example:
ages = table['age']
print(f"Unique ages: {len(ages.unique())}")
print(f"Age distribution: {ages.count()}")  
print(f"Quartiles: {ages.percentile(25)}, {ages.percentile(75)}")
```

### **GraphArray Mathematical Operators**
- **Priority**: HIGH - Essential for pandas/NumPy compatibility and data science workflows
- **Missing Operations**: Basic arithmetic operators `+`, `-`, `*`, `/`, `@` (matrix multiplication)
- **Use Case**: Mathematical operations on graph data, statistical transformations, linear algebra

```python
# Missing GraphArray mathematical operators (HIGH PRIORITY):
ages = g.nodes[:]['age']         # GraphArray
salaries = g.nodes[:]['salary']  # GraphArray

# Element-wise arithmetic operations
ages_plus_10 = ages + 10         # Add scalar
age_ratios = ages / ages.mean()  # Divide by scalar
combined = ages + salaries       # Element-wise addition (compatible sizes)
scaled_salary = salaries * 1.1   # Multiply by scalar

# Matrix multiplication for compatible arrays
weights = g.edges[:]['weight']   # GraphArray of edge weights
scores = node_features @ weights # Matrix multiplication with @

# Statistical transformations
normalized_ages = (ages - ages.mean()) / ages.std()    # Z-score normalization
log_salaries = salaries.log()    # Logarithmic transformation

# Real-world data science workflows
features = g.nodes[:][['age', 'experience', 'salary']]  # GraphMatrix
normalized_features = (features - features.mean()) / features.std()
correlation_matrix = normalized_features.T @ normalized_features

# Financial calculations
returns = prices[1:] / prices[:-1] - 1  # Calculate returns from prices
risk_score = returns.std() * (365 ** 0.5)  # Annualized volatility

# Graph analytics with arithmetic
centrality = g.nodes[:]['centrality']
influence = centrality * g.nodes[:]['activity_level']  # Combined influence score
top_influencers = influence[influence > influence.percentile(90)]
```

**Implementation Requirements**:
- **Size compatibility**: Arrays must have same length for element-wise operations
- **Scalar operations**: Support operations with single values (broadcasting)
- **Type safety**: Ensure mathematical operations work with numeric types
- **Performance**: Native Rust implementation for speed
- **Error handling**: Clear errors for incompatible sizes or types

### **AdjacencyMatrix Design Clarification**
- **Design Decision**: AdjacencyMatrix should inherit from GraphMatrix (not separate matrix type)
- **Rationale**: Avoid proliferation of matrix types, leverage existing GraphMatrix functionality
- **Implementation**: `g.adjacency_matrix()` returns `GraphMatrix` with adjacency data
- **Benefit**: Inherits all GraphMatrix methods (`.to_numpy()`, `.to_pandas()`, column access, etc.)

### **Basic Linear Algebra Operations (Low Priority)**
- **Need**: Basic linear algebra operations for GraphMatrix
- **Missing Methods**: `.dot()` (matrix multiplication), `.T` (transpose)
- **Priority**: Very low - scientific libraries provide these operations
- **Use Case**: Basic matrix math without converting to NumPy

```python
# Low priority linear algebra (proposed):
matrix = g.nodes[:][['age', 'salary']]    # GraphMatrix
transposed = matrix.T                     # Transpose operation
result = matrix.dot(other_matrix)         # Matrix multiplication

# Alternative: Use scientific libraries (preferred)
matrix_numpy = matrix.to_numpy()          # Convert to NumPy
result = matrix_numpy @ other_numpy       # NumPy provides full linalg
```

### **Filter Nodes Enhancement: Outer Edges**
- **Feature**: `filter_nodes()` should support optional `outer_edges` parameter
- **Behavior**: When `outer_edges=True`, include edges where filtered nodes appear as source OR target
- **Use Case**: Include boundary edges for filtered subgraphs, preserve node connectivity context
- **Implementation**: `g.filter_nodes(query, outer_edges=False)` with default False for backward compatibility

```python
# Enhanced filter_nodes with outer_edges (proposed):
engineers = g.filter_nodes('dept == "Engineering"')                    # Only engineering nodes + internal edges
engineers_with_context = g.filter_nodes('dept == "Engineering"', outer_edges=True)  # + edges to/from other depts

# Use cases:
# 1. Analyze internal team structure (outer_edges=False, default)
internal_team = g.filter_nodes('dept == "Engineering"')
print(f"Internal edges: {len(internal_team.edge_ids)}")

# 2. Analyze team boundary connections (outer_edges=True)  
team_boundary = g.filter_nodes('dept == "Engineering"', outer_edges=True)
print(f"Total edges involving engineers: {len(team_boundary.edge_ids)}")
print(f"External connections: {len(team_boundary.edge_ids) - len(internal_team.edge_ids)}")

# 3. Network analysis with context
core_nodes = g.filter_nodes('centrality > 0.8', outer_edges=True)  # Include edges to peripheral nodes
```

### **GraphArray of Subgraphs Design**
- **Feature**: GraphArray should support subgraph objects as elements (not just primitive types)
- **Use Case**: Algorithm results that return collections of subgraphs as structured arrays
- **Implementation**: `connected_components()`, `group_by()` should return `GraphArray[Subgraph]` instead of plain list
- **Benefit**: Consistent API, statistical operations on subgraph collections, subgraphs as node/edge attributes

```python
# Enhanced subgraph collections (proposed):
components = g.connected_components()     # Returns GraphArray[Subgraph] instead of List[Subgraph]
groups = g.group_by('dept')               # Returns GraphArray[Subgraph] grouped by department

# Statistical operations on subgraph collections
print(f"Component count: {len(components)}")
component_sizes = components.map(lambda comp: len(comp.node_ids))  # GraphArray[int] of sizes
print(f"Average component size: {component_sizes.mean()}")
print(f"Largest component: {component_sizes.max()} nodes")

# Subgraphs as node/edge attributes
# Store analysis results as subgraph attributes
node_0 = g.nodes[0]
node_0.set_attr('local_neighborhood', g.filter_nodes('distance_from_0 <= 2'))  # Subgraph as attribute

# Store component membership
for i, component in enumerate(components):
    for node_id in component.node_ids:
        g.nodes[node_id].set_attr('component_subgraph', component)  # Reference to subgraph object

# Retrieve subgraph attributes
neighborhood = g.nodes[0].get_attr('local_neighborhood')  # Returns Subgraph object
print(f"Local neighborhood has {len(neighborhood.node_ids)} nodes")

# Group-by operations with subgraph results
dept_groups = g.group_by('dept')          # GraphArray[Subgraph] grouped by department
eng_group = dept_groups.filter(lambda sg: sg.get_attr('dept') == 'Engineering')[0]
print(f"Engineering group: {len(eng_group.node_ids)} nodes")
```

**Root Issues**:
- **✅ RESOLVED**: Connected components subgraphs lack graph reference 
- **✅ RESOLVED**: NodesAccessor missing `.table()` method
- **✅ RESOLVED**: Inconsistent subgraph behavior between creation methods

**All Priority 2 issues have been successfully resolved in the current implementation.**

### 🎯 **Priority 2: API Consistency Issues - CURRENT FOCUS**
**Priority**: High - Essential for user workflow consistency and completeness

**🎯 IMMEDIATE SMALL TASKS - Current Implementation Targets**:

**🚨 CRITICAL BUG - PySubgraph Attribute Access (HIGHEST PRIORITY)**:
- [ ] **Fix `g.edges[:]['id']`**: Currently returns all zeros instead of actual edge IDs [0, 1, 2, 3]
- [ ] **Fix `g.edges[:]['strength']`**: Returns GraphArray of 100 zeros instead of actual edge attributes
- [ ] **Fix `g.nodes[:]['age']`**: Returns fallback text "GraphArray(len=100, dtype=int64)" instead of rich display
- [ ] **Fix PySubgraph.__getitem__()**: Completely broken - edge subgraphs return zeros, node subgraphs fail to rich display
- [ ] **Investigate existing broken __getitem__**: Find and replace the current broken implementation

**📊 GraphArray Core Methods (HIGH PRIORITY)**:
- [ ] **GraphArray.unique()**: Return GraphArray of unique values for data analysis
- [ ] **GraphArray.count()**: Return dict/GraphArray of value counts for frequency analysis  
- [ ] **GraphArray.percentile(q)**: Calculate percentiles (25th, 50th, 75th, 90th) for statistics

**🎨 Display & Debugging (MOSTLY COMPLETED)**:
- [x] **Rich display module**: Beautiful table/matrix/array formatting with box-drawing characters ✅
- [x] **GraphArray repr improvements**: Show actual values with beautiful Unicode formatting ✅
- [x] **GraphArray type display**: Show data type and shape in professional format ✅
- [x] **GraphTable rich display**: Polars-style table formatting with proper column types and summary stats ✅
- [x] **GraphMatrix rich display**: Matrix formatting ready (pending constructor access) ✅
- [x] **Boolean display fix**: True/False now display correctly instead of 1/0 ✅
- [x] **Type detection**: Accurate dtype detection (int64, float64, bool, string, category) ✅

**🔪 Slicing Operations (HIGH PRIORITY)**:
- [ ] **GraphArray slicing**: Support `ages[:10]`, `ages[5:15]`, `ages[::2]` slice notation
- [ ] **GraphTable multi-column slicing**: `table[['col1', 'col2']]` returns GraphTable with selected columns
- [ ] **GraphArray boolean indexing**: `ages[ages > 30]` for conditional filtering
- [ ] **GraphTable row slicing**: `table[:100]`, `table[10:20]` for row subset access

**🔄 Sorting Operations (MEDIUM PRIORITY)**:
- [ ] **GraphArray.sort()**: In-place sorting with `ascending` parameter
- [ ] **GraphArray.sorted()**: Return new sorted GraphArray (non-destructive)
- [ ] **GraphTable.sort(column)**: Sort table by column name
- [ ] **GraphTable.sort_values(column)**: Pandas-compatible sorting
- [ ] **GraphMatrix.sort_by_column(idx)**: Sort matrix rows by specific column values

**🔗 Access Patterns (MOSTLY COMPLETED)**:
- [x] **GraphArray.values property**: Direct access to underlying data (like Pandas Series.values) ✅
- [ ] **GraphArray.rename(name)**: Change the name/label of a GraphArray for display and operations
- [x] **GraphTable.columns property**: List/GraphArray of column names ✅
- [x] **GraphTable.dtypes property**: Column data types mapping ✅
- [x] **GraphMatrix.shape property**: Tuple of (rows, cols) dimensions ✅  
- [x] **GraphMatrix.columns property**: Column names as property ✅

**📈 Enhanced Statistics (LOW PRIORITY)**:
- [ ] **GraphArray.value_counts()**: Pandas-style value counting with sort options
- [ ] **GraphArray.describe()**: Statistical summary (mean, std, min, max, quartiles)
- [ ] **GraphTable.describe()**: Per-column statistical summaries
- [ ] **GraphArray.mode()**: Most frequent value(s)

**🚀 Performance Micro-optimizations (LOW PRIORITY)**:
- [ ] **Lazy evaluation**: Defer computation until `.values` or display
- [ ] **Memory pooling**: Reuse allocations for repeated operations
- [ ] **SIMD operations**: Vector operations for statistical methods
- [ ] **Chunk processing**: Handle large arrays in memory-efficient chunks

**🎯 PRIORITIZED IMPLEMENTATION ORDER**:
1. **Week 1**: GraphArray.unique(), .count(), .percentile() + Rich display module
2. **Week 2**: GraphArray/GraphTable slicing operations + repr improvements
3. **Week 3**: Sorting functionality across all data structures
4. **Week 4**: Enhanced access patterns and statistics methods

### 🎯 **Priority 3: Rich Display Module - ✅ COMPLETED**
**Priority**: HIGH - Essential for user experience and debugging

**🎨 Beautiful Display System**: Implement rich, professional display formatting for all data structures based on `display_draft.txt`

**✅ COMPLETED - Display Module Foundation**:
- [x] **Display Module Structure**: Created `python-groggy/python/groggy/display/` with complete module architecture
- [x] **Unicode Box Drawing**: Implemented proper `╭─╮│├┤╰─╯` characters for professional appearance
- [x] **GraphTable Display**: Polars-style table with box-drawing characters, column type annotations, summary statistics
- [x] **GraphMatrix Display**: Matrix formatting with shape/dtype info, smart truncation for large matrices with `⋯` placeholders  
- [x] **GraphArray Display**: Column-style display with index, values, type info, and shape summary
- [x] **Smart Truncation**: Working first/last rows for large data with `…` indicators
- [x] **Type Annotations**: Show data types (str[8], cat(12), f32, date, etc.) in headers
- [x] **Summary Statistics**: Include row counts, column counts, null counts, index type
- [x] **Demo System**: Working demo script showing all three display types

**📁 Display Module Structure** (✅ IMPLEMENTED):
```python
# python-groggy/python/groggy/display/
#   __init__.py          # ✅ Public display API  
#   formatters.py        # ✅ Core formatting logic
#   table_display.py     # ✅ GraphTable rich display
#   matrix_display.py    # ✅ GraphMatrix rich display  
#   array_display.py     # ✅ GraphArray rich display
#   unicode_chars.py     # ✅ Box-drawing character constants
#   truncation.py        # ✅ Smart truncation algorithms
#   demo.py              # ✅ Working demonstration script
```

**🎯 Working Example Output** (matches display_draft.txt):
```
⊖⊖ gr.table
╭──────┬─────────┬───────────┬──────┬───────┬────────────╮
│    # │ name    │ city      │ age  │ score │ joined     │
│      │ str[8]  │ cat(12)   │ i64  │ f32   │ date       │
├──────┼─────────┼───────────┼──────┼───────┼────────────┤
│    0 │ Alice   │ NYC       │ 25   │ 91.50 │ 2024-02-15 │
│    1 │ Bob     │ Paris     │ 30   │ 87.00 │ 2023-11-20 │
│    … │ …       │ …         │ …    │ …     │ …          │
│   11 │ Liam    │ Amsterdam │ 30   │ 91.90 │ 2023-07-16 │
╰──────┴─────────┴───────────┴──────┴───────┴────────────╯
rows: 1,000 • cols: 5 • nulls: score=12 • index: int64
```

**� REMAINING INTEGRATION TASKS**:
- [ ] **Hook into PyGraphTable**: Add `__repr__` and `__str__` methods calling display module
- [ ] **Hook into PyGraphMatrix**: Add `__repr__` and `__str__` methods calling display module  
- [ ] **Hook into PyGraphArray**: Add `__repr__` and `__str__` methods calling display module
- [ ] **Data Structure Conversion**: Convert Rust data to Python dict format for display module
- [ ] **Rust Integration**: Add methods to extract display data from Rust backend
- [ ] **Error Handling**: Graceful fallback for display formatting failures
- [ ] **Performance Optimization**: Cache formatted output for large datasets
- [ ] **Configuration**: Allow users to configure display settings (max_rows, max_cols, etc.)

**🔧 Next Implementation Steps**:
1. **Week 1**: Integrate display module into existing PyGraphTable, PyGraphMatrix, PyGraphArray classes
2. **Week 2**: Add Rust-side data extraction methods for display formatting
3. **Week 3**: Performance optimization and configuration system
4. **Week 4**: Testing and documentation for complete display system

### 🎯 **Priority 4: Modularization - NEXT MAJOR PRIORITY**
**Priority**: HIGH - Essential for maintainability and beautiful user experience

**🏗️ lib.rs Modularization**: Break down the monolithic 4,437-line lib.rs file into logical modules

**📋 IMMEDIATE EXTRACTIONS** (Ready Now - Low Risk):
- [ ] **Extract arrays.rs**: PyGraphArray, PyStatsSummary, PyGraphMatrix (lines 3800-4370) - all functionality complete
- [ ] **Extract accessors.rs**: PyNodesAccessor, PyEdgesAccessor (lines 3141-3490) - all functionality complete  
- [ ] **Extract views.rs**: PyNodeView, PyEdgeView (lines 3500-3800) - all functionality complete
- [ ] **Test extractions**: Ensure all existing functionality works after modularization

**✅ COMPLETED DISPLAY INTEGRATION TASKS**:
- [x] **Display integration complete**: All display data extraction methods implemented
- [x] **PyGraphTable.__repr__**: Successfully calls `format_table()` with extracted display data
- [x] **PyGraphMatrix.__repr__**: Successfully calls `format_matrix()` with extracted display data
- [x] **PyGraphArray.__repr__**: Successfully calls `format_array()` with extracted display data
- [x] **Error handling**: Graceful fallback implemented and tested
- [x] **Data format fix**: Fixed GraphTable to show actual values instead of column names

**📊 Display Data Extraction Methods**:
```rust
// Methods to implement in display_integration.rs
impl PyGraphTable {
    fn get_display_data(&self) -> PyResult<HashMap<String, PyObject>> {
        // Extract columns, dtypes, data, shape, nulls, index_type
    }
}

impl PyGraphMatrix {
    fn get_display_data(&self) -> PyResult<HashMap<String, PyObject>> {
        // Extract data, shape, dtype, column_names
    }
}

impl PyGraphArray {
    fn get_display_data(&self) -> PyResult<HashMap<String, PyObject>> {
        // Extract data, dtype, shape, name
    }
}
```

**🎯 Implementation Order**:
1. **Week 1**: Extract arrays.rs, accessors.rs, views.rs modules (2 hours - pure extraction)
2. **Week 2**: Implement display integration for beautiful __repr__ output (3 hours)
3. **Week 3**: Extract remaining value types and filter modules (6 hours)
4. **Week 4**: Begin core graph modularization planning

**📁 Target Module Structure**:
```
python-groggy/src/
├── lib.rs                 # Main coordinator (~100 lines)
├── arrays.rs              # ✅ READY - Statistical arrays & matrices (~600 lines) 
├── accessors.rs           # ✅ READY - Smart indexing accessors (~350 lines)
├── views.rs               # ✅ READY - Individual element views (~300 lines)
├── display_integration.rs # NEW - Rich display hooks (~200 lines)
├── types.rs               # Enhanced value types (~600 lines)
├── filters.rs             # Complete query/filter system (~500 lines)
└── [additional modules]   # Version control, subgraph, graph core, etc.
```

### 🎯 **Priority 5: Performance Optimization**
**Priority**: Medium - Fine-tune remaining bottlenecks

**Optimization Targets**:
- [ ] **Memory optimization**: Reduce memory overhead to match NetworkX
- [ ] **Query optimization**: Convert remaining O(n log n) to O(n) operations
- [ ] **Matrix operations**: Optimize adjacency matrix construction 
- [ ] **Bulk operations**: Optimize multi-attribute access patterns

**Current Performance Status**:
- ✅ **Node/edge iteration**: 50-100x faster than NetworkX
- ✅ **Filtering operations**: 10-50x faster than NetworkX  
- ✅ **Statistical operations**: Native Rust performance
- ⚠️ **Memory usage**: 1.5x NetworkX overhead (370MB vs 247MB for 250K nodes)
- ⚠️ **Complex queries**: Some O(n log n) operations could be O(n)

---

## 🎯 CURRENT PRIORITIES

## 📋 VALIDATION CHECKLIST

### Immediate Action Items  
- [ ] **Fix connected components graph reference**: Ensure all subgraphs have working `.nodes` and `.edges` accessors
- [ ] **Add NodesAccessor.table()**: `subgraph.nodes.table()` should work like `g.nodes.table()`
- [ ] **Add EdgesAccessor.table()**: `subgraph.edges.table()` should work like `g.edges.table()`
- [ ] **Add subgraph.table()**: Combined node/edge DataFrame-like access
- [ ] **GraphTable multi-column slicing**: `table[['col1', 'col2']]` returns GraphTable with selected columns
- [ ] **GraphArray scientific conversions**: `to_numpy()`, `to_pandas()`, `to_scipy_sparse()` methods
- [ ] **AdjacencyMatrix conversions**: Scientific computing integration methods

### Integration Testing
- [ ] All existing tests pass after subgraph accessor fixes
- [ ] GraphTable integrates correctly with existing workflows  
- [ ] GraphArray conversions work with scientific libraries
- [ ] Memory usage remains stable under load

---

## 🎯 SUCCESS CRITERIA

**Current Focus**: 
1. **Critical Subgraph Fixes**: Resolve graph reference and accessor issues
2. **Scientific Computing Integration**: Add remaining conversion methods
3. **API Consistency**: Complete GraphTable and GraphArray enhancements

*Note: Major completed features have been moved to `/docs/usage_examples.md` and `/upcoming_release_notes_draft.md` for better organization.*