# Next Steps - Current Priorities

## ✅ LATEST PROGRESS UPDATE (August 15, 2025)

### 🎯 **MAJOR FEATURE COMPLETIONS!**
**Recent Achievements**: Successfully completed GraphArray integration and API consistency fixes

**Latest Completed Features**:
- ✅ **GraphArray Rename**: PyArray → GraphArray completed for better API consistency
- ✅ **GraphTable Integration**: GraphArray integration with GraphTable column access complete
- ✅ **Subgraph API Fix**: node_ids and edge_ids properties added to PySubgraph
- ✅ **Connected Components Fix**: Edge collection now works correctly in connected components
- ✅ **Performance Benchmarks**: Updated benchmarks to use optimized GraphArray API
- ✅ **API Testing**: Comprehensive test suite for GraphArray statistical operations

**Key Integrations**:
```python
# NEW: GraphTable columns return GraphArray objects with native statistics
table = g.table()
ages = table['age']              # Returns GraphArray (not plain list)
print(ages.mean(), ages.std())   # Native Rust statistical operations

# FIXED: Subgraph properties now work consistently
subgraph = g.connected_components()[0]
print(subgraph.node_ids)         # ✅ Works - shows component nodes
print(subgraph.edge_ids)         # ✅ Works - shows component edges
```

**Impact**: GraphArray provides seamless transition from graph data to statistical analysis, matching pandas DataFrame column workflow with native performance.

**⚠️ Known Issues to Address**:
- **Multi-column selection**: `g.nodes[:][['id','index']]` returns list of GraphArrays instead of GraphTable
- **GraphTable column subset**: `table[['col1', 'col2']]` not yet implemented (TypeError)
- **GraphArray repr**: Shows only length, needs to display actual values for debugging
- **GraphArray.values**: Missing pandas-like `.values` property for raw data access
- **Adjacency matrix methods**: Rust implementation exists but Python bindings not exposed
- **node_ids/edge_ids as GraphArray**: Should return GraphArray instead of plain list for statistical capabilities
- **GraphArray scientific conversions**: Missing `to_numpy()`, `to_scipy_sparse()`, `to_pandas()` methods
- **Consistent GraphArray returns**: All list outputs should be GraphArray for efficiency and analytics

---

## 🎯 CURRENT PRIORITIES

### 🎯 **Priority 1: Adjacency Matrix Support**
**Priority**: High - Essential graph analytics feature

**Goal**: Provide efficient adjacency matrix generation for graphs and subgraphs

**API Design**:
```python
# Full graph adjacency matrix
adj_matrix = g.adjacency()                    # Returns gr.array (sparse/dense matrix)

# Subgraph adjacency matrix with index mapping  
subgraph = g.filter_nodes("dept == 'Engineering'")
adj_matrix = subgraph.adjacency(map_index=True)   # Default: True - compact gr.array
index_mapping = subgraph.index_mapping()           # Maps subgraph indices to original node IDs

# Option: Full-size matrix (rare use case)
adj_matrix_full = subgraph.adjacency(map_index=False)  # Full graph size gr.array, sparse for subgraph

# Laplacian matrix support
laplacian = g.laplacian(epsilon=-0.5, k=1)            # Graph Laplacian as gr.array
laplacian_sub = subgraph.laplacian(epsilon=-0.5, k=1) # Subgraph Laplacian as gr.array

# Fast conversion to scientific computing libraries
adj_numpy = adj_matrix.to_numpy()              # Convert to NumPy array
adj_scipy = adj_matrix.to_scipy_sparse()       # Convert to SciPy sparse matrix  
adj_pandas = adj_matrix.to_pandas()            # Convert to Pandas DataFrame

# GraphArray integration - node/edge IDs as statistical arrays
node_ids = g.node_ids                           # Returns GraphArray([0, 1, 2, 3, 4])
edge_ids = subgraph.edge_ids                    # Returns GraphArray([5, 7, 12])

# Statistical analysis on IDs
print(f"Node ID range: {node_ids.min()}-{node_ids.max()}")
print(f"Average node ID: {node_ids.mean()}")   # Useful for sparse graphs
print(f"Subgraph connectivity: {len(edge_ids)} edges for {len(node_ids)} nodes")

# Scientific computing with ID arrays
nodes_numpy = node_ids.to_numpy()              # NumPy array of node IDs
edges_pandas = edge_ids.to_pandas()            # Pandas Series of edge IDs

# Usage examples
import numpy as np
eigenvals = np.linalg.eigvals(adj_matrix.to_numpy())      # Graph spectral analysis
degrees = adj_matrix.to_numpy().sum(axis=1)               # Node degrees from matrix
laplacian_eigenvals = np.linalg.eigvals(laplacian.to_numpy())  # Laplacian spectrum
```

**Key Features**:
- **gr.array Return Type**: Matrices returned as groggy array objects with native performance
- **Fast Conversions**: Easy `to_numpy()`, `to_scipy_sparse()`, `to_pandas()` methods
- **Efficient Storage**: Sparse matrices by default for large graphs
- **Index Mapping**: Subgraph matrices use compact indexing with mapping to original IDs
- **Flexible Options**: Choice between compact (default) or full-size matrices
- **Integration**: Works seamlessly with NumPy/SciPy via conversion methods
- **Performance**: Native Rust implementation for fast matrix construction
- **Rust Implementation**: All adjacency matrix logic and rendering implemented in Rust core
- **Laplacian Support**: Graph Laplacian matrix with configurable epsilon and k parameters

**Implementation Strategy**:
```python
# PyGraph method
def adjacency(self, sparse=True, dtype=np.float64):
    """Generate adjacency matrix for full graph"""
    return self._rust_graph.adjacency_matrix(sparse, dtype)

def laplacian(self, epsilon=-0.5, k=1, sparse=True, dtype=np.float64):
    """Generate Laplacian matrix for full graph"""
    return self._rust_graph.laplacian_matrix(epsilon, k, sparse, dtype)

# PySubgraph method  
def adjacency(self, map_index=True, sparse=True, dtype=np.float64):
    """Generate adjacency matrix for subgraph"""
    if map_index:
        # Compact matrix using subgraph node indices
        return self._build_compact_adjacency_matrix(sparse, dtype)
    else:
        # Full-size matrix with zeros for non-subgraph nodes
        return self._build_full_adjacency_matrix(sparse, dtype)

def laplacian(self, epsilon=-0.5, k=1, map_index=True, sparse=True, dtype=np.float64):
    """Generate Laplacian matrix for subgraph"""
    if map_index:
        return self._build_compact_laplacian_matrix(epsilon, k, sparse, dtype)
    else:
        return self._build_full_laplacian_matrix(epsilon, k, sparse, dtype)

def index_mapping(self):
    """Return mapping from compact indices to original node IDs"""
    return {i: node_id for i, node_id in enumerate(self.node_ids)}
```

**Use Cases**:
- **Spectral Analysis**: Eigenvalue/eigenvector computations for graph properties
- **Community Detection**: Matrix-based clustering algorithms  
- **Path Analysis**: Powers of adjacency matrix for multi-step connectivity
- **Graph Comparison**: Matrix norms and distances between graph structures
- **Machine Learning**: Graph neural networks, embedding algorithms
- **Laplacian Analysis**: Graph signal processing, diffusion processes, spectral clustering

**Implementation Tasks**:
- [ ] **Add adjacency() to PyGraph**: Expose existing Rust implementation in Python bindings  
- [ ] **Add adjacency() to PySubgraph**: Python bindings for compact and full-size options
- [ ] **Add laplacian() to PyGraph**: Expose Rust Laplacian matrix methods in Python
- [ ] **Add laplacian() to PySubgraph**: Python bindings for subgraph Laplacian with index mapping
- [ ] **Add index_mapping() to PySubgraph**: Mapping from compact to original indices
- [ ] **Implement GraphArray conversion methods**: `to_numpy()`, `to_scipy_sparse()`, `to_pandas()`
- [ ] **Convert node_ids/edge_ids to GraphArray**: Enable statistical analysis on ID collections
- [ ] **Add weighted matrix support**: Include edge weights in matrix values
- [ ] **Performance optimization**: Native Rust matrix construction and storage
- [ ] **Integration testing**: Verify matrix operations work with conversion methods
- [ ] **Documentation**: Examples for spectral analysis and ML applications

**Note**: Rust implementation may already exist - focus on Python binding exposure

**Success Metrics**:
- Graph adjacency matrices generate correctly for any graph size
- Subgraph compact matrices use minimal memory with proper index mapping
- Matrix operations integrate seamlessly with scientific Python ecosystem
- Performance competitive with NetworkX for large graphs

### 🎯 **Priority 2: Multi-Column Selection and GraphArray Integration Issues**
**Priority**: High - Critical API consistency bugs and GraphArray enhancement

**Issues Identified**:

**1. Multi-Column Selection Returns GraphArray List Instead of GraphTable**
```python
# Current problematic behavior:
result = g.nodes[:][['id','index']]  
print(result)  # [GraphArray(len=100), GraphArray(len=100)] - Wrong!

# Should return: GraphTable with 2 columns like pandas DataFrame
# Expected: GraphTable with 'id' and 'index' columns, not separate GraphArrays
```

**2. GraphTable Multi-Column Access Not Implemented**
```python
# Current: This fails with TypeError
table = g.table()
subset = table[['id', 'index']]  
# TypeError: Key must be string (column), int (row), or slice

# Should work: Return GraphTable with only selected columns
# Expected: subset.to_pandas() shows DataFrame with 'id', 'index' columns only
```

**3. GraphArray String Representation Needs Improvement**
```python
# Current: Uninformative representation
ages = GraphArray([25, 30, 35, 40, 45])
print(ages)  # GraphArray(len=5) - Not helpful!

# Should show: GraphArray(len=5, values=[25, 30, 35, 40, 45])
# For large arrays: GraphArray(len=1000, values=[25, 30, 35, ..., 98, 99, 100])
```

**4. Missing .values Property on GraphArray**
```python
# Should work like pandas Series.values or .to_list()
ages = table['age']              # Returns GraphArray
raw_data = ages.values           # Should return plain Python list
# OR ages.to_list() already works, but .values is more intuitive
```

**5. node_ids/edge_ids Should Return GraphArray**
```python
# Current: Plain Python lists
print(g.node_ids)                # [0, 1, 2, 3, 4] - plain list
print(subgraph.node_ids)         # [1, 3, 5] - plain list

# Enhanced: GraphArray with statistical capabilities
node_ids = g.node_ids           # Returns GraphArray([0, 1, 2, 3, 4])
print(node_ids.min(), node_ids.max())  # ID range analysis
print(node_ids.mean())          # Average node ID (useful for sparse graphs)
print(len(node_ids.unique()))   # Count unique IDs (validation)

# Subgraph node analysis
subgraph_nodes = subgraph.node_ids  # GraphArray([1, 3, 5])  
print(f"Subgraph has {len(subgraph_nodes)} nodes")
print(f"Node ID range: {subgraph_nodes.min()}-{subgraph_nodes.max()}")
```

**6. Missing Scientific Computing Conversions**
```python
# Essential for integration with NumPy/SciPy/Pandas ecosystem
node_ids = g.node_ids           # GraphArray
ages = table['age']             # GraphArray

# Scientific conversions (MISSING)
ids_numpy = node_ids.to_numpy()        # Convert to numpy array
ages_pandas = ages.to_pandas()          # Convert to pandas Series
sparse_matrix = adjacency.to_scipy_sparse()  # For sparse matrices

# Integration examples
import numpy as np
import pandas as pd
age_stats = np.histogram(ages.to_numpy(), bins=10)  # NumPy integration
age_series = ages.to_pandas()                       # Pandas integration
correlation = np.corrcoef(ages.to_numpy(), salaries.to_numpy())  # Multi-array analysis
```

**7. Consistent GraphArray Returns for Efficiency**
```python
# All list outputs should be GraphArray for consistency and performance
attribute_values = g.attributes.nodes['age']   # Should be GraphArray
filtered_ids = g.filter_nodes('dept == "Engineering"').node_ids  # Should be GraphArray
edge_weights = g.attributes.edges['weight']    # Should be GraphArray
component_sizes = [len(comp.node_ids) for comp in g.connected_components()]  # Each comp.node_ids should be GraphArray
```

**Implementation Tasks**:
- [ ] **Add multi-column support to GraphTable.__getitem__**: Handle `table[['col1', 'col2']]` 
- [ ] **Fix subgraph multi-column selection**: Should return GraphTable, not list of GraphArrays
- [ ] **Implement GraphArray.__repr__**: Show values with truncation for large arrays
- [ ] **Add GraphArray.values property**: Return plain Python list like .to_list()
- [ ] **Convert node_ids/edge_ids to GraphArray**: Return GraphArray instead of plain lists
- [ ] **Add scientific conversions to GraphArray**: `to_numpy()`, `to_scipy_sparse()`, `to_pandas()` methods
- [ ] **Systematically convert list returns**: Audit all list-returning methods to use GraphArray
- [ ] **Create ColumnSubsetView class**: Handle column-subset GraphTable operations
- [ ] **Update multi-column slicing logic**: Ensure consistent GraphTable return across all contexts

**Success Metrics**:
- `g.nodes[:][['id','index']]` returns GraphTable with 2 columns
- `table[['col1', 'col2']]` returns GraphTable with selected columns  
- GraphArray repr shows actual data with truncation
- `array.values` works as alias for `array.to_list()`
- `g.node_ids` and `subgraph.node_ids` return GraphArray with statistical methods
- `node_ids.to_numpy()`, `ages.to_pandas()` work for scientific computing integration
- All list outputs consistently return GraphArray for enhanced analytics capabilities

### 🎯 **Priority 3: Performance Optimization**
**Priority**: Medium - Fine-tune remaining O(n log n) operations

**Remaining Optimization Opportunities**:
- **Single Attribute filtering**: 84→109ns (could target ~85ns constant)
- **Complex AND queries**: 92→134ns (could target ~95ns constant)  
- **Memory usage**: Reduce 1.5x overhead vs NetworkX (370MB vs 247MB for 250K nodes)

---

## 📋 VALIDATION CHECKLIST

### Feature Validation  
- [ ] **Multi-column selection**: `g.nodes[:][['id','index']]` returns GraphTable (not GraphArray list)
- [ ] **GraphTable column subset**: `table[['col1', 'col2']]` returns GraphTable with selected columns
- [ ] **GraphArray repr**: Shows actual values with truncation: `GraphArray(len=5, values=[1,2,3,4,5])`
- [ ] **GraphArray.values property**: Works as alias for `.to_list()` for pandas compatibility
- [ ] **node_ids/edge_ids as GraphArray**: Return GraphArray with statistical methods
- [ ] **GraphArray scientific conversions**: `to_numpy()`, `to_pandas()`, `to_scipy_sparse()` methods
- [ ] **Adjacency matrix methods**: Python bindings exposed for existing Rust implementation
- [ ] **Unified View API**: Single entity views with `dict()` and `set()` methods

### Integration Testing
- [ ] All existing tests pass after GraphArray enhancements
- [ ] GraphTable integrates correctly with existing workflows  
- [ ] GraphArray conversions work with scientific libraries
- [ ] Memory usage remains stable under load

---

## 🎯 SUCCESS CRITERIA

### 🎯 **CURRENT FOCUS**  
1. **GraphArray Enhancement**: Complete scientific computing integration with conversion methods
2. **API Consistency**: Fix multi-column selection and GraphTable column subset issues
3. **Matrix Operations**: Expose adjacency matrix Python bindings for graph analytics
4. **Unified Experience**: All list outputs return GraphArray for consistent analytics capabilities

**Current Status**: 🚀 **Major foundations complete!** GraphArray statistical analytics, subgraph API consistency, and performance optimization achieved. Focus now on scientific computing integration and remaining API consistency issues.