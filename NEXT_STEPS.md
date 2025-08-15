# Next Steps - Current Priorities

## 🎯 CURRENT STATUS (August 15, 2025)

**Major Milestones Achieved**: GraphArray integration, Adjacency matrices, Multi-column GraphMatrix support

**✅ COMPLETED MAJOR FEATURES:**
- **Lazy Rust View Architecture**: All data structures (GraphArray, GraphMatrix, GraphTable) implemented as lazy views that only materialize via `.values`
- **node_ids/edge_ids Return GraphArray**: Breaking architectural change - `node_ids` and `edge_ids` now return `GraphArray` directly instead of Python lists (use `.values` for lists)  
- **Connected Components Fixed**: All subgraphs have working `.nodes`, `.edges` accessors and include proper induced edges
- **Enhanced GraphMatrix**: Multi-index access `matrix[row,col]`, row access `matrix[row]`, column access `matrix['col']`, and `.is_square()` method
- **GraphMatrix Positional Access**: Full support for `matrix[0, 1]` (single cell), `matrix[0]` (row as dict/list), `matrix['age']` (column as GraphArray)
- **Scientific Computing Integration**: GraphArray and GraphMatrix have `.to_numpy()`, `.to_pandas()`, `.to_scipy_sparse()` methods
- **Multi-Column GraphMatrix**: `g.nodes[:][['age', 'dept']]` returns structured `GraphMatrix`
- **Statistical GraphArray**: Full statistical operations (`.mean()`, `.min()`, `.max()`, `.sum()`) on all array types

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

### 🎯 **Priority 1: ✅ COMPLETED - Architecture Unification** 
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

### 🎯 **Priority 2: Critical Subgraph Accessor Issues**
**Implementation Priority**: Critical - Core subgraph functionality is broken, affects all graph analysis workflows

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
- **Connected components subgraphs lack graph reference**: `components[0].nodes` → RuntimeError
- **NodesAccessor missing `.table()` method**: `subgraph.nodes.table()` → AttributeError
- **Inconsistent subgraph behavior**: Different creation methods have different capabilities

**Required Fixes**:
```python
# ALL of these should work identically:
subgraph1 = g.filter_nodes('component_id == 0')      # ✅ Has graph reference
subgraph2 = g.connected_components()[0]               # ❌ Missing graph reference  
subgraph3 = g.filter_edges('weight > 0.5')           # Status unknown

# Universal subgraph API that should work:
subgraph.nodes           # NodesAccessor with graph reference
subgraph.edges           # EdgesAccessor with graph reference  
subgraph.nodes.table()   # GraphTable of subgraph nodes
subgraph.edges.table()   # GraphTable of subgraph edges
subgraph.table()         # Combined GraphTable (pending implementation)
```

### 🎯 **Priority 3: API Consistency Issues**
**Priority**: High - Essential for user workflow consistency

**Missing Features**:
- **Subgraph table access**: `subgraph.table()` missing
- **GraphTable multi-column slicing**: `table[['col1', 'col2']]` not implemented
- **GraphTable sort functionality**: `table.sort('column_name')` or `table.sort_values('column_name')` missing
- **GraphArray repr improvements**: Show actual values for debugging
- **GraphArray.values property**: Pandas-like raw data access
- **GraphArray type display**: Show data type in repr (e.g., `GraphArray(len=5, dtype=int, values=[1,2,3,4,5])`)
- **GraphArray slicing**: Should support slicing operations like GraphMatrix (e.g., `ages[:10]`, `ages[5:15]`, `ages[::2]`)
- **GraphTable sort**: Add sorting functionality (e.g., `table.sort('column_name')`, `table.sort_values('column_name')`)
- **GraphArray sort**: Add sorting functionality (e.g., `ages.sort()`, `ages.sort(ascending=False)`)
- **GraphMatrix sort**: Add column-wise sorting functionality (e.g., `matrix.sort('column_name')`, `matrix.sort_by_column(0)`)

### 🎯 **Priority 4: Performance Optimization**
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