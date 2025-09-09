# Implementation Steps: Unified Delegation Architecture

## Current Status ✅
- ComponentsArray → SubgraphArray delegation working
- TableArray with basic delegation complete  
- Universal ArrayIterator pattern established
- Method chaining: `g.connected_components().iter().table().collect()` ✅

## Architecture Update: BaseArray vs StatsArray Separation ✅
- **BaseArray**: Contains only basic operations (len, is_empty, get, iter)
- **StatsArray**: Inherits BaseArray + adds numerical/statistical operations
- **Delegation Pattern**: TableArray and MatrixArray use StatsArray, others use BaseArray
- **Purpose**: Clean separation between basic array operations and statistical computations

## Phase 2: Create Missing Specialized Arrays

### Step 2.1: NodesArray Implementation
**Goal**: Collections of NodesAccessor objects with delegation

**Files to Create/Modify**:
- `src/ffi/storage/nodes_array.rs` (new)
- `src/ffi/storage/mod.rs` (add export)
- `src/lib.rs` (register class)

**Implementation**:
```rust
#[pyclass(name = "NodesArray", unsendable)]
pub struct PyNodesArray {
    // Delegates to BaseArray for basic operations
    base: BaseArray<PyNodesAccessor>,
}

#[pymethods]
impl PyNodesArray {
    // BaseArray delegation
    fn len(&self) -> usize { self.base.len() }
    fn is_empty(&self) -> bool { self.base.is_empty() }
    fn get(&self, index: usize) -> Option<PyNodesAccessor> { 
        self.base.get(index).cloned() 
    }
    
    // Domain-specific methods (apply_on_each pattern)
    fn connected_components(&self) -> PyResult<PySubgraphArray>;
    fn table(&self) -> PyResult<PyTableArray>;
    fn filter(&self, query: String) -> PyResult<Self>;
    
    // Reduce operations (many-to-one)
    fn union(&self) -> PyResult<PyNodesAccessor>; // Combine all node sets
    fn to_list(&self) -> Vec<PyNodesAccessor>;    // Convert to Python list
}

// No separate iterator class needed - methods are directly on the array!
```

**Usage Enabled**:
```python
# Direct method calls - no iter().collect()
nodes_arrays = g.connected_components().nodes()        # NodesArray

# Chain operations directly on arrays
filtered = g.some_nodes_array.filter("degree > 5")    # NodesArray
components = g.nodes_array.connected_components()      # SubgraphArray
```

---

### Step 2.2: EdgesArray Implementation  
**Goal**: Collections of EdgesAccessor objects with delegation

**Files to Create/Modify**:
- `src/ffi/storage/edges_array.rs` (new)
- `src/ffi/storage/mod.rs` (add export)
- `src/lib.rs` (register class)

**Implementation**:
```rust
#[pyclass(name = "EdgesArray", unsendable)]
pub struct PyEdgesArray {
    // Delegates to BaseArray for basic operations
    base: BaseArray<PyEdgesAccessor>,
}

#[pymethods]
impl PyEdgesArray {
    // BaseArray delegation
    fn len(&self) -> usize { self.base.len() }
    fn is_empty(&self) -> bool { self.base.is_empty() }
    fn get(&self, index: usize) -> Option<PyEdgesAccessor> { 
        self.base.get(index).cloned() 
    }
    
    fn iter(&self) -> PyEdgesArrayChainIterator;
    fn union(&self) -> PyResult<PyEdgesAccessor>; // Combine all edge sets
    fn collect(&self) -> Vec<PyEdgesAccessor>;
}

#[pyclass(name = "EdgesArrayIterator", unsendable)]
pub struct PyEdgesArrayChainIterator {
    inner: ArrayIterator<PyEdgesAccessor>,
}

#[pymethods]  
impl PyEdgesArrayChainIterator {
    fn connected_components(&mut self) -> PyResult<PySubgraphArray>;
    fn table(&mut self) -> PyResult<PyTableArray>;
    fn filter(&mut self, query: String) -> PyResult<Self>;
    fn nodes(&mut self) -> PyResult<PyNodesArray>; // Get source/target nodes
    fn collect(&mut self) -> PyResult<PyEdgesArray>;
}
```

**Usage Enabled**:
```python
# Get edges from multiple subgraphs
edge_arrays = g.connected_components().iter().edges().collect()

# Filter edge collections
high_weight = g.edges_array.iter().filter("weight > 0.8").collect()
```

---

### Step 2.3: MatrixArray Implementation
**Goal**: Collections of Matrix objects with delegation

**Files to Create/Modify**:
- `src/ffi/storage/matrix_array.rs` (new)  
- `src/ffi/storage/mod.rs` (add export)
- `src/lib.rs` (register class)

**Implementation**:
```rust
#[pyclass(name = "MatrixArray", unsendable)]
pub struct PyMatrixArray {
    // Delegates to StatsArray for numerical operations
    stats: StatsArray<PyGraphMatrix>,
}

#[pymethods]
impl PyMatrixArray {
    // BaseArray delegation (via StatsArray)
    fn len(&self) -> usize { self.stats.base.len() }
    fn is_empty(&self) -> bool { self.stats.base.is_empty() }
    fn get(&self, index: usize) -> Option<PyGraphMatrix> { 
        self.stats.base.get(index).cloned() 
    }
    
    // StatsArray delegation - statistical operations on matrix collections
    fn mean_eigenvalue(&self) -> PyResult<f64>; // Average eigenvalue across matrices
    fn correlation_matrix(&self) -> PyResult<PyGraphMatrix>; // Correlation of matrix elements
    
    fn iter(&self) -> PyMatrixArrayChainIterator;
    fn stack(&self) -> PyResult<PyGraphMatrix>; // Stack matrices
    fn collect(&self) -> Vec<PyGraphMatrix>;
}

#[pyclass(name = "MatrixArrayIterator", unsendable)]
pub struct PyMatrixArrayChainIterator {
    inner: ArrayIterator<PyGraphMatrix>,
}

#[pymethods]
impl PyMatrixArrayChainIterator {
    fn multiply(&mut self, other: &Self) -> PyResult<Self>;
    fn eigen(&mut self) -> PyResult<PyStatsArray>; // Eigenvalues as StatsArray
    fn transform(&mut self, operation: String) -> PyResult<Self>;
    fn collect(&mut self) -> PyResult<PyMatrixArray>;
}
```

**Usage Enabled**:
```python
# Get matrices from multiple subgraphs  
matrices = g.connected_components().iter().matrix().collect()

# Matrix operations on collections
eigenvals = matrices.iter().eigen().collect()
```

---

## Phase 2.5: BaseArray and StatsArray Foundation

### Step 2.5.1: Create BaseArray Implementation
**Goal**: Universal base class for all array operations

**Files to Create/Modify**:
- `src/storage/array/base_array.rs` (new)
- `src/storage/array/mod.rs` (add export)

**Implementation**:
```rust
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BaseArray<T> {
    pub inner: Arc<Vec<T>>,
}

impl<T> BaseArray<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { inner: Arc::new(data) }
    }
    
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }
    pub fn get(&self, index: usize) -> Option<&T> { self.inner.get(index) }
    pub fn iter(&self) -> impl Iterator<Item = &T> { self.inner.iter() }
    pub fn clone_vec(&self) -> Vec<T> where T: Clone { (*self.inner).clone() }
}

impl<T> From<Vec<T>> for BaseArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}
```

---

### Step 2.5.2: Create StatsArray Implementation  
**Goal**: Statistical operations layer on top of BaseArray

**Files to Create/Modify**:
- `src/storage/array/stats_array.rs` (new)
- `src/storage/array/mod.rs` (add export)

**Implementation**:
```rust
use super::BaseArray;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]  
pub struct StatsArray<T> {
    pub base: BaseArray<T>,
}

impl<T> StatsArray<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { base: BaseArray::new(data) }
    }
    
    // Delegate basic operations to BaseArray
    pub fn len(&self) -> usize { self.base.len() }
    pub fn is_empty(&self) -> bool { self.base.is_empty() }
    pub fn get(&self, index: usize) -> Option<&T> { self.base.get(index) }
    pub fn iter(&self) -> impl Iterator<Item = &T> { self.base.iter() }
}

impl<T> StatsArray<T> 
where 
    T: Copy + Add<Output = T> + Mul<f64, Output = T> + PartialOrd + Into<f64>
{
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() { return None; }
        let sum: f64 = self.iter().map(|&x| x.into()).sum();
        Some(sum / self.len() as f64)
    }
    
    pub fn sum(&self) -> f64 {
        self.iter().map(|&x| x.into()).sum()
    }
    
    pub fn min(&self) -> Option<f64> {
        self.iter().map(|&x| x.into()).min_by(|a, b| a.partial_cmp(b).unwrap())
    }
    
    pub fn max(&self) -> Option<f64> {
        self.iter().map(|&x| x.into()).max_by(|a, b| a.partial_cmp(b).unwrap())
    }
    
    pub fn std_dev(&self) -> Option<f64> {
        let mean = self.mean()?;
        if self.len() <= 1 { return Some(0.0); }
        
        let variance = self.iter()
            .map(|&x| {
                let diff = x.into() - mean;
                diff * diff
            })
            .sum::<f64>() / (self.len() - 1) as f64;
        
        Some(variance.sqrt())
    }
    
    pub fn correlate(&self, other: &Self) -> Option<f64> {
        if self.len() != other.len() || self.is_empty() { return None; }
        
        let self_mean = self.mean()?;
        let other_mean = other.mean()?;
        
        let numerator: f64 = self.iter().zip(other.iter())
            .map(|(&x, &y)| (x.into() - self_mean) * (y.into() - other_mean))
            .sum();
            
        let self_var: f64 = self.iter()
            .map(|&x| (x.into() - self_mean).powi(2))
            .sum();
            
        let other_var: f64 = other.iter()
            .map(|&y| (y.into() - other_mean).powi(2))
            .sum();
        
        let denominator = (self_var * other_var).sqrt();
        if denominator == 0.0 { None } else { Some(numerator / denominator) }
    }
}

impl<T> From<Vec<T>> for StatsArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}
```

**Usage**: TableArray and MatrixArray will use StatsArray, others use BaseArray

---

## Phase 3: Cross-Type Conversions

### Step 3.1: Add Conversion Methods to Core Objects

**Subgraph Conversions**:
```rust
// In src/ffi/subgraphs/subgraph.rs
#[pymethods]
impl PySubgraph {
    // Already have: table(), sample(), neighborhood()
    
    pub fn nodes(&self) -> PyResult<PyNodesAccessor>;
    pub fn edges(&self) -> PyResult<PyEdgesAccessor>; 
    pub fn matrix(&self) -> PyResult<PyGraphMatrix>;
}
```

**NodesAccessor Conversions**:
```rust  
// In src/ffi/storage/accessors.rs
#[pymethods]
impl PyNodesAccessor {
    // Already have: table(), connected_components()
    
    pub fn subgraphs(&self) -> PyResult<PySubgraphArray>; // Create subgraphs from node groups
    pub fn edges(&self) -> PyResult<PyEdgesAccessor>; // Get incident edges
}
```

**EdgesAccessor Conversions**:
```rust
// In src/ffi/storage/accessors.rs  
#[pymethods]
impl PyEdgesAccessor {
    pub fn nodes(&self) -> PyResult<PyNodesAccessor>; // Get source/target nodes
    pub fn subgraphs(&self) -> PyResult<PySubgraphArray>; // Create subgraphs from edge groups
    pub fn table(&self) -> PyResult<PyObject>; // Convert to table
    pub fn connected_components(&self) -> PyResult<PySubgraphArray>;
}
```

**Table Conversions**:
```rust
// In src/ffi/storage/table.rs
#[pymethods]  
impl PyGraphTable {
    pub fn nodes(&self) -> PyResult<PyNodesAccessor>; // If has node_id column
    pub fn edges(&self) -> PyResult<PyEdgesAccessor>; // If has edge_id column  
    pub fn subgraphs(&self) -> PyResult<PySubgraphArray>; // Group by component_id
    
    // Statistical conversions - extract numerical columns
    pub fn stats(&self, columns: Vec<String>) -> PyResult<PyStatsArray>; // Extract numerical data
    pub fn correlation_matrix(&self) -> PyResult<PyStatsArray>; // Column correlations
}
```

**Matrix Conversions**:
```rust
// In src/ffi/storage/matrix.rs
#[pymethods]
impl PyGraphMatrix {
    pub fn subgraphs(&self) -> PyResult<PySubgraphArray>; // Connected components from adjacency
    pub fn table(&self) -> PyResult<PyObject>; // Convert to edge list table
    pub fn nodes(&self) -> PyResult<PyNodesAccessor>; // Get all nodes
    pub fn edges(&self) -> PyResult<PyEdgesAccessor>; // Convert to edge list
}
```

---

### Step 3.2: Implement Delegation Traits

**Create Delegation Trait System**:
```rust
// In src/storage/array/delegation.rs (new file)

pub trait HasTable {
    fn table(&self, py: Python) -> PyResult<PyObject>;
}

pub trait HasNodes {  
    fn nodes(&self) -> PyResult<PyNodesAccessor>;
}

pub trait HasEdges {
    fn edges(&self) -> PyResult<PyEdgesAccessor>;
}

pub trait HasSubgraphs {
    fn subgraphs(&self) -> PyResult<PySubgraphArray>;
}

pub trait HasMatrix {
    fn matrix(&self) -> PyResult<PyGraphMatrix>;
}

// Implement for all relevant types
impl HasTable for PySubgraph { /* delegate to subgraph.table() */ }
impl HasNodes for PySubgraph { /* delegate to subgraph.nodes() */ }
// ... etc for all combinations
```

**Update ArrayIterator with Trait-Based Methods**:
```rust
// In src/storage/array/iterator.rs
impl<T> ArrayIterator<T> {
    pub fn table(self) -> PyResult<PyTableArray> 
    where T: HasTable + Clone + 'static {
        let tables: Vec<PyObject> = self.elements
            .into_iter()
            .filter_map(|element| {
                Python::with_gil(|py| element.table(py).ok())
            })
            .collect();
        Ok(PyTableArray::new(tables))
    }
    
    pub fn nodes(self) -> PyResult<PyNodesArray>
    where T: HasNodes + Clone + 'static {
        let nodes: Vec<PyNodesAccessor> = self.elements
            .into_iter()
            .filter_map(|element| element.nodes().ok())
            .collect();
        Ok(PyNodesArray::new(nodes))
    }
    
    // Similar for edges(), subgraphs(), matrix()
}
```

---

### Step 3.3: Enable Universal Method Chaining

**Update All Array Iterators**:
- Add trait-based delegation methods to every array iterator
- Ensure type safety with proper trait bounds
- Add comprehensive error handling

**Example Result**:
```python
# Multi-hop conversions now work everywhere
g.nodes().connected_components().iter().edges().iter().table().collect()
g.edges().connected_components().iter().nodes().iter().subgraphs().collect()  
g.tables_array.iter().nodes().iter().connected_components().collect()

# Complex chains become natural
analysis = (g.connected_components()           # SubgraphArray
           .iter()                             # SubgraphArrayIterator  
           .filter(lambda sg: len(sg) > 50)    # SubgraphArrayIterator
           .edges()                            # EdgesArray
           .iter()                             # EdgesArrayIterator
           .filter("weight > 0.5")             # EdgesArrayIterator  
           .nodes()                            # NodesArray
           .iter()                             # NodesArrayIterator
           .table()                            # TableArray
           .collect())                         # List[Table]
```

---

## Phase 4: Integration & Optimization

### Step 4.1: Deprecate ComponentsArray
- Add deprecation warnings to ComponentsArray  
- Update all examples to use SubgraphArray
- Create migration guide

### Step 4.2: Performance Optimization
- Benchmark delegation overhead
- Implement lazy materialization everywhere
- Add zero-copy optimizations  
- Parallel processing for large arrays

### Step 4.3: Documentation & Testing
- Comprehensive examples for all conversion paths
- Performance benchmarks
- Integration tests for complex chains
- Update API documentation

---

## Verification Tests

Create comprehensive test cases for each phase:

```python
def test_unified_delegation():
    g = gr.generators.watts_strogatz(1000, 6, 0.3)
    
    # Test all conversion paths work
    components = g.connected_components()
    nodes = components.iter().nodes().collect()  
    edges = components.iter().edges().collect()
    tables = components.iter().table().collect()
    matrices = components.iter().matrix().collect()
    
    # Test reverse conversions
    back_to_subgraphs = nodes.iter().connected_components().collect()
    
    # Test complex chains
    result = (g.nodes()
             .connected_components()
             .iter()
             .filter(lambda sg: len(sg) > 100)
             .edges() 
             .iter()
             .table()
             .collect())
    
    assert len(result) > 0
    assert all(isinstance(table, gr.Table) for table in result)
```

This implementation plan transforms Groggy into a truly unified graph ecosystem! 🚀