# Shared Traits Migration Plan

## Overview

Migrating from individual subgraph implementations to a shared trait system for better maintainability and consistency across all subgraph-like types.

## Motivation

As we add more subgraph types (PyNeighborhoodSubgraph, PyPathSubgraph, PyClusterSubgraph, etc.), we need a consistent architecture that:
- ✅ Prevents code duplication
- ✅ Ensures consistent APIs across all subgraph types
- ✅ Makes it easy to add new subgraph types
- ✅ Allows type-specific optimizations
- ✅ Enables generic functions that work on any subgraph type

## Phase 1: Architecture Design

### New File Structure
```
python-groggy/src/ffi/traits/
├── mod.rs                     # Export all traits
├── subgraph_operations.rs     # Core SubgraphOperations trait
└── future_traits.rs           # Placeholder for other traits
```

### Core Trait Definition

All subgraph-like objects will implement the `SubgraphOperations` trait:

```rust
// python-groggy/src/ffi/traits/subgraph_operations.rs
use pyo3::prelude::*;
use crate::ffi::core::subgraph::PySubgraph;

/// Common operations that all subgraph-like objects support
pub trait SubgraphOperations {
    // === CORE DATA ACCESS ===
    fn table(&self, py: Python) -> PyResult<PyObject>;
    fn nodes(&self) -> Vec<usize>;
    fn edges(&self) -> Vec<usize>;
    fn size(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn node_count(&self) -> usize;
    
    // === ATTRIBUTE ACCESS ===
    fn get_node_attribute(&self, py: Python, node_id: usize, attr_name: String) -> PyResult<Option<PyObject>>;
    fn get_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String) -> PyResult<Option<PyObject>>;
    fn get_node_attribute_column(&self, py: Python, attr_name: String) -> PyResult<PyObject>;
    fn get_edge_attribute_column(&self, py: Python, attr_name: String) -> PyResult<PyObject>;
    fn set_node_attribute(&self, py: Python, node_id: usize, attr_name: String, value: PyObject) -> PyResult<()>;
    fn set_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String, value: PyObject) -> PyResult<()>;
    
    // === GRAPH ALGORITHMS ===
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>>;
    fn bfs(&self, py: Python, start_node: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>;
    fn dfs(&self, py: Python, start_node: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>;
    fn shortest_path(&self, py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>>;
    fn has_path(&self, py: Python, source: usize, target: usize) -> PyResult<bool>;
    fn degree(&self, py: Python, node_id: usize) -> PyResult<usize>;
    fn neighbors(&self, py: Python, node_id: usize) -> PyResult<Vec<usize>>;
    
    // === FILTERING & QUERYING ===
    fn filter_nodes(&self, py: Python, query: String) -> PyResult<PySubgraph>;
    fn filter_edges(&self, py: Python, query: String) -> PyResult<PySubgraph>;
    fn subgraph_from_nodes(&self, py: Python, node_ids: Vec<usize>) -> PyResult<PySubgraph>;
    fn subgraph_from_edges(&self, py: Python, edge_ids: Vec<usize>) -> PyResult<PySubgraph>;
    
    // === STATISTICS & ANALYSIS ===
    fn node_ids(&self) -> Vec<usize>;
    fn edge_ids(&self) -> Vec<usize>;
    fn edge_endpoints(&self, py: Python, edge_id: usize) -> PyResult<(usize, usize)>;
    fn adjacency_matrix(&self, py: Python) -> PyResult<PyObject>;
    fn incidence_matrix(&self, py: Python) -> PyResult<PyObject>;
    
    // === EXPORT/CONVERSION ===
    fn to_networkx(&self, py: Python) -> PyResult<PyObject>;
    fn to_dict(&self, py: Python) -> PyResult<PyObject>;
    
    // === ITERATION SUPPORT ===
    fn iter_nodes(&self, py: Python) -> PyResult<PyObject>;
    fn iter_edges(&self, py: Python) -> PyResult<PyObject>;
    fn iter_node_attributes(&self, py: Python, attr_name: String) -> PyResult<PyObject>;
    fn iter_edge_attributes(&self, py: Python, attr_name: String) -> PyResult<PyObject>;
    
    // === VIEW OPERATIONS ===
    fn view(&self, py: Python) -> PyResult<PySubgraph>;
    fn copy(&self, py: Python) -> PyResult<PySubgraph>;
    fn induced_subgraph(&self, py: Python, node_ids: Vec<usize>) -> PyResult<PySubgraph>;
    
    // === VALIDATION ===
    fn contains_node(&self, node_id: usize) -> bool;
    fn contains_edge(&self, edge_id: usize) -> bool;
    fn has_edge(&self, source: usize, target: usize) -> bool;
    
    // === DISPLAY & FORMATTING ===
    fn summary(&self) -> String;
    fn display_info(&self, py: Python) -> PyResult<String>;
}
```

## Phase 2: Implementation Strategy

### Step 1: Create the trait file ✅
### Step 2: Move PySubgraph methods to trait implementation 
### Step 3: Update PyNeighborhoodSubgraph to implement trait
### Step 4: Add PyO3 method delegation

### Method Delegation Pattern

Each PyO3 class delegates to the trait implementation:

```rust
#[pymethods]
impl PySubgraph {
    // PyO3 methods delegate to trait implementations
    fn table(&self, py: Python) -> PyResult<PyObject> {
        SubgraphOperations::table(self, py)
    }
    
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>> {
        SubgraphOperations::connected_components(self, py)
    }
    
    // ... all other methods delegate to trait
}

#[pymethods] 
impl PyNeighborhoodSubgraph {
    // === UNIQUE NEIGHBORHOOD METHODS ===
    fn central_nodes(&self) -> Vec<usize> { 
        self.inner.central_nodes.clone()
    }
    
    fn hops(&self) -> usize { 
        self.inner.hops 
    }
    
    // === COMMON METHODS DELEGATE TO TRAIT ===
    fn table(&self, py: Python) -> PyResult<PyObject> {
        SubgraphOperations::table(self, py)
    }
    
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>> {
        SubgraphOperations::connected_components(self, py)
    }
    
    // ... all other common methods delegate to trait
}
```

## Phase 3: Future Subgraph Types

With this architecture, adding new subgraph types becomes trivial:

```rust
#[pyclass(name = "PathSubgraph")]
pub struct PyPathSubgraph {
    pub nodes: Vec<usize>,
    pub edges: Vec<usize>,
    pub path_length: f64,
    pub source: usize,
    pub target: usize,
}

impl SubgraphOperations for PyPathSubgraph {
    // Default trait implementations work for most methods
    // Custom implementations for performance-critical methods
    
    fn shortest_path(&self, py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>> {
        // Path-specific optimization: we already ARE the shortest path!
        if source == self.source && target == self.target {
            Ok(Some(self.view(py)?))
        } else {
            // Fall back to general implementation
            todo!("implement general shortest path")
        }
    }
}

#[pymethods]
impl PyPathSubgraph {
    // Unique path methods
    fn path_length(&self) -> f64 { self.path_length }
    fn source_node(&self) -> usize { self.source }
    fn target_node(&self) -> usize { self.target }
    
    // Common methods delegate to trait
    fn table(&self, py: Python) -> PyResult<PyObject> {
        SubgraphOperations::table(self, py)
    }
    // ... etc
}
```

## Phase 4: Style Guide & Standards

### Trait Implementation Rules

1. **Default implementations** for common patterns where possible
2. **Required methods** for type-specific behavior  
3. **Consistent error handling** using `PyResult<T>`
4. **Comprehensive documentation** for all trait methods
5. **Performance notes** in docstrings for optimization opportunities

### Error Handling Standards

```rust
impl SubgraphOperations for PyCustomSubgraph {
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Always use consistent error types
        self.ensure_inner()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Subgraph error: {}", e)
            ))?;
        
        // Implementation...
    }
}
```

### Documentation Standards

```rust
/// Get table representation of subgraph nodes with attributes
/// 
/// Returns a GraphTable containing all nodes and their attributes,
/// suitable for data analysis and filtering operations.
/// 
/// # Arguments
/// * `py` - Python context for creating Python objects
/// 
/// # Returns  
/// * `PyResult<PyObject>` - GraphTable as Python object
/// 
/// # Performance Notes
/// * O(n) where n is number of nodes
/// * Lazy evaluation - table computed on demand
/// * Consider caching for repeated calls
fn table(&self, py: Python) -> PyResult<PyObject>;
```

## Phase 5: Migration Execution Plan

### Priority Order
1. **Create trait definition** - establish interface
2. **Implement for PySubgraph** - move existing methods
3. **Implement for PyNeighborhoodSubgraph** - enable delegation  
4. **Add comprehensive tests** - ensure correctness
5. **Performance validation** - no regressions
6. **Documentation updates** - user-facing docs

### Testing Strategy
- **Unit tests** for each trait implementation
- **Integration tests** for cross-type consistency
- **Performance benchmarks** to ensure no regressions
- **API compatibility tests** to ensure Python interface unchanged

## Benefits This Architecture Unlocks

- ✅ **Generic functions**: `fn analyze<T: SubgraphOperations>(subgraph: &T)`
- ✅ **Consistent APIs**: All subgraphs have same core methods
- ✅ **Type-specific optimizations**: Each type can optimize differently
- ✅ **Easy testing**: Mock implementations for testing
- ✅ **Future-proof**: New types just implement the trait
- ✅ **Maintainability**: Single source of truth for behavior
- ✅ **Extensibility**: Easy to add domain-specific subgraph types

## Expected Timeline

- **Phase 1 (Design)**: 1 day
- **Phase 2 (Core Implementation)**: 2-3 days  
- **Phase 3 (PyNeighborhoodSubgraph)**: 1 day
- **Phase 4 (Testing & Validation)**: 1 day
- **Phase 5 (Documentation)**: 1 day

**Total: ~1 week for complete migration**