# PySubgraph Refactoring Plan: Eliminate Parallel Systems

**Status**: Planning Phase  
**Goal**: Eliminate code duplication and architectural inconsistency in PySubgraph implementation  
**Impact**: ~300+ lines deleted, unified delegation pattern  

## Problem Statement

We currently have **three parallel systems** for subgraph operations:

1. ❌ **Direct PySubgraph methods** (`ffi/core/subgraph.rs`) - 500+ lines of reimplementation
2. ✅ **PySubgraphOperations trait** (`ffi/traits/subgraph_operations.rs`) - Pure delegation pattern  
3. ✅ **Core SubgraphOperations trait** (`src/core/traits/`) - Actual implementations

This creates maintenance burden, code duplication, and architectural inconsistency with PyGraph's delegation pattern.

## Solution Architecture

### Pattern: Trait-Based Delegation

Follow the same pattern as PyGraph:
- **PyGraph** → delegates to **PyGraphAnalysis**, **PyGraphMatrix**, etc.
- **PySubgraph** → delegates to **PySubgraphOperations trait**

### Final Structure Overview

```
PySubgraph (Python class)
├── Python-specific methods (accessors, __getitem__, etc.)  
├── implements PySubgraphOperations trait
└── delegates core methods to trait default implementations
    └── trait calls self.core_subgraph() 
        └── returns &dyn CoreSubgraphOperations
            └── actual algorithm implementations
```

## Proposed Final Structure

### A. Core PySubgraph Class (`ffi/core/subgraph.rs`)

**Size: ~150 lines** (down from ~500 lines)

```rust
//! Simplified PySubgraph - Pure delegation to trait implementations

use crate::ffi::traits::subgraph_operations::PySubgraphOperations;
use groggy::core::traits::SubgraphOperations as CoreSubgraphOperations;
use groggy::core::subgraph::Subgraph;
use pyo3::prelude::*;

#[pyclass(name = "Subgraph", unsendable)]
pub struct PySubgraph {
    pub inner: Subgraph,
}

// ============================================================================
// TRAIT IMPLEMENTATION - Core delegation pattern
// ============================================================================

impl PySubgraphOperations for PySubgraph {
    /// Provide access to core trait object for delegation
    fn core_subgraph(&self) -> PyResult<&dyn CoreSubgraphOperations> {
        Ok(&self.inner)
    }
    
    /// Override trait method for concrete Subgraph downcast when needed
    fn try_downcast_to_subgraph(&self) -> Option<&groggy::core::subgraph::Subgraph> {
        Some(&self.inner)
    }
}

// ============================================================================
// PYTHON-SPECIFIC METHODS - Keep these for ergonomic Python API
// ============================================================================

#[pymethods]
impl PySubgraph {
    // === PYTHON ACCESSORS (Different from trait) ===
    
    /// Get nodes accessor for Python ergonomics: subgraph.nodes[id], subgraph.nodes['attr']
    fn nodes(&self, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        Py::new(py, PyNodesAccessor {
            graph: self.inner.graph(),
            constrained_nodes: Some(self.inner.node_set().iter().copied().collect()),
        })
    }
    
    /// Get edges accessor for Python ergonomics: subgraph.edges[id] 
    fn edges(&self, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        Py::new(py, PyEdgesAccessor {
            graph: self.inner.graph(),
            constrained_edges: Some(self.inner.edge_set().iter().copied().collect()),
        })
    }
    
    // === PYTHON SYNTAX SUPPORT ===
    
    /// Support subgraph['attribute'] syntax for bulk attribute access
    fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> {
        if let Ok(attr_name) = key.extract::<String>() {
            // Return GraphArray of attribute values for nodes in subgraph
            // Implementation stays the same as current
        }
        Err(PyTypeError::new_err("Subgraph indexing only supports string attribute names"))
    }
    
    /// String representations
    fn __repr__(&self) -> String {
        format!("Subgraph(nodes={}, edges={})", self.inner.node_count(), self.inner.edge_count())
    }
    
    fn __str__(&self) -> String {
        format!("Subgraph with {} nodes and {} edges", self.inner.node_count(), self.inner.edge_count())
    }
    
    // === DELEGATION WRAPPERS - Where Python API differs from trait ===
    
    /// Filter nodes - delegate to trait but handle Python types properly
    fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Convert Python filter to appropriate type, then delegate to trait
        let result = if let Ok(query_str) = filter.extract::<String>() {
            self.filter_nodes_by_string(py, query_str)?
        } else {
            // Handle filter objects
            self.filter_nodes_by_object(py, filter)?
        };
        Ok(result)
    }
    
    /// Filter edges - delegate to trait but handle Python types properly  
    fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Similar pattern to filter_nodes
        // Delegate to trait methods with proper type conversion
    }
    
    /// Neighborhood - delegate to trait but return PyNeighborhoodResult
    fn neighborhood(&self, py: Python, central_nodes: Vec<NodeId>, hops: usize) -> PyResult<PyNeighborhoodResult> {
        // Use the graph_analysis wrapper pattern from our earlier fix
        self.neighborhood_internal(py, central_nodes, hops)
    }
    
    /// Calculate similarity - delegate directly to trait (signatures match)
    fn calculate_similarity(&self, other: &PySubgraph, metric: &str, py: Python) -> PyResult<f64> {
        PySubgraphOperations::calculate_similarity(self, py, other, metric.to_string())
    }
}

// ============================================================================
// INTERNAL HELPERS - Type conversion and delegation
// ============================================================================

impl PySubgraph {
    /// Create from core Subgraph
    pub fn from_core(inner: Subgraph) -> Self {
        Self { inner }
    }
    
    // Internal delegation helpers for complex type conversions
    fn filter_nodes_by_string(&self, py: Python, query: String) -> PyResult<PySubgraph> {
        // Parse string query and delegate to trait
    }
    
    fn neighborhood_internal(&self, py: Python, central_nodes: Vec<NodeId>, hops: usize) -> PyResult<PyNeighborhoodResult> {
        // Create temporary PyGraph and use existing graph_analysis pattern
        // This handles the multi-signature dispatch we implemented earlier
    }
}
```

### B. PySubgraphOperations Trait (`ffi/traits/subgraph_operations.rs`)

**Status: Already exists - minimal changes needed**

```rust
//! Pure delegation trait - already implemented correctly

pub trait PySubgraphOperations {
    /// Core delegation method - must be implemented by concrete types
    fn core_subgraph(&self) -> PyResult<&dyn CoreSubgraphOperations>;
    
    /// Try downcast for concrete methods (already exists)
    fn try_downcast_to_subgraph(&self) -> Option<&groggy::core::subgraph::Subgraph> {
        None  // Override in concrete implementations
    }
    
    // All delegation methods already exist:
    // - node_count(), edge_count(), contains_node(), etc.
    // - connected_components(), bfs(), dfs(), etc. 
    // - calculate_similarity(), merge_with(), etc.
    // - set_node_attrs(), set_edge_attrs(), etc.
    
    // ADD: Filter methods that don't exist yet
    fn filter_nodes_internal(&self, py: Python, query: String) -> PyResult<PySubgraph> {
        py.allow_threads(|| {
            let filtered = self.core_subgraph()?.filter_nodes(&query.into())
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
            Ok(PySubgraph::from_core(filtered))
        })
    }
    
    fn filter_edges_internal(&self, py: Python, query: String) -> PyResult<PySubgraph> {
        py.allow_threads(|| {
            let filtered = self.core_subgraph()?.filter_edges(&query.into())
                .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
            Ok(PySubgraph::from_core(filtered))
        })
    }
}
```

### C. Import and Registration (`ffi/mod.rs` and `lib.rs`)

```rust
// ffi/mod.rs
pub mod traits {
    pub mod subgraph_operations;
}

// lib.rs - no changes needed, PySubgraph already registered
```

## Implementation Pattern

### Delegation Pattern Template

For any method that needs delegation:

```rust
// 1. Python-facing method in PySubgraph
fn python_method(&self, py: Python, args: PythonArgs) -> PyResult<PythonReturnType> {
    // Convert Python types to Rust types
    let rust_args = convert_python_args(args)?;
    
    // Delegate to trait method
    let rust_result = PySubgraphOperations::trait_method(self, py, rust_args)?;
    
    // Convert Rust types back to Python types if needed
    convert_to_python_result(rust_result)
}

// 2. Trait method with pure delegation
fn trait_method(&self, py: Python, args: RustArgs) -> PyResult<RustReturnType> {
    py.allow_threads(|| {
        self.core_subgraph()?.core_method(args)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    })
}
```

### Type Conversion Patterns

```rust
// String query → Rust Filter
let filter = parse_query_string(query_str)?;

// Python NodeId list → Rust NodeId Vec  
let node_ids: Vec<NodeId> = py_nodes.iter().map(|n| n.extract().unwrap()).collect();

// Rust Subgraph → Python PySubgraph
PySubgraph::from_core(rust_subgraph)

// Core result → Python result with error handling
core_result.map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
```

## Deletion Plan

### Methods to DELETE from PySubgraph (~300+ lines):

```rust
// DELETE these - they'll come from trait
fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> { ... }        // 92 lines
fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> { ... }       // 92 lines  
fn neighborhood(&self, py: Python, ...) -> PyResult<PyNeighborhoodResult> { ... }        // 65 lines
fn calculate_similarity(&self, other: &PySubgraph, ...) -> PyResult<f64> { ... }         // 25 lines
// Any other methods that duplicate trait functionality
```

### Methods to KEEP in PySubgraph (~150 lines):

```rust  
// KEEP these - Python-specific ergonomics
fn nodes(&self, py: Python) -> PyResult<Py<PyNodesAccessor>> { ... }                     // Returns accessor, not list
fn edges(&self, py: Python) -> PyResult<Py<PyEdgesAccessor>> { ... }                     // Returns accessor, not list  
fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> { ... }             // Python syntax support
fn __str__(&self) -> String { ... }                                                      // Python display
fn __repr__(&self) -> String { ... }                                                     // Python display

// KEEP but refactor - delegation wrappers where APIs differ
fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> { ... }       // 10 lines - just delegates
fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> { ... }       // 10 lines - just delegates
```

## Implementation Steps

### Phase 1: Preparation
1. ✅ Verify PySubgraphOperations trait has all needed methods
2. ⏳ Add missing methods to trait if needed (filter_nodes_internal, etc.)
3. ⏳ Test trait methods work correctly

### Phase 2: PySubgraph Refactoring  
1. ⏳ Add trait implementation to PySubgraph
2. ⏳ Replace method implementations with delegation wrappers
3. ⏳ Delete duplicate implementations
4. ⏳ Update imports and dependencies

### Phase 3: Testing & Validation
1. ⏳ Ensure all Python API functionality works
2. ⏳ Verify performance (should be same or better)  
3. ⏳ Run existing tests
4. ⏳ Test edge cases and error handling

## Benefits After Refactoring

### ✅ Code Reduction
- **Before**: ~500 lines in PySubgraph  
- **After**: ~150 lines in PySubgraph
- **Savings**: ~350 lines of duplicate code eliminated

### ✅ Architectural Consistency
- PyGraph pattern: **Graph** → **Helper Classes** → **Core**
- PySubgraph pattern: **Subgraph** → **Trait** → **Core**  
- Unified delegation approach

### ✅ Maintenance Benefits
- Single source of truth for algorithm logic
- Bug fixes in core automatically benefit all subgraph types
- New core methods automatically available via trait

### ✅ Performance Benefits  
- Trait methods use `py.allow_threads()` correctly
- Optimized core implementations
- No FFI overhead duplication

## Risk Mitigation

### Compatibility Testing
- Test all public Python methods work identically
- Verify error messages and types match
- Check performance characteristics  

### Rollback Plan
- Keep backup of original subgraph.rs
- Implement changes in feature branch
- Incremental testing at each step

### Method Signature Verification
- Document any API changes needed
- Ensure Python ergonomics maintained
- Test with existing user code

---

**Next Action**: Review this plan, then proceed with Phase 1 implementation.