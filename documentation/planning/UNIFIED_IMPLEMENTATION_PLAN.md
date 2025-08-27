# Unified SubgraphOperations & HierarchicalOperations Implementation Plan

## 📋 **EXECUTIVE SUMMARY**

**Total Methods Needed**: 70 methods (45 SubgraphOperations + 25 HierarchicalOperations)
**Implementation Status**: Clean slate approach - focus on missing core methods + FFI traits

### **🚫 DO NOT REIMPLEMENT (Already Optimized):**
- `connected_components()` - **4.5x faster than NetworkX** - ALREADY OPTIMIZED ✅
- `bfs()`, `dfs()` - **Use TraversalEngine** - ALREADY OPTIMIZED ✅  
- `shortest_path()`, `degree()` - **Core Graph methods** - ALREADY OPTIMIZED ✅
- **~35 methods** can delegate to existing optimized implementations

### **✅ NEED IMPLEMENTATION (Missing from Core):**
- **~10 SubgraphOperations**: structural metrics, set operations
- **~8 HierarchicalOperations**: node conversion, subgraph aggregation, graph integration
- **Total: ~18 NEW methods** need core implementation

### **🎯 Implementation Strategy:**
- **Phase 1**: Implement missing methods in core Rust
- **Phase 2**: Create FFI traits that delegate to core (existing + new)
- **Phase 3**: Integration, testing, and optimization

---

# COMPLETE IMPLEMENTATION PLAN

## **Phase 1: Core Rust Implementation (Weeks 1-3)**

### **Week 1: Missing SubgraphOperations in Core**
**File Location**: `/Users/michaelroth/Documents/Code/groggy/src/core/subgraph.rs`

⚠️  **CRITICAL: DO NOT REIMPLEMENT EXISTING OPTIMIZED METHODS** ⚠️
- `connected_components()`, `bfs()`, `dfs()`, `shortest_path()`, `degree()` are ALREADY OPTIMIZED
- These methods should only be wrapped in FFI traits, NOT reimplemented

**NEW Methods to implement in `impl Subgraph`:**

```rust
// === STRUCTURAL METRICS ===
pub fn clustering_coefficient(&self, node_id: Option<NodeId>) -> GraphResult<f64> {
    // Calculate clustering coefficient for node or average for all nodes
    // Formula: 2 * triangles / (degree * (degree - 1))
}

pub fn transitivity(&self) -> GraphResult<f64> {
    // Calculate global clustering coefficient
    // Formula: 3 * triangles / triads
}

pub fn density(&self) -> f64 {
    // Calculate graph density
    // Formula: 2 * edges / (nodes * (nodes - 1))
}

pub fn is_connected(&self) -> GraphResult<bool> {
    // Use existing BFS to check connectivity
    // Run BFS from first node, check if all nodes reached
}

// === SUBGRAPH SET OPERATIONS ===
pub fn merge_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
    // Union of nodes and edges from both subgraphs
    // Preserve attributes from both sources
}

pub fn intersect_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
    // Intersection of nodes and induced edges
    // Keep attributes from self (primary subgraph)
}

pub fn subtract_from(&self, other: &Subgraph) -> GraphResult<Subgraph> {
    // Remove other's nodes/edges from self
    // Return remaining subgraph
}

pub fn calculate_similarity(&self, other: &Subgraph, metric: SimilarityMetric) -> GraphResult<f64> {
    // Calculate Jaccard, Dice, or Cosine similarity between subgraphs
    // Based on node overlap, edge overlap, or attribute similarity
}

pub fn find_overlaps(&self, others: Vec<&Subgraph>) -> GraphResult<Vec<Subgraph>> {
    // Find all overlapping regions between this subgraph and others
    // Return list of intersection subgraphs
}

#[derive(Debug, Clone)]
pub enum SimilarityMetric {
    Jaccard,     // |A ∩ B| / |A ∪ B|
    Dice,        // 2 * |A ∩ B| / (|A| + |B|)
    Cosine,      // A·B / (||A|| * ||B||)
    Overlap,     // |A ∩ B| / min(|A|, |B|)
}
```

### **Week 2: Enhanced Query & Filter Operations** 
**File Location**: `/Users/michaelroth/Documents/Code/groggy/src/core/subgraph.rs`

```rust
// === ADVANCED FILTERING ===
pub fn filter_nodes_complex(&self, filter: &NodeFilter) -> GraphResult<Subgraph> {
    // Enhanced node filtering with complex query support
    // Support for: attribute ranges, boolean logic, regex patterns
}

pub fn filter_edges_complex(&self, filter: &EdgeFilter) -> GraphResult<Subgraph> {
    // Enhanced edge filtering with complex query support
    // Support for: weight ranges, attribute filters, endpoint conditions
}

pub fn query_nodes_advanced(&self, query: &NodeQuery) -> GraphResult<Vec<NodeId>> {
    // Advanced node querying with SQL-like syntax
    // Support for: WHERE clauses, ORDER BY, LIMIT
}

pub fn query_edges_advanced(&self, query: &EdgeQuery) -> GraphResult<Vec<EdgeId>> {
    // Advanced edge querying with SQL-like syntax
    // Support for: JOIN conditions, aggregations
}
```

### **Week 3: HierarchicalOperations Core Implementation**
**File Location**: `/Users/michaelroth/Documents/Code/groggy/src/core/hierarchical.rs` (NEW FILE)

**New types to create:**

```rust
// === ORIGINAL HIERARCHY TYPES ===
#[derive(Debug, Clone)]
pub struct SubgraphNode {
    pub node_id: NodeId,
    pub contained_subgraph: Option<Subgraph>,
    pub aggregation_metadata: HashMap<String, AttrValue>,
    pub original_node_count: usize,
    pub original_edge_count: usize,
}

#[derive(Debug, Clone)]
pub struct AggregationSpec {
    pub node_aggregations: HashMap<AttrName, AggregationFunction>,
    pub edge_aggregations: HashMap<AttrName, AggregationFunction>,
}

#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Sum, Average, Min, Max, Count, Concat, First, Last
}
```

**Methods to implement:**

```rust
// === ORIGINAL HIERARCHICAL OPERATIONS ===
impl Subgraph {
    pub fn to_subgraph_node(&self, aggregation: Option<&AggregationSpec>) -> GraphResult<SubgraphNode> {
        // Convert subgraph to a SubgraphNode with aggregated attributes
        // Store original subgraph structure for expansion
    }

    pub fn aggregate_attributes(&self, functions: &HashMap<String, AggregationFunction>) -> GraphResult<HashMap<String, AttrValue>> {
        // Apply aggregation functions to node/edge attributes
        // Return consolidated attribute map
    }

    pub fn contains_nodes(&self, node_ids: &[NodeId]) -> bool {
        // Check if all specified nodes are in this subgraph
        node_ids.iter().all(|&id| self.nodes.contains(&id))
    }

    pub fn contains_edges(&self, edge_ids: &[EdgeId]) -> bool {
        // Check if all specified edges are in this subgraph
        edge_ids.iter().all(|&id| self.edges.contains(&id))
    }

    pub fn add_to_graph(&self, target_graph: &mut Graph, options: Option<&HashMap<String, AttrValue>>) -> GraphResult<NodeId> {
        // Add this subgraph as a single node in target graph
        // Use aggregated attributes for the new node
        // Return the new node ID
    }
}

// === SUBGRAPH NODE OPERATIONS ===
impl SubgraphNode {
    pub fn expand(&self) -> GraphResult<Option<Subgraph>> {
        // Expand SubgraphNode back to contained subgraph
        // Restore original structure
        Ok(self.contained_subgraph.clone())
    }

    pub fn collapse(&self, aggregation: &AggregationSpec) -> GraphResult<SubgraphNode> {
        // Re-collapse with different aggregation specification
        if let Some(ref subgraph) = self.contained_subgraph {
            subgraph.to_subgraph_node(Some(aggregation))
        } else {
            Ok(self.clone())
        }
    }
}
```

---

## **Phase 2: FFI Traits Implementation (Weeks 4-5)**

### **Week 4: SubgraphOperations FFI Trait**
**File Location**: `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/subgraph_operations.rs`

**Implementation Strategy**: Thin wrappers that delegate to core implementations

```rust
use pyo3::prelude::*;
use crate::ffi::core::subgraph::PySubgraph;

pub trait SubgraphOperations {
    // === DELEGATION TO EXISTING OPTIMIZED CORE METHODS ===
    // ⚠️  DO NOT REIMPLEMENT - THESE ARE ALREADY OPTIMIZED ⚠️
    fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>> {
        // ✅ Delegate to EXISTING self.inner.connected_components() - OPTIMIZED
        // 🚫 DO NOT reimplement - existing version is 4.5x faster than NetworkX
    }
    
    fn bfs(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        // ✅ Delegate to EXISTING self.inner.bfs(start, TraversalOptions) - OPTIMIZED
        // 🚫 DO NOT reimplement - existing version uses TraversalEngine
    }
    
    fn dfs(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        // ✅ Delegate to EXISTING self.inner.dfs(start, TraversalOptions) - OPTIMIZED
        // 🚫 DO NOT reimplement
    }
    
    fn shortest_path(&self, py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>> {
        // ✅ Delegate to EXISTING PyGraphAnalytics.shortest_path() - OPTIMIZED
        // 🚫 DO NOT reimplement
    }
    
    fn degree(&self, py: Python, node_id: usize) -> PyResult<usize> {
        // ✅ Delegate to EXISTING Graph.degree() - OPTIMIZED
        // 🚫 DO NOT reimplement
    }
    
    // === DELEGATION TO NEW CORE METHODS (IMPLEMENT THESE) ===
    fn clustering_coefficient(&self, py: Python, node_id: Option<usize>) -> PyResult<f64> {
        // ✅ Delegate to NEW self.inner.clustering_coefficient(node_id)
    }
    
    fn merge_with(&self, py: Python, other: &dyn SubgraphOperations) -> PyResult<PySubgraph> {
        // ✅ Delegate to NEW self.inner.merge_with(other.inner)
    }
    
    // === CORE DATA ACCESS ===
    fn table(&self, py: Python) -> PyResult<PyObject>;
    fn nodes(&self) -> Vec<usize>;
    fn edges(&self) -> Vec<usize>;
    fn size(&self) -> usize;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn summary(&self) -> String;

    // === NODE ATTRIBUTES ===
    fn get_node_attribute(&self, py: Python, node_id: usize, attr_name: String) -> PyResult<PyObject>;
    fn set_node_attribute(&self, py: Python, node_id: usize, attr_name: String, value: PyObject) -> PyResult<()>;
    fn get_node_attributes(&self, py: Python, node_id: usize) -> PyResult<PyDict>;
    fn set_node_attributes(&self, py: Python, node_id: usize, attrs: &PyDict) -> PyResult<()>;
    fn has_node_attribute(&self, node_id: usize, attr_name: String) -> bool;
    fn list_node_attributes(&self) -> Vec<String>;

    // === EDGE ATTRIBUTES ===
    fn get_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String) -> PyResult<PyObject>;
    fn set_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String, value: PyObject) -> PyResult<()>;
    fn get_edge_attributes(&self, py: Python, edge_id: usize) -> PyResult<PyDict>;
    fn set_edge_attributes(&self, py: Python, edge_id: usize, attrs: &PyDict) -> PyResult<()>;
    fn has_edge_attribute(&self, edge_id: usize, attr_name: String) -> bool;
    fn list_edge_attributes(&self) -> Vec<String>;

    // === STRUCTURAL OPERATIONS ===
    fn neighbors(&self, py: Python, node_id: usize) -> PyResult<Vec<usize>>;
    fn contains_node(&self, node_id: usize) -> bool;
    fn contains_edge(&self, source: usize, target: usize) -> bool;
    fn has_edge(&self, source: usize, target: usize) -> bool;
    fn has_path(&self, py: Python, source: usize, target: usize) -> PyResult<bool>;

    // === FILTERING & QUERYING ===
    fn filter_nodes(&self, py: Python, query: String) -> PyResult<PySubgraph>;
    fn filter_edges(&self, py: Python, query: String) -> PyResult<PySubgraph>;
    fn query_nodes(&self, py: Python, filter: PyObject) -> PyResult<Vec<usize>>;
    fn query_edges(&self, py: Python, filter: PyObject) -> PyResult<Vec<usize>>;

    // === STATISTICS & METRICS ===
    fn transitivity(&self, py: Python) -> PyResult<f64>;
    fn density(&self) -> f64;
    fn is_connected(&self, py: Python) -> PyResult<bool>;

    // === EXPORT OPERATIONS ===
    fn to_networkx(&self, py: Python) -> PyResult<PyObject>;
    fn to_graph(&self, py: Python) -> PyResult<PyGraph>;
    fn copy(&self, py: Python) -> PyResult<PySubgraph>;

    // === DISPLAY & INFO ===
    fn display_info(&self, py: Python) -> PyResult<String>;
    fn get_info(&self) -> String;
}

// Implement for PySubgraph
impl SubgraphOperations for PySubgraph {
    // Direct delegation to self.inner methods
}

// Implement for PyNeighborhoodSubgraph  
impl SubgraphOperations for PyNeighborhoodSubgraph {
    // Delegate to underlying PySubgraph via trait calls
}
```

### **Week 5: HierarchicalOperations FFI Trait**
**File Location**: `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/hierarchical_operations.rs`

```rust
use pyo3::prelude::*;
use crate::ffi::core::subgraph_node::{PySubgraphNode, PyAggregationSpec};

pub trait HierarchicalOperations {
    // === ORIGINAL HIERARCHICAL OPERATIONS ===
    fn to_node(&self, py: Python) -> PyResult<PyObject>;
    fn to_subgraph_node(&self, py: Python, aggregation: Option<&PyAggregationSpec>) -> PyResult<PySubgraphNode>;
    fn add_to_graph(&self, py: Python, target_graph: &PyGraph, kwargs: Option<&PyDict>) -> PyResult<usize>;
    fn contains_nodes(&self, node_ids: Vec<usize>) -> bool;
    fn contains_edges(&self, edge_ids: Vec<usize>) -> bool;
    fn aggregate(&self, py: Python, functions: &PyDict) -> PyResult<PyDict>;
    fn collapse_to_node(&self, py: Python, aggregation: &PyAggregationSpec) -> PyResult<PySubgraphNode>;
    fn expand_from_node(&self, py: Python, subgraph_node: &PySubgraphNode) -> PyResult<PySubgraph>;
}
```

**New FFI types needed:**

```rust
// File: /Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/core/subgraph_node.rs
#[pyclass(name = "SubgraphNode")]
pub struct PySubgraphNode {
    inner: groggy::core::hierarchical::SubgraphNode,
    graph: Option<Py<PyGraph>>,
}

#[pyclass(name = "AggregationSpec")]  
pub struct PyAggregationSpec {
    inner: groggy::core::hierarchical::AggregationSpec,
}

#[pyclass(name = "AggregationFunction")]
pub struct PyAggregationFunction {
    inner: groggy::core::hierarchical::AggregationFunction,
}
```

---

## **Phase 3: Integration & Testing (Week 6)**

### **Week 6: Complete Integration**
1. **Update module structure** - Add new files to mod.rs
2. **Integration testing** - Test all trait methods work together
3. **Python API testing** - Verify all methods accessible from Python
4. **Performance benchmarking** - Compare with NetworkX equivalents
5. **Documentation** - Complete API docs and examples

---

## **IMPLEMENTATION FILES SUMMARY**

### **Core Rust Files (Need Creation/Updates):**
1. `/Users/michaelroth/Documents/Code/groggy/src/core/subgraph.rs` - Add 10 missing methods
2. `/Users/michaelroth/Documents/Code/groggy/src/core/hierarchical.rs` - NEW FILE - 22 hierarchy methods
3. `/Users/michaelroth/Documents/Code/groggy/src/core/mod.rs` - Add hierarchical module

### **FFI Files (Need Creation):**
1. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/mod.rs` - NEW
2. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/subgraph_operations.rs` - NEW - 45 methods
3. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/hierarchical_operations.rs` - NEW - 25 methods  
4. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/core/subgraph_node.rs` - NEW
5. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/mod.rs` - Update to include traits

### **Files to Archive (Replace with unified plan):**
- `documentation/planning/integrated_traits_hierarchy_plan.md` ❌
- `documentation/planning/methods_audit.md` ❌  
- `documentation/planning/shared_traits_migration_plan.md` ❌
- `documentation/planning/week2_enhancement_plan.md` ❌

---

## **SUCCESS METRICS**

### **Week 1 Success:** ✅ **COMPLETED**
- [x] All 10 missing SubgraphOperations methods implemented in core
- [x] Methods compile and pass basic tests
- [x] Performance benchmarks show no regression
- [x] **BONUS**: SimilarityMetric enum added and exported
- [x] **BONUS**: Comprehensive unit tests for structural metrics and set operations

### **Week 2-3 Success:**  
- [ ] Enhanced filtering and all hierarchical methods implemented
- [ ] Complete core functionality available

### **Week 4-5 Success:**
- [ ] FFI traits provide unified API across all subgraph types
- [ ] All methods accessible from Python
- [ ] Performance matches or exceeds existing optimized methods

### **Week 6 Success:**
- [ ] Complete integration with comprehensive test coverage
- [ ] Documentation and examples complete
- [ ] Production-ready hierarchical graph operations

**Total New/Updated Files: 8 files**
**Total New Methods: ~35 methods in core + ~70 methods in FFI**

This represents the complete unified implementation plan that consolidates all previous planning documents into one comprehensive guide.