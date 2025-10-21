# Complete SubgraphOperations & HierarchicalOperations Implementation Plan

## EXECUTIVE SUMMARY

**Total Methods Needed**: 70 methods (45 SubgraphOperations + 25 HierarchicalOperations)

### **🚫 DO NOT REIMPLEMENT (Already Optimized):**
- `connected_components()` - 4.5x faster than NetworkX - ALREADY OPTIMIZED ✅
- `bfs()`, `dfs()` - Use TraversalEngine - ALREADY OPTIMIZED ✅  
- `shortest_path()`, `degree()` - Core Graph methods - ALREADY OPTIMIZED ✅
- **~35 methods** can delegate to existing optimized implementations

### **✅ NEED IMPLEMENTATION (Missing from Core):**
- **~10 SubgraphOperations**: structural metrics, set operations
- **~25 HierarchicalOperations**: meta-nodes, hierarchy navigation, serialization
- **Total: ~35 NEW methods** need core implementation

### **🎯 Implementation Strategy:**
- **FFI Traits**: Thin wrappers that delegate to core (existing + new)
- **Core Methods**: Only implement what's missing, preserve optimizations

---

## DETAILED METHOD AUDIT

### Method Categories We Need

### 1. Core Data Access (8 methods)
- `table(py: Python) -> PyResult<PyObject>`
- `nodes() -> Vec<usize>`  
- `edges() -> Vec<usize>`
- `size() -> usize`
- `node_count() -> usize`
- `edge_count() -> usize`
- `is_empty() -> bool`
- `summary() -> String`

### 2. Node Attributes (6 methods)
- `get_node_attribute(node_id: usize, attr_name: String) -> PyResult<PyObject>`
- `set_node_attribute(node_id: usize, attr_name: String, value: PyObject) -> PyResult<()>`
- `get_node_attributes(node_id: usize) -> PyResult<PyDict>`
- `set_node_attributes(node_id: usize, attrs: PyDict) -> PyResult<()>`
- `has_node_attribute(node_id: usize, attr_name: String) -> bool`
- `list_node_attributes() -> Vec<String>`

### 3. Edge Attributes (6 methods)  
- `get_edge_attribute(edge_id: usize, attr_name: String) -> PyResult<PyObject>`
- `set_edge_attribute(edge_id: usize, attr_name: String, value: PyObject) -> PyResult<()>`
- `get_edge_attributes(edge_id: usize) -> PyResult<PyDict>`
- `set_edge_attributes(edge_id: usize, attrs: PyDict) -> PyResult<()>`
- `has_edge_attribute(edge_id: usize, attr_name: String) -> bool`
- `list_edge_attributes() -> Vec<String>`

### 4. Graph Algorithms (6 methods)
- `connected_components(py: Python) -> PyResult<Vec<PySubgraph>>`
- `bfs(py: Python, start_node: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>`
- `dfs(py: Python, start_node: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>`
- `shortest_path(py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>>`
- `has_path(py: Python, source: usize, target: usize) -> PyResult<bool>`
- `degree(py: Python, node_id: usize) -> PyResult<usize>`

### 5. Structural Operations (4 methods)
- `neighbors(py: Python, node_id: usize) -> PyResult<Vec<usize>>`
- `contains_node(node_id: usize) -> bool`
- `contains_edge(source: usize, target: usize) -> bool`
- `has_edge(source: usize, target: usize) -> bool`

### 6. Filtering & Querying (4 methods)
- `filter_nodes(py: Python, query: String) -> PyResult<PySubgraph>`
- `filter_edges(py: Python, query: String) -> PyResult<PySubgraph>`
- `query_nodes(py: Python, filter: PyObject) -> PyResult<Vec<usize>>`
- `query_edges(py: Python, filter: PyObject) -> PyResult<Vec<usize>>`

### 7. Statistics & Metrics (4 methods)
- `clustering_coefficient(py: Python, node_id: Option<usize>) -> PyResult<f64>`
- `transitivity(py: Python) -> PyResult<f64>`
- `density() -> f64`
- `is_connected(py: Python) -> PyResult<bool>`

### 8. Export Operations (3 methods)
- `to_networkx(py: Python) -> PyResult<PyObject>`
- `to_graph(py: Python) -> PyResult<PyGraph>`
- `copy(py: Python) -> PyResult<PySubgraph>`

### 9. Display & Info (4 methods)
- `display_info(py: Python) -> PyResult<String>`
- `get_info() -> String`
- `__str__() -> String`
- `__repr__() -> String`

**Total SubgraphOperations: 45 methods**

---

## HierarchicalOperations Methods (NEW - need implementation)

### 1. Meta-node Creation (5 methods)
- `to_meta_node(py: Python, parent_graph: &PyGraph) -> PyResult<PySubgraphNode>`
- `collapse_to_node(py: Python, aggregation: AggregationSpec) -> PyResult<PySubgraphNode>`
- `aggregate(py: Python, functions: PyDict) -> PyResult<PyDict>`
- `create_meta_node_with_attrs(py: Python, attrs: PyDict) -> PyResult<PySubgraphNode>`
- `batch_collapse(py: Python, subgraphs: Vec<Box<dyn SubgraphOperations>>) -> PyResult<Vec<PySubgraphNode>>`

### 2. Hierarchy Navigation (5 methods)
- `expand_from_node(py: Python, meta_node: &PySubgraphNode) -> PyResult<Self>`
- `get_hierarchy_level() -> usize`
- `get_parent_meta_node() -> Option<PySubgraphNode>`
- `list_child_subgraphs() -> Vec<Box<dyn SubgraphOperations>>`
- `hierarchy_path() -> Vec<usize>`

### 3. Subgraph Relationships (5 methods)
- `merge_with(py: Python, other: &dyn SubgraphOperations) -> PyResult<PySubgraph>`
- `intersect_with(py: Python, other: &dyn SubgraphOperations) -> PyResult<PySubgraph>`
- `subtract_from(py: Python, other: &dyn SubgraphOperations) -> PyResult<PySubgraph>`
- `find_overlaps(py: Python, others: Vec<&dyn SubgraphOperations>) -> PyResult<Vec<PySubgraph>>`
- `calculate_similarity(py: Python, other: &dyn SubgraphOperations) -> PyResult<f64>`

### 4. Edge Handling in Hierarchy (5 methods)
- `connect_meta_nodes(py: Python, source: &PySubgraphNode, target: &PySubgraphNode) -> PyResult<usize>`
- `aggregate_edge_weights(py: Python, aggregation_func: String) -> PyResult<PyDict>`
- `preserve_boundary_edges(py: Python) -> PyResult<Vec<(usize, usize)>>`
- `map_internal_edges(py: Python) -> PyResult<PyDict>`
- `create_hierarchy_edges(py: Python, level: usize) -> PyResult<Vec<(usize, usize)>>`

### 5. Serialization & Persistence (5 methods)
- `serialize_hierarchy(py: Python) -> PyResult<PyObject>`
- `deserialize_hierarchy(py: Python, data: PyObject) -> PyResult<Self>`
- `export_hierarchy_json(py: Python) -> PyResult<String>`
- `import_hierarchy_json(py: Python, json_str: String) -> PyResult<Self>`
- `save_hierarchy_state(py: Python, filepath: String) -> PyResult<()>`

**Total HierarchicalOperations: 25 methods**

---

## AUDIT RESULTS

### ✅ Already Exist in Core - DO NOT REIMPLEMENT:
**RustSubgraph methods (OPTIMIZED - USE AS-IS):**
- `connected_components()` ✅ - Already optimized, just wrap in FFI
- `bfs()` ✅ - Already optimized, just wrap in FFI
- `dfs()` ✅ - Already optimized, just wrap in FFI

**Graph methods (OPTIMIZED - USE AS-IS):**
- `degree()` ✅ - Already optimized, just wrap in FFI  
- `shortest_path()` ✅ - Already optimized, just wrap in FFI
- `neighbors()` ✅ - Available via get_neighbors(), just wrap in FFI

**PyGraphAnalytics methods (OPTIMIZED - USE AS-IS):**
- All existing analytics methods have performance optimizations
- FFI traits should delegate to these, NOT reimplement

### ❌ Missing from Core - NEED IMPLEMENTATION:

#### **Missing SubgraphOperations (need core implementation):**
1. **Structural metrics**: `clustering_coefficient()`, `transitivity()`, `density()`, `is_connected()`  
2. **Advanced filtering**: Better `filter_nodes()`, `filter_edges()` with complex queries
3. **Subgraph operations**: `merge_with()`, `intersect_with()`, `subtract_from()`

#### **Missing HierarchicalOperations (ALL need core implementation):**
1. **Meta-node creation**: All 5 methods - completely new functionality
2. **Hierarchy navigation**: All 5 methods - completely new functionality  
3. **Subgraph relationships**: All 5 methods - completely new functionality
4. **Edge handling in hierarchy**: All 5 methods - completely new functionality
5. **Serialization & persistence**: All 5 methods - completely new functionality

**Total methods needing core implementation: ~35 methods**

---

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
// === NEW TYPES FOR HIERARCHY ===
#[derive(Debug, Clone)]
pub struct MetaNode {
    pub node_id: NodeId,
    pub contained_subgraph: Option<Subgraph>,
    pub aggregation_metadata: HashMap<String, AttrValue>,
    pub hierarchy_level: usize,
    pub parent_id: Option<NodeId>,
    pub child_nodes: Vec<NodeId>,
}

#[derive(Debug, Clone)]
pub struct AggregationSpec {
    pub node_aggregations: HashMap<AttrName, AggregationFunction>,
    pub edge_aggregations: HashMap<AttrName, AggregationFunction>,
    pub preserve_topology: bool,
    pub boundary_edge_handling: BoundaryEdgeMode,
}

#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Sum, Average, Min, Max, Count, Concat, First, Last
}

#[derive(Debug, Clone)]
pub enum BoundaryEdgeMode {
    Preserve,    // Keep edges connecting to external nodes
    Aggregate,   // Combine into single edge with aggregated weights
    Remove,      // Delete boundary edges
}
```

**Methods to implement:**

```rust
// === META-NODE CREATION ===
impl Subgraph {
    pub fn to_meta_node(&self, aggregation: &AggregationSpec) -> GraphResult<MetaNode> {
        // Convert entire subgraph to a single meta-node
        // Aggregate all internal node/edge attributes
        // Handle boundary edges according to spec
    }

    pub fn collapse_to_node(&self, node_id: NodeId, aggregation: &AggregationSpec) -> GraphResult<MetaNode> {
        // Collapse subgraph to specific node ID in parent graph
        // Preserve original graph structure with replacement
    }

    pub fn aggregate_attributes(&self, spec: &AggregationSpec) -> GraphResult<HashMap<String, AttrValue>> {
        // Apply aggregation functions to all attributes
        // Return consolidated attribute map for meta-node
    }

    pub fn batch_collapse(&self, subgraphs: Vec<Subgraph>, specs: Vec<AggregationSpec>) -> GraphResult<Vec<MetaNode>> {
        // Efficiently collapse multiple subgraphs simultaneously
        // Optimize for bulk operations
    }
}

// === HIERARCHY NAVIGATION ===  
impl MetaNode {
    pub fn expand(&self) -> GraphResult<Option<Subgraph>> {
        // Expand meta-node back to contained subgraph
        // Restore original node/edge structure
    }

    pub fn get_hierarchy_level(&self) -> usize {
        // Return depth level in hierarchy tree
    }

    pub fn get_parent_meta_node(&self) -> Option<&MetaNode> {
        // Navigate up hierarchy tree
    }

    pub fn list_child_subgraphs(&self) -> Vec<&Subgraph> {
        // Navigate down hierarchy tree
    }

    pub fn hierarchy_path(&self) -> Vec<NodeId> {
        // Return path from root to this meta-node
    }
}

// === HIERARCHY EDGE HANDLING ===
impl Subgraph {
    pub fn connect_meta_nodes(&self, source: &MetaNode, target: &MetaNode, weight_aggregation: AggregationFunction) -> GraphResult<EdgeId> {
        // Create edges between meta-nodes based on internal connectivity
        // Aggregate edge weights according to specified function
    }

    pub fn aggregate_edge_weights(&self, aggregation: AggregationFunction) -> GraphResult<HashMap<EdgeId, f64>> {
        // Aggregate edge weights within subgraph for meta-node creation
    }

    pub fn preserve_boundary_edges(&self) -> GraphResult<Vec<(NodeId, NodeId)>> {
        // Identify and preserve edges connecting to external nodes
        // Critical for maintaining graph connectivity during hierarchy creation
    }

    pub fn map_internal_edges(&self) -> GraphResult<HashMap<EdgeId, (NodeId, NodeId)>> {
        // Create mapping of internal edges for hierarchy serialization
    }
}

// === HIERARCHY SERIALIZATION ===
impl MetaNode {
    pub fn serialize_hierarchy(&self) -> GraphResult<serde_json::Value> {
        // Serialize entire hierarchy tree to JSON
        // Include all meta-nodes, contained subgraphs, and relationships
    }

    pub fn deserialize_hierarchy(data: &serde_json::Value) -> GraphResult<Self> {
        // Reconstruct hierarchy tree from JSON
        // Restore all meta-node relationships and contained subgraphs
    }

    pub fn export_hierarchy_json(&self) -> GraphResult<String> {
        // Export hierarchy to human-readable JSON format
        // Include metadata about aggregation functions and hierarchy structure
    }

    pub fn save_hierarchy_state(&self, filepath: &str) -> GraphResult<()> {
        // Persist hierarchy state to disk
        // Support incremental saves and lazy loading
    }
}
```

---

## **Phase 2: FFI Traits Implementation (Weeks 4-5)**

### **Week 4: SubgraphOperations FFI Trait**
**File Location**: `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/subgraph_operations.rs`

**Implementation Strategy**: Thin wrappers that delegate to core implementations

```rust
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
    
    // ... remaining methods as thin wrappers
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
pub trait HierarchicalOperations {
    fn to_meta_node(&self, py: Python, aggregation: PyAggregationSpec) -> PyResult<PyMetaNode> {
        // Convert PyAggregationSpec to core AggregationSpec
        // Delegate to self.inner.to_meta_node(spec)
        // Wrap result in PyMetaNode
    }
    
    fn collapse_to_node(&self, py: Python, node_id: usize, aggregation: PyAggregationSpec) -> PyResult<PyMetaNode> {
        // Delegate to self.inner.collapse_to_node(node_id, spec)
    }
    
    // ... all 25 hierarchical methods as thin wrappers
}

// New FFI types needed:
#[pyclass(name = "MetaNode")]
pub struct PyMetaNode {
    inner: MetaNode,
    graph: Option<Py<PyGraph>>,
}

#[pyclass(name = "AggregationSpec")]  
pub struct PyAggregationSpec {
    inner: AggregationSpec,
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
1. `/Users/michaelroth/Documents/Code/groggy/src/core/subgraph.rs` - Add missing methods
2. `/Users/michaelroth/Documents/Code/groggy/src/core/hierarchical.rs` - NEW FILE for hierarchy
3. `/Users/michaelroth/Documents/Code/groggy/src/core/mod.rs` - Add hierarchical module

### **FFI Files (Need Creation):**
1. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/mod.rs` - NEW
2. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/subgraph_operations.rs` - NEW
3. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/traits/hierarchical_operations.rs` - NEW
4. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/core/meta_node.rs` - NEW
5. `/Users/michaelroth/Documents/Code/groggy/python-groggy/src/ffi/mod.rs` - Update to include traits

### **Total New/Updated Files: 8 files**
### **Total New Methods: ~35 methods in core + ~70 methods in FFI**

This represents a significant but well-structured implementation that will provide comprehensive subgraph operations and hierarchical graph capabilities.