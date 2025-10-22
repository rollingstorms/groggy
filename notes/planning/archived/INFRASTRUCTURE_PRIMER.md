# Groggy Infrastructure Primer: Understanding the Three-Tier Architecture

## Executive Summary

Groggy employs a sophisticated three-tier architecture that separates concerns across **Core (Rust)**, **FFI (Python Bindings)**, and **API (High-level Interface)** layers. This document serves as a comprehensive primer for understanding how these tiers interact, why they exist, and how to navigate the complexity they introduce.

## Table of Contents

1. [The Three-Tier Challenge](#the-three-tier-challenge)
2. [Architecture Overview](#architecture-overview)
3. [Core Layer (Rust Foundation)](#core-layer-rust-foundation)
4. [FFI Layer (Language Bridge)](#ffi-layer-language-bridge)
5. [API Layer (User Interface)](#api-layer-user-interface)
6. [Data Flow Patterns](#data-flow-patterns)
7. [Duplication Patterns and Solutions](#duplication-patterns-and-solutions)
8. [Performance Implications](#performance-implications)
9. [Maintenance Strategies](#maintenance-strategies)
10. [Future Modularization Path](#future-modularization-path)

---

## The Three-Tier Challenge

### Current Pain Points

**Method Duplication**: The same functionality appears across all three tiers:
- `src/core/subgraph.rs` → Core graph algorithms
- `python-groggy/src/ffi/core/subgraph.rs` → Python bindings for subgraphs  
- `python-groggy/src/ffi/api/graph.rs` → High-level Python graph interface

**Complexity Multiplication**: Each new feature requires implementation at multiple levels:
1. Core logic in Rust
2. FFI wrapper with memory management
3. Pythonic API with error handling
4. Documentation at each layer

**Inconsistent Interfaces**: Different paradigms at each tier create cognitive overhead:
- Core: Rust ownership, Result types, explicit memory management
- FFI: PyO3 bindings, Python reference counting, exception propagation
- API: Pythonic patterns, duck typing, convenience methods

### Strategic Vision

This primer documents a path toward **modular core traits** that can be specialized into focused structs like:
- `ComponentSubgraph` - Connected component analysis
- `NeighborhoodSubgraph` - Local neighborhood sampling
- `TemporalSubgraph` - Time-based graph slices

These specialized components would share common traits but implement domain-specific optimizations, reducing duplication while maintaining performance.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│                    Python User Space                   │
│                   (import groggy)                      │
└─────────────────┬───────────────────────────────────────┘
                  │ Python API Calls
                  ▼
┌─────────────────────────────────────────────────────────┐
│                 API LAYER (Tier 3)                     │
│          High-Level Python Interface                   │
│         python-groggy/python/groggy/                   │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Graph     │  │ Enhanced    │  │  NetworkX       │ │
│  │   Class     │  │  Query      │  │ Compatibility   │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │ FFI Calls
                  ▼
┌─────────────────────────────────────────────────────────┐
│                 FFI LAYER (Tier 2)                     │
│             Python-Rust Bridge                         │
│          python-groggy/src/ffi/                        │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  PyGraph    │  │ Core Types  │  │   Memory        │ │
│  │  Wrappers   │  │   (PyO3)    │  │ Management      │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │ Direct Calls
                  ▼
┌─────────────────────────────────────────────────────────┐
│                 CORE LAYER (Tier 1)                    │
│              Pure Rust Engine                          │
│                  src/                                  │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │    Pool     │  │   History   │  │   Algorithms    │ │
│  │ (Storage)   │  │ (Version)   │  │ (Traversal)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Space     │  │    Query    │  │   Subgraph      │ │
│  │  (Active)   │  │ (Filtering) │  │   (Views)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Information Flow

1. **Downward**: User requests flow from API → FFI → Core
2. **Upward**: Data and errors propagate from Core → FFI → API  
3. **Horizontal**: Each layer has internal coordination between components

---

## Core Layer (Rust Foundation)

### Location: `src/`

The Core layer contains the pure Rust implementation with no Python dependencies. This is the performance-critical foundation that implements all graph algorithms, data structures, and business logic.

### Key Components

#### Storage System
```rust
// src/core/pool.rs
pub struct GraphPool {
    node_attributes: HashMap<AttrName, AttributeColumn>,
    edge_attributes: HashMap<AttrName, AttributeColumn>,
    edges: HashMap<EdgeId, (NodeId, NodeId)>,
}

// Columnar storage for cache efficiency
pub struct AttributeColumn {
    values: Vec<AttrValue>,      // Append-only for performance
    next_index: usize,
}
```

#### Version Control System
```rust  
// src/core/history.rs
pub struct HistoryForest {
    states: HashMap<StateId, Arc<StateObject>>,
    deltas: HashMap<[u8; 32], Arc<DeltaObject>>,
}

// Git-like branching
// src/core/ref_manager.rs
pub struct RefManager {
    branches: HashMap<BranchName, Branch>,
    current_branch: BranchName,
}
```

#### Analysis Engine
```rust
// src/core/traversal.rs  
pub struct TraversalEngine {
    adjacency_cache: AdjacencyCache,
    state_pool: TraversalStatePool,
}

// src/core/subgraph.rs
pub struct Subgraph {
    graph: Rc<RefCell<Graph>>,
    nodes: HashSet<NodeId>,
    edges: HashSet<EdgeId>,
}
```

### Design Principles

1. **Zero Dependencies on Python**: Pure Rust for maximum performance
2. **Trait-Based Design**: Common interfaces enable polymorphism
3. **Memory Safety**: Leverage Rust's ownership system
4. **Columnar Storage**: Optimize for bulk analytical operations

---

## FFI Layer (Language Bridge)

### Location: `python-groggy/src/ffi/`

The FFI layer provides Python bindings using PyO3, managing the complex boundary between Python and Rust memory models.

### Architecture Pattern

Each Core component gets a corresponding FFI wrapper:

```rust
// python-groggy/src/ffi/core/subgraph.rs
#[pyclass]
pub struct PySubgraph {
    pub inner: groggy::core::subgraph::Subgraph,
}

#[pymethods]  
impl PySubgraph {
    fn node_count(&self, py: Python) -> PyResult<usize> {
        py.allow_threads(|| {
            Ok(self.inner.node_count())
        })
    }
    
    fn filter_nodes(&self, py: Python, filter: &PyNodeFilter) -> PyResult<PySubgraph> {
        py.allow_threads(|| {
            let core_result = self.inner.filter_nodes(&filter.to_core())?;
            Ok(PySubgraph { inner: core_result })
        })
    }
}
```

### Key Responsibilities

#### Memory Management
```rust
// Automatic conversion between Python and Rust types
impl FromPyObject<'_> for PyAttrValue {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        if let Ok(s) = obj.extract::<String>() {
            Ok(PyAttrValue::Text(s))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(PyAttrValue::Int(i))  
        } // ... other types
    }
}
```

#### Error Propagation
```rust
// Convert Rust errors to Python exceptions
impl From<groggy::GraphError> for PyErr {
    fn from(err: groggy::GraphError) -> Self {
        match err {
            GraphError::NodeNotFound(id) => {
                PyKeyError::new_err(format!("Node {} not found", id))
            },
            GraphError::InvalidState(msg) => {
                PyRuntimeError::new_err(msg)
            },
        }
    }
}
```

#### Performance Optimization
```rust
// Release GIL for CPU-intensive operations
#[pymethods]
impl PyGraph {
    fn connected_components(&self, py: Python) -> PyResult<Vec<Vec<u64>>> {
        py.allow_threads(|| {
            let components = self.inner.borrow().connected_components()?;
            Ok(components)
        })
    }
}
```

---

## API Layer (User Interface)

### Location: `python-groggy/python/groggy/`

The API layer provides a Pythonic interface that hides FFI complexity and adds convenience methods, enhanced querying, and ecosystem integration.

### Key Components

#### Enhanced Graph Interface
```python
# python-groggy/python/groggy/graph.py
class Graph:
    def __init__(self, directed=True):
        self._graph = _groggy.PyGraph(directed)
        
    def add_node(self, **attributes):
        """Add a node with optional attributes."""
        node_id = self._graph.add_node()
        for key, value in attributes.items():
            self._graph.set_node_attr(node_id, key, value)
        return node_id
        
    def subgraph(self, nodes=None, edges=None):
        """Create a subgraph view."""
        if nodes is not None:
            return SubgraphView(self._graph.subgraph_from_nodes(nodes))
        elif edges is not None:  
            return SubgraphView(self._graph.subgraph_from_edges(edges))
```

#### NetworkX Compatibility
```python
# python-groggy/python/groggy/networkx_compat.py
def to_networkx(groggy_graph):
    """Convert Groggy graph to NetworkX."""
    import networkx as nx
    
    if groggy_graph.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()
        
    # Add nodes with attributes
    for node in groggy_graph.nodes():
        attrs = groggy_graph.get_node_attrs(node)
        G.add_node(node, **attrs)
        
    return G
```

#### Enhanced Query Interface
```python
# python-groggy/python/groggy/enhanced_query.py
class GraphQuery:
    def __init__(self, graph):
        self.graph = graph
        self.filters = []
        
    def filter_nodes(self, **conditions):
        """Filter nodes by attribute conditions."""
        for attr, value in conditions.items():
            if isinstance(value, (list, tuple)):
                filter_obj = NodeFilter.attribute_in(attr, value)
            else:
                filter_obj = NodeFilter.attribute_equals(attr, value)
            self.filters.append(filter_obj)
        return self
        
    def execute(self):
        """Execute the query and return results."""
        result_nodes = self.graph.nodes()
        for filter_obj in self.filters:
            result_nodes = self.graph.filter_nodes(result_nodes, filter_obj)
        return result_nodes
```

---

## Data Flow Patterns

### Pattern 1: Simple Query
```text
User: graph.node_count()
  ↓
API: graph.py → _graph.node_count()  
  ↓
FFI: PyGraph.node_count() → inner.node_count()
  ↓  
Core: Graph.node_count() → space.active_nodes.len()
  ↑
Returns: usize → PyResult<usize> → int
```

### Pattern 2: Complex Operation
```text
User: graph.connected_components()
  ↓
API: Enhanced error handling, result formatting
  ↓  
FFI: Memory management, GIL release, error conversion
  ↓
Core: Algorithm implementation, optimization
  ↑
Returns: Vec<Vec<NodeId>> → PyResult<Vec<Vec<u64>>> → List[List[int]]
```

### Pattern 3: Attribute Access
```text  
User: node["name"] = "Alice"
  ↓
API: Pythonic attribute syntax (__setitem__)
  ↓
FFI: Type conversion (str → PyAttrValue → AttrValue)
  ↓  
Core: Columnar storage update
```

---

## Duplication Patterns and Solutions

### Current Duplication Examples

#### Subgraph Creation
**Core**: `src/core/subgraph.rs:45-67`
```rust
impl Subgraph {
    pub fn from_nodes(graph: Rc<RefCell<Graph>>, nodes: HashSet<NodeId>) -> GraphResult<Self> {
        // Validation logic
        // Edge induction logic  
        // Subgraph construction
    }
}
```

**FFI**: `python-groggy/src/ffi/core/subgraph.rs:89-112`
```rust
#[pymethods]
impl PySubgraph {
    #[new]
    fn new(graph: &PyGraph, nodes: Vec<u64>) -> PyResult<Self> {
        // Convert Vec<u64> to HashSet<NodeId>
        // Call core method
        // Wrap result in PySubgraph
    }
}
```

**API**: `python-groggy/python/groggy/graph.py:234-251`
```python
def subgraph(self, nodes=None):
    """Create subgraph with enhanced error messages and validation."""
    if nodes is None:
        raise ValueError("Must specify nodes")
    if not isinstance(nodes, (list, set)):
        nodes = list(nodes)  
    return SubgraphView(self._graph.subgraph_from_nodes(nodes))
```

### Proposed Modular Solutions

#### Trait-Based Core Architecture
```rust
// src/core/traits.rs
pub trait GraphComponent {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    fn contains_node(&self, node: NodeId) -> bool;
}

pub trait FilterableComponent: GraphComponent {
    fn filter_nodes<F>(&self, predicate: F) -> GraphResult<Self>
    where F: Fn(NodeId) -> bool, Self: Sized;
}

pub trait AnalyzableComponent: GraphComponent {  
    fn connected_components(&self) -> GraphResult<Vec<Vec<NodeId>>>;
    fn degree_distribution(&self) -> GraphResult<HashMap<usize, usize>>;
}
```

#### Specialized Implementations
```rust
// src/core/components/neighborhood.rs
pub struct NeighborhoodSubgraph {
    base: SubgraphCore,
    center_nodes: HashSet<NodeId>,
    max_depth: usize,
    sampling_rate: f64,
}

impl GraphComponent for NeighborhoodSubgraph { /* common methods */ }
impl FilterableComponent for NeighborhoodSubgraph { /* filtering logic */ }
impl NeighborhoodOperations for NeighborhoodSubgraph { /* specialized methods */ }

// src/core/components/connected.rs  
pub struct ComponentSubgraph {
    base: SubgraphCore,
    component_id: usize,
    bridge_edges: HashSet<EdgeId>,
}

impl GraphComponent for ComponentSubgraph { /* common methods */ }
impl ConnectedOperations for ComponentSubgraph { /* component-specific methods */ }
```

#### Generated FFI Wrappers
```rust
// Macro to generate FFI wrappers automatically
macro_rules! generate_py_wrapper {
    ($core_type:ty, $py_name:ident) => {
        #[pyclass]
        pub struct $py_name {
            pub inner: $core_type,
        }
        
        #[pymethods]
        impl $py_name {
            // Auto-generate common methods
            fn node_count(&self) -> usize { self.inner.node_count() }
            fn edge_count(&self) -> usize { self.inner.edge_count() }
            // ... other trait methods
        }
    };
}

generate_py_wrapper!(NeighborhoodSubgraph, PyNeighborhoodSubgraph);
generate_py_wrapper!(ComponentSubgraph, PyComponentSubgraph);
```

---

## Performance Implications

### Layer Overhead Analysis

#### Memory Overhead
- **Core**: Zero overhead (pure Rust)
- **FFI**: ~8-16 bytes per wrapped object (PyO3 overhead)  
- **API**: ~64-128 bytes per Python object (interpreter overhead)

#### Call Overhead  
- **Core → Core**: ~1-5ns (inlined function calls)
- **FFI → Core**: ~50-100ns (PyO3 boundary crossing)
- **API → FFI**: ~200-500ns (Python method dispatch)

#### Optimization Strategies

1. **Batch Operations**: Minimize boundary crossings
   ```python
   # Inefficient: Multiple FFI calls
   for node in nodes:
       graph.set_node_attr(node, "processed", True)
       
   # Efficient: Single batch call  
   graph.set_node_attrs_bulk("processed", 
       [(node, True) for node in nodes])
   ```

2. **GIL Management**: Release during CPU-intensive operations
   ```rust
   py.allow_threads(|| {
       self.inner.connected_components() // Long-running Rust operation
   })
   ```

3. **Memory Pooling**: Reuse allocations across operations
   ```rust
   pub struct TraversalEngine {
       state_pool: TraversalStatePool, // Reusable buffers
   }
   ```

---

## Maintenance Strategies

### Code Organization

#### Module Mapping
Maintain strict correspondence between layers:
```text
src/core/subgraph.rs
  ↕ (1:1 mapping)
python-groggy/src/ffi/core/subgraph.rs  
  ↕ (convenience layer)
python-groggy/python/groggy/subgraph.py
```

#### Testing Strategy
```text
Unit Tests:    src/                     (Rust unit tests)
Integration:   python-groggy/src/ffi/   (Rust-Python integration)  
End-to-End:    python-groggy/python/    (Python user scenarios)
Performance:   benchmarks/              (Cross-layer benchmarks)
```

#### Documentation Synchronization
- **Core**: Rust doc comments (`///`)
- **FFI**: PyO3 doc strings (`#[doc = "..."]`)  
- **API**: Python docstrings (`"""..."""`)
- **Cross-Reference**: Link between layers in documentation

### Change Management

#### Feature Addition Workflow
1. **Design**: Define trait interfaces in Core
2. **Implement**: Add Core functionality with comprehensive tests
3. **Bind**: Create FFI wrapper with memory safety
4. **Enhance**: Add API convenience methods and documentation
5. **Validate**: End-to-end integration testing

#### Deprecation Strategy  
1. **Core**: Mark with `#[deprecated]` and migration path
2. **FFI**: Add warnings in PyO3 methods
3. **API**: Use Python `warnings.warn()` for user notification
4. **Remove**: Coordinated removal across all layers

---

## Future Modularization Path

### Phase 1: Trait Extraction
Extract common interfaces from existing components:
```rust
// Current: Monolithic subgraph
pub struct Subgraph { /* everything */ }

// Future: Trait-based composition
pub trait SubgraphCore { /* essential methods */ }
pub trait SubgraphFiltering { /* query methods */ }
pub trait SubgraphAnalytics { /* algorithm methods */ }
```

### Phase 2: Specialized Components  
Create focused implementations:
```rust
pub struct ComponentSubgraph: SubgraphCore + ConnectedAnalysis;
pub struct NeighborhoodSubgraph: SubgraphCore + NeighborhoodSampling;
pub struct TemporalSubgraph: SubgraphCore + TimeSeriesOperations;
```

### Phase 3: Code Generation
Automate FFI layer generation:
```rust
// Derive macro for automatic FFI wrapper generation
#[derive(PyWrapper)]
pub struct NeighborhoodSubgraph { /* ... */ }

// Generates PyNeighborhoodSubgraph automatically
```

### Phase 4: Dynamic Composition
Allow runtime composition of capabilities:
```python  
# User selects capabilities needed
subgraph = graph.subgraph(nodes)
    .with_connectivity_analysis()
    .with_neighborhood_sampling(depth=2)
    .with_temporal_filtering()
```

---

## Conclusion

The three-tier architecture provides essential separation of concerns but introduces significant complexity. Understanding the data flow patterns, performance implications, and maintenance strategies is crucial for effective development.

The proposed modularization path offers a way to reduce duplication while maintaining performance and adding flexibility. By extracting common traits and creating specialized implementations, we can achieve better code organization without sacrificing the benefits of the layered architecture.

The key insight is that the current duplication is not inherently bad—it serves important purposes (performance, safety, usability). The goal is to reduce accidental complexity while preserving the essential complexity that delivers value to users.

---

## Next Steps

1. **Review with Team**: Validate understanding and gather feedback
2. **Prototype Traits**: Experiment with trait-based design in a branch
3. **Measure Performance**: Benchmark current vs. proposed approaches  
4. **Plan Migration**: Define incremental steps for refactoring
5. **Document Patterns**: Create templates for future component development