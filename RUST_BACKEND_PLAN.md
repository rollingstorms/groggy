# Phase 2: Rust Backend Implementation Plan

## Architecture Overview

```
gli/
├── gli/                    # Python package
│   ├── __init__.py        # Public API
│   ├── graph.py           # Python Graph wrapper
│   ├── store.py           # Python GraphStore wrapper
│   └── _core.pyi          # Type stubs for Rust module
├── src/                   # Rust source
│   ├── lib.rs             # PyO3 bindings
│   ├── graph/
│   │   ├── mod.rs         # Graph module
│   │   ├── core.rs        # Core graph structure
│   │   ├── operations.rs  # Graph operations
│   │   └── algorithms.rs  # Graph algorithms
│   ├── storage/
│   │   ├── mod.rs         # Storage module
│   │   ├── content_pool.rs # Content addressing
│   │   └── state_store.rs  # State management
│   └── utils/
│       ├── mod.rs
│       └── hash.rs        # Fast hashing utilities
├── Cargo.toml             # Rust dependencies
├── pyproject.toml         # Python build config with maturin
└── benchmarks/            # Performance benchmarks
```

## Implementation Steps

### Step 1: Setup Rust Environment

```toml
# Cargo.toml
[package]
name = "gli-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "gli_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
petgraph = "0.6"
xxhash-rust = { version = "0.8", features = ["xxh3"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rayon = "1.7"  # For parallel operations
arrow = "50.0"  # Columnar data layout
```

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "gli"
requires-python = ">=3.8"
dynamic = ["version"]

[tool.maturin]
python-source = "python"
module-name = "gli._core"
```

### Step 2: Core Graph Structure in Rust

```rust
// src/graph/core.rs
use pyo3::prelude::*;
use petgraph::{Graph as PetGraph, Directed, NodeIndex, EdgeIndex};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: String,
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub attributes: HashMap<String, serde_json::Value>,
}

#[pyclass]
pub struct FastGraph {
    graph: PetGraph<NodeData, EdgeData, Directed>,
    node_id_to_index: HashMap<String, NodeIndex>,
    node_index_to_id: HashMap<NodeIndex, String>,
}

#[pymethods]
impl FastGraph {
    #[new]
    fn new() -> Self {
        Self {
            graph: PetGraph::new(),
            node_id_to_index: HashMap::new(),
            node_index_to_id: HashMap::new(),
        }
    }
    
    fn add_node(&mut self, node_id: String, attributes: HashMap<String, serde_json::Value>) -> PyResult<()> {
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }
        
        let node_data = NodeData { id: node_id.clone(), attributes };
        let node_index = self.graph.add_node(node_data);
        
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id);
        
        Ok(())
    }
    
    fn add_edge(&mut self, source: &str, target: &str, attributes: HashMap<String, serde_json::Value>) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", source)))?;
        let target_idx = self.node_id_to_index.get(target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", target)))?;
        
        let edge_data = EdgeData { attributes };
        self.graph.add_edge(*source_idx, *target_idx, edge_data);
        
        Ok(())
    }
    
    #[pyo3(signature = (node_ids, attributes=None))]
    fn batch_add_nodes(&mut self, node_ids: Vec<String>, attributes: Option<Vec<HashMap<String, serde_json::Value>>>) -> PyResult<()> {
        for (i, node_id) in node_ids.iter().enumerate() {
            let attrs = attributes.as_ref()
                .and_then(|attr_vec| attr_vec.get(i))
                .cloned()
                .unwrap_or_default();
            
            self.add_node(node_id.clone(), attrs)?;
        }
        Ok(())
    }
    
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }
    
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    
    fn get_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id)))?;
        
        let neighbors: Vec<String> = self.graph.neighbors(*node_idx)
            .map(|idx| self.node_index_to_id[&idx].clone())
            .collect();
        
        Ok(neighbors)
    }
}
```

### Step 3: Parallel Operations

```rust
// src/graph/operations.rs
use rayon::prelude::*;

impl FastGraph {
    fn parallel_node_operation<F, R>(&self, operation: F) -> Vec<R>
    where
        F: Fn(&NodeData) -> R + Sync + Send,
        R: Send,
    {
        self.graph.node_weights()
            .par_iter()
            .map(operation)
            .collect()
    }
    
    fn parallel_subgraph_by_filter<F>(&self, node_filter: F) -> FastGraph
    where
        F: Fn(&NodeData) -> bool + Sync + Send,
    {
        // Filter nodes in parallel
        let filtered_nodes: Vec<_> = self.graph.node_indices()
            .par_bridge()
            .filter(|&idx| {
                if let Some(node_data) = self.graph.node_weight(idx) {
                    node_filter(node_data)
                } else {
                    false
                }
            })
            .collect();
        
        // Build subgraph
        let mut subgraph = FastGraph::new();
        
        // Add filtered nodes
        for &node_idx in &filtered_nodes {
            if let Some(node_data) = self.graph.node_weight(node_idx) {
                subgraph.add_node(node_data.id.clone(), node_data.attributes.clone()).unwrap();
            }
        }
        
        // Add edges between filtered nodes
        let filtered_node_set: std::collections::HashSet<_> = filtered_nodes.iter().collect();
        for edge_idx in self.graph.edge_indices() {
            if let Some((source_idx, target_idx)) = self.graph.edge_endpoints(edge_idx) {
                if filtered_node_set.contains(&source_idx) && filtered_node_set.contains(&target_idx) {
                    let source_id = &self.node_index_to_id[&source_idx];
                    let target_id = &self.node_index_to_id[&target_idx];
                    let edge_data = self.graph.edge_weight(edge_idx).unwrap();
                    
                    subgraph.add_edge(source_id, target_id, edge_data.attributes.clone()).unwrap();
                }
            }
        }
        
        subgraph
    }
}
```

### Step 4: Content-Addressed Storage in Rust

```rust
// src/storage/content_pool.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone)]
pub struct ContentHash(pub u64);

#[pyclass]
pub struct ContentPool {
    nodes: Arc<Mutex<HashMap<ContentHash, Arc<NodeData>>>>,
    edges: Arc<Mutex<HashMap<ContentHash, Arc<EdgeData>>>>,
    node_refs: Arc<Mutex<HashMap<ContentHash, usize>>>,
    edge_refs: Arc<Mutex<HashMap<ContentHash, usize>>>,
}

impl ContentPool {
    fn hash_node(node: &NodeData) -> ContentHash {
        let serialized = serde_json::to_string(node).unwrap();
        ContentHash(xxh3_64(serialized.as_bytes()))
    }
    
    fn hash_edge(edge: &EdgeData) -> ContentHash {
        let serialized = serde_json::to_string(edge).unwrap();
        ContentHash(xxh3_64(serialized.as_bytes()))
    }
    
    pub fn intern_node(&self, node: NodeData) -> ContentHash {
        let hash = Self::hash_node(&node);
        let arc_node = Arc::new(node);
        
        {
            let mut nodes = self.nodes.lock().unwrap();
            nodes.entry(hash.clone()).or_insert(arc_node);
        }
        
        {
            let mut refs = self.node_refs.lock().unwrap();
            *refs.entry(hash.clone()).or_insert(0) += 1;
        }
        
        hash
    }
}
```

### Step 5: Python Wrapper with Seamless API

```python
# gli/graph.py
from typing import Dict, List, Any, Optional, Callable
from ._core import FastGraph as _FastGraph

class Graph:
    """High-level Python wrapper for Rust FastGraph"""
    
    def __init__(self, _core_graph: Optional[_FastGraph] = None):
        self._core = _core_graph or _FastGraph()
        self._cached_effective_data = None
    
    @classmethod
    def empty(cls, graph_store=None):
        """Create empty graph"""
        return cls()
    
    def add_node(self, node_id: str, **attributes) -> 'Graph':
        """Add node - returns new graph instance (immutable API)"""
        new_core = self._core.copy()  # Rust-level copy
        new_core.add_node(node_id, attributes)
        return Graph(new_core)
    
    def add_edge(self, source: str, target: str, **attributes) -> 'Graph':
        """Add edge - returns new graph instance"""
        new_core = self._core.copy()
        new_core.add_edge(source, target, attributes)
        return Graph(new_core)
    
    def batch_operations(self):
        """Context manager for efficient batch operations"""
        return BatchOperationContext(self)
    
    @property
    def nodes(self):
        """Lazy property that returns node view"""
        return NodeView(self._core)
    
    @property 
    def edges(self):
        """Lazy property that returns edge view"""
        return EdgeView(self._core)
    
    def create_subgraph(self, node_filter: Callable = None, **kwargs) -> 'Graph':
        """Create subgraph using Rust parallel implementation"""
        if node_filter:
            # Convert Python function to Rust-compatible filter
            rust_subgraph = self._core.parallel_subgraph_by_filter(node_filter)
            return Graph(rust_subgraph)
        
        # Other filtering logic...
        return self

class BatchOperationContext:
    """Context manager that batches operations in Rust"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self._pending_nodes = []
        self._pending_edges = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pending_nodes:
            self.graph._core.batch_add_nodes([n[0] for n in self._pending_nodes],
                                           [n[1] for n in self._pending_nodes])
        # Apply edge operations...
    
    def add_node(self, node_id: str, **attributes):
        self._pending_nodes.append((node_id, attributes))
    
    def add_edge(self, source: str, target: str, **attributes):
        self._pending_edges.append((source, target, attributes))
```

### Step 6: Performance Targets

With Rust backend, expected performance improvements:

| Operation | Current Python | Target Rust | Expected Speedup |
|-----------|----------------|-------------|------------------|
| Graph Creation (5K nodes) | 2.8s | 0.1s | **28x** |
| Node Addition | 1.2ms | 0.01ms | **120x** |
| Subgraph Creation | 4.3ms | 0.2ms | **21x** |
| Connected Component | 7.8s | 0.05s | **156x** |
| State Reconstruction | 3.2ms | 0.1ms | **32x** |

### Step 7: Migration Strategy

1. **Incremental Migration**: Keep Python API identical
2. **Feature Parity**: Implement all current GLI features in Rust
3. **Backward Compatibility**: Ensure existing code works unchanged
4. **Performance Testing**: Continuous benchmarking during development
5. **Fallback Support**: Python implementation as fallback for edge cases

This Rust backend would put GLI in the same performance class as Polars and other high-performance data libraries.
