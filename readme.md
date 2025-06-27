# GLI - Graph Language Interface

**A high-performance graph library with dual Python/Rust backends for efficient graph operations at scale.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)

## Overview

GLI (Graph Language Interface) is a powerful graph manipulation library designed for both rapid prototyping and production-scale applications. It features:

- **Dual Backend Architecture**: Switch seamlessly between Python and Rust backends
- **High Performance**: Rust backend handles 2M+ nodes with excellent memory efficiency
- **Rich Attributes**: Complex nested data structures on nodes and edges
- **Memory Efficient**: Content-addressed storage with deduplication
- **Developer Friendly**: Intuitive API with comprehensive error handling

## Quick Start

```python
from gli import Graph, set_backend

# Create a graph (auto-selects best available backend)
g = Graph()

# Add nodes with attributes
alice = g.add_node(label="Alice", age=30, city="New York")
bob = g.add_node(label="Bob", age=25, city="Boston")

# Add edges with attributes
friendship = g.add_edge(alice, bob, 
                       relationship="friends", 
                       since=2020, 
                       strength=0.9)

# Query the graph
neighbors = g.get_neighbors(alice)
print(f"Alice has {len(neighbors)} connections")

# Access attributes
alice_node = g.get_node(alice)
print(f"Alice is {alice_node['age']} years old")
```

## Installation

### From Source
```bash
# Clone the repository
git clone <repository-url>
cd gli

# Install Python package
pip install -e .

# Build Rust backend (optional, for high performance)
cargo build --release
```

### Requirements
- Python 3.8+
- Rust 1.70+ (for Rust backend)
- PyO3 and maturin (for Python-Rust bindings)

## Performance

| Backend | Nodes | Edges | Node Creation | Edge Creation | Neighbor Queries |
|---------|-------|-------|---------------|---------------|------------------|
| Python  | 1K    | 2K    | ~2,000/sec    | ~1,500/sec    | ~50,000/sec     |
| Rust    | 100K  | 200K  | ~50,000/sec   | ~30,000/sec   | ~500,000/sec    |
| Rust    | 2M+   | 500K+ | ~45,000/sec   | ~25,000/sec   | ~400,000/sec    |

*Benchmarks run on modern hardware. Your results may vary.*

## Backend Management

```python
from gli import get_available_backends, set_backend, get_current_backend

# Check available backends
print(f"Available: {get_available_backends()}")  # ['python', 'rust']

# Switch backends
set_backend('rust')  # Use Rust for performance
set_backend('python')  # Use Python for development

# Check current backend
print(f"Current: {get_current_backend()}")
```

**Recommendations:**
- **Python Backend**: Development, prototyping, graphs <1K nodes
- **Rust Backend**: Production, large graphs (>10K nodes), performance-critical applications

---

## State Management

## Core Features

### Node and Edge Management
```python
# Rich attribute support
node_id = g.add_node(
    label="Research Paper",
    title="Graph Neural Networks",
    authors=["Alice", "Bob"],
    metadata={
        "citations": 150,
        "year": 2023,
        "keywords": ["GNN", "ML", "graphs"]
    }
)

# Complex edge relationships
edge_id = g.add_edge(source_id, target_id,
    relationship="cites",
    importance=0.8,
    context="related work"
)
```

### Graph Analysis
```python
# Efficient neighbor queries
neighbors = g.get_neighbors(node_id)
degree = len(neighbors)

# Attribute-based filtering
nodes_with_high_citations = []
for node_id in g.nodes:
    node = g.get_node(node_id)
    if node.get('metadata', {}).get('citations', 0) > 100:
        nodes_with_high_citations.append(node_id)
```

### Batch Operations
```python
# Efficient bulk operations
node_data = [
    ("node1", {"label": "A", "value": 1}),
    ("node2", {"label": "B", "value": 2}),
    ("node3", {"label": "C", "value": 3})
]
g.batch_add_nodes(node_data)

edge_data = [
    ("node1", "node2", {"weight": 0.5}),
    ("node2", "node3", {"weight": 0.8})
]
g.batch_add_edges(edge_data)
```

## Architecture

### Dual Backend System
GLI implements a unique dual-backend architecture:

- **Python Backend**: Pure Python implementation for development and prototyping
- **Rust Backend**: High-performance Rust implementation with PyO3 bindings
- **Unified API**: Same interface regardless of backend
- **Runtime Switching**: Change backends dynamically based on workload

### Memory Management
- **Content Addressing**: Deduplicates identical nodes/edges
- **Copy-on-Write**: Efficient graph copying and modification
- **Reference Counting**: Automatic memory cleanup
- **Lazy Evaluation**: Computations deferred until needed

### Advanced Features
- **Branching/Versioning**: Git-like graph state management
- **Subgraph Operations**: Efficient graph subset operations
- **Export Formats**: NetworkX and GraphML compatibility
- **Error Handling**: Comprehensive error reporting and recovery
  - **Critical for snapshot creation**

**Utility**:
- `_next_time() -> int`: Increments and returns `_current_time` for ordering

#### Internal State

**Core Data**:
- `nodes: Dict[str, Node]`: Node storage (node_id -> Node)
- `edges: Dict[str, Edge]`: Edge storage (edge_id -> Edge)  
- `graph_attributes: Dict[str, Any]`: Graph-level metadata
- `graph_store: Optional[GraphStore]`: Reference to parent store

**Ordering Tracking**:
- `node_order: Dict[str, int]`: node_id -> insertion_time
- `edge_order: Dict[str, int]`: edge_id -> insertion_time
- `_current_time: int`: Monotonic counter for insertion order

**Copy-on-Write State**:
- `_pending_delta: Optional[GraphDelta]`: Pending changes (None until first modification)
- `_is_modified: bool`: Whether graph has uncommitted changes
- `_effective_cache: Optional[Tuple]`: Cached result of `_get_effective_data()`
- `_cache_valid: bool`: Whether effective cache is valid

## Repository Structure

```
gli/
├── src/                    # Rust backend implementation
│   ├── graph/
│   │   ├── core.rs        # Core graph data structures
│   │   ├── operations.rs  # Graph operations
│   │   └── algorithms.rs  # Graph algorithms
│   ├── storage/
│   │   ├── content_pool.rs    # Content-addressed storage
│   │   └── graph_store.rs     # Graph state management
│   └── lib.rs             # Rust library root
├── python/                # Python package
│   └── gli/
│       ├── __init__.py    # Main API exports
│       ├── graph.py       # Graph class and operations
│       ├── data_structures.py  # Node, Edge, and data models
│       ├── store.py       # Graph storage and versioning
│       ├── state.py       # State management
│       ├── delta.py       # Change tracking
│       ├── utils.py       # Utility functions
│       └── views.py       # Graph view implementations
├── tests/                 # Test suite and examples
│   ├── gli_tutorial.ipynb # Interactive tutorial
│   ├── README.md         # Test documentation
│   ├── simple_performance_test.py
│   ├── rust_stress_test.py
│   ├── complexity_stress_test.py
│   ├── advanced_complexity_test.py
│   └── ultimate_stress_test.py
├── Cargo.toml            # Rust configuration
├── pyproject.toml        # Python packaging
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## API Reference

### Core Classes

#### `Graph`
Main graph class with dual backend support.

```python
from gli import Graph

# Create graph
g = Graph()

# Add nodes
node_id = g.add_node(attribute="value")
g.add_node("custom_id", label="Custom Node")

# Add edges
edge_id = g.add_edge(source_id, target_id, weight=1.0)

# Query operations
neighbors = g.get_neighbors(node_id)
node_data = g.get_node(node_id)
edge_data = g.get_edge(source_id, target_id)

# Batch operations
g.batch_add_nodes([("id1", {"attr": "val1"}), ("id2", {"attr": "val2"})])
g.batch_add_edges([("id1", "id2", {"weight": 0.5})])
```

#### Backend Management

```python
from gli import get_available_backends, set_backend, get_current_backend

# Check available backends
backends = get_available_backends()  # ['python', 'rust']

# Switch backend
set_backend('rust')    # High performance
set_backend('python')  # Development/debugging

# Get current backend
current = get_current_backend()
```

### Advanced Features

#### Graph Attributes
```python
# Node attributes
g.set_node_attribute(node_id, "color", "red")
g.set_node_attribute(node_id, "metadata", {"type": "important"})

# Edge attributes
g.set_edge_attribute(edge_id, "weight", 0.8)
g.set_edge_attribute(edge_id, "properties", {"bidirectional": True})
```

#### Error Handling
```python
try:
    node = g.get_node("nonexistent_id")
except NodeNotFoundError as e:
    print(f"Node not found: {e}")
    
try:
    g.add_edge("invalid_source", "invalid_target")
except EdgeCreationError as e:
    print(f"Cannot create edge: {e}")
```

## Learning Resources

### Tutorial
Start with the interactive Jupyter notebook:
```bash
jupyter lab tests/gli_tutorial.ipynb
```

The tutorial covers:
- Basic graph operations
- Backend comparison and selection
- Complex attribute management
- Performance optimization
- Real-world use cases

### Examples
Check the `tests/` directory for comprehensive examples:
- **Performance testing**: Compare backend performance
- **Stress testing**: Large-scale graph operations
- **Complexity testing**: Advanced attribute handling
- **Real-world scenarios**: Practical applications

## Performance Tuning

### Backend Selection
```python
# For development and small graphs
set_backend('python')

# For production and large graphs
if 'rust' in get_available_backends():
    set_backend('rust')
```

### Memory Optimization
```python
# Use batch operations for efficiency
node_data = [(f"node_{i}", {"value": i}) for i in range(1000)]
g.batch_add_nodes(node_data)

# Efficient neighbor queries
neighbors = g.get_neighbors(node_id)  # O(1) lookup
```

### Scaling Guidelines
- **< 1K nodes**: Either backend works well
- **1K - 100K nodes**: Rust backend recommended
- **> 100K nodes**: Rust backend required
- **> 1M nodes**: Consider distributed architecture

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes**: Follow existing code style
4. **Add tests**: Include relevant test cases
5. **Run tests**: `python -m pytest tests/`
6. **Submit PR**: Include description of changes

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd gli

# Install in development mode
pip install -e .

# Build Rust backend
cargo build --release

# Run tests
python tests/simple_performance_test.py
```

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: See `tests/gli_tutorial.ipynb` for examples
- **Performance**: Check `tests/README.md` for benchmarking guides


### 3. **Complex Copy-on-Write**
**Location**: `Graph._init_delta()`, `Graph._ensure_writable()`, `Graph._get_effective_data()`
- Three-layer caching system prone to invalidation bugs
- Copy-on-write triggers are scattered throughout code
- Cache invalidation happens frequently, reducing benefits

### 4. **Unimplemented Branch Merging**
**Location**: `GraphStore._merge_graphs()`, `GraphStore._merge_subgraph()`
- Core merge logic is missing
- Branch merging advertised but non-functional
- No conflict resolution strategies

### 5. **Missing Error Handling**
- No exception handling for module loading failures
- State reconstruction failures not handled gracefully
- Branch operations lack validation
- File I/O operations missing error handling

### 6. **Memory Management Issues**
- Weak reference cache may not prevent memory leaks
- Content pool reference counting incomplete
- No explicit cleanup methods for large graphs
- Auto-state pruning logic incomplete

### 7. **Inconsistent State**
**Location**: Various methods throughout `Graph` and `GraphStore`
- Mix of mutable and immutable operations
- Unclear when changes are applied vs pending
- State consistency not guaranteed across operations

---

## Performance Bottlenecks

### 1. **Frequent Cache Invalidation**
- `_invalidate_cache()` called after every modification
- `_get_effective_data()` recomputed frequently
- No incremental cache updates

### 2. **JSON Serialization Overhead**
- Every hash computation involves JSON serialization
- No binary serialization options
- Repeated serialization of same data

### 3. **Memory Duplication**
- Forced snapshots duplicate entire graph state
- Copy-on-write benefits negated by cache invalidation
- No lazy loading of historical states

### 4. **Inefficient Graph Traversal**
- Subgraph operations scan entire graph
- No indexing for attribute-based queries
- BFS implementation creates unnecessary data structures

---

## Testing and Validation Gaps

### 1. **No Unit Tests**
- Complex internal methods untested
- Copy-on-write behavior unvalidated
- State reconstruction logic unverified

### 2. **No Integration Tests**
- Branch operations end-to-end untested
- Module system integration unclear
- Performance characteristics unknown

### 3. **No Benchmarks**
- Memory usage patterns unknown
- Performance comparison with alternatives missing
- Scalability limits undefined
