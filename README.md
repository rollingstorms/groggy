# <span style="font-family: 'American Typewriter', monospace; font-size: 4em;">groggy</span>

A high-performance graph language engine built with Rust and Python bindings.

## Overview

Groggy is a graph processing library designed for efficient manipulation and analysis of large-scale graphs. It combines the performance of Rust with the ease of use of Python, providing a powerful toolkit for graph-based applications.

Groggy is in development! I am excited to release this early public version for testing and development. Please contribute any thoughts or comments!

## Features

- **High Performance**: Rust-based core for maximum speed and memory efficiency
- **Python Integration**: Easy-to-use Python API with familiar syntax
- **Scalable**: Handles large graphs with millions of nodes and edges
- **Batch Operations**: Efficient bulk operations (330x faster than individual)
- **Graph Operations**: Comprehensive set of graph algorithms and operations
- **Memory Efficient**: Optimized data structures for minimal memory footprint
- **State Management**: Save, load, and track graph states over time
- **Comprehensive Testing**: Full test suite with performance benchmarks

## Installation

### From Source

```bash
git clone https://github.com/rollingstorms/groggy.git
cd groggy

# Install development dependencies
pip install maturin

# Build and install
maturin develop --release
```

## Quick Start

```python
import groggy as gr

# Create a new graph
g = gr.Graph()

# Add nodes with attributes
g.add_node("alice", age=30, role="engineer")
g.add_node("bob", age=25, role="designer")
g.add_node("charlie", age=35, role="manager")

# Add edges with attributes
g.add_edge("alice", "bob", relationship="collaborates")
g.add_edge("charlie", "alice", relationship="manages")

# Query the graph
print(f"Nodes: {len(g.nodes)}")
print(f"Edges: {len(g.edges)}")

# Check connectivity
print(f"Alice and Bob connected: {g.has_edge('alice', 'bob')}")

# Get node/edge data
alice_data = g.get_node("alice")
print(f"Alice: {alice_data}")

# Filter nodes
engineers = g.filter_nodes(lambda node_id, attrs: attrs.get("role") == "engineer")
print(f"Engineers: {engineers}")

# Efficient batch operations
nodes_data = [
    {'id': 'user_1', 'score': 100, 'active': True},
    {'id': 'user_2', 'score': 200, 'active': False},
    {'id': 'user_3', 'score': 150, 'active': True}
]
g.add_nodes(nodes_data)

edges_data = [
    {'source': 'user_1', 'target': 'user_2', 'weight': 0.8},
    {'source': 'user_2', 'target': 'user_3', 'weight': 0.6}
]
g.add_edges(edges_data)

# State management
g.save_state("initial")
g.update_node("user_1", {"score": 250, "promoted": True})
g.save_state("after_promotion")
```

## Advanced Usage

```python
import groggy as gr

# Create graph with batch operations
g = gr.Graph()

# Bulk operations for efficiency
nodes = [{'id': f'node_{i}', 'value': i} for i in range(1000)]
g.add_nodes(nodes)

edges = [{'source': f'node_{i}', 'target': f'node_{i+1}', 'weight': 1.0} 
         for i in range(999)]
g.add_edges(edges)

# Batch updates
updates = {f'node_{i}': {'updated': True, 'timestamp': '2025-06-28'} 
          for i in range(0, 1000, 10)}
g.update_nodes(updates)

```

## Documentation

Full documentation is available at [groggy.readthedocs.io](https://groggy.readthedocs.io) or can be built locally:

```bash
cd docs
make html
```

## Performance

Groggy is designed for high-performance graph processing with a unified Rust-based columnar storage system:

- **Optimized Filtering**: Fast bitmap-based exact matching and range queries
- **Columnar Storage**: Efficient attribute storage with O(1) lookups for exact matches
- **Scalable Architecture**: Handles large graphs efficiently
- **Memory Efficient**: Optimized data structures for minimal memory footprint

Key architectural features:
- Unified type system (NodeData, EdgeData, GraphType)
- Bitmap indexing for fast attribute filtering
- Optimized numeric and string comparison operations
- Efficient batch operations for bulk data processing

## Development

### Building from Source

Requirements:
- Rust 1.70+
- Python 3.8+
- Maturin for building Python extensions

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Build the Rust extension
maturin develop

```

## Testing

Groggy includes comprehensive test suites to ensure reliability and performance:

### Running Tests

**Basic functionality tests:**
```bash
# Run all tests with pytest
python -m pytest tests/test_functionality.py -v

# Or run directly
python tests/test_functionality.py
```

**Stress test (10K nodes/edges):**
```bash
python tests/test_stress.py
```

### Test Coverage

The test suite includes:

- **Functionality Tests** (`test_functionality.py`):
  - Basic graph operations (nodes, edges, properties)
  - State management and tracking
  - Node and edge filtering
  - Batch updates and modifications
  - Graph analysis and neighbor queries

- **Stress Tests** (`test_stress.py`, `test_stress_quick.py`):
  - Large-scale graph creation (10K nodes/edges)
  - Batch operations performance testing
  - Memory efficiency validation
  - Graph analysis on large datasets
  - State management with bulk data

- **Performance Benchmarks**:
  - Comprehensive filtering performance testing
  - Graph creation and manipulation benchmarks
  - Comparison with other graph libraries
  - Memory efficiency validation

### Test Environment Setup

For development testing:
```bash
# Install test dependencies
pip install pytest pytest-benchmark

# Install groggy in development mode
pip install -e .

# Run full test suite
python -m pytest tests/ -v
```

### Performance Testing

For performance validation:
```bash
# Run comprehensive functionality tests
python run_tests.py

# Run stress tests
python tests/test_stress.py

# Benchmark against other libraries
python benchmark_graph_libraries.py
```


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
