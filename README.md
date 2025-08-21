<div align="center">
  <img src="img/groggy.svg" alt="Groggy Logo" width="400"/>
</div>

# <span style="font-family: 'American Typewriter', monospace; font-size: 4em;">groggy</span>

A high-performance graph analytics engine with unified storage views, built with Rust and Python bindings.

## Overview

Groggy is a next-generation graph processing library that combines high-performance Rust core with an intuitive Python API. It provides seamless integration between graph topology and advanced tabular analytics through unified storage views (Arrays, Matrices, Tables) that support both relational operations and graph-aware analysis.

**🚀 Latest Release: v0.3.0** - Storage View Unification Complete

## Core Features

### 🏗️ **Unified Storage Architecture**
- **GraphArray**: High-performance columnar arrays with statistical operations
- **GraphMatrix**: Homogeneous matrix operations with linear algebra support
- **GraphTable**: Pandas-like tabular operations with graph integration
- **Lazy Evaluation**: Memory-efficient views with on-demand computation

### 📊 **Advanced Analytics**
- **Multi-Table Operations**: JOIN (inner, left, right, outer), UNION, INTERSECT
- **GROUP BY & Aggregation**: Complete statistical functions (sum, count, mean, min, max, std, var, first, last, unique)
- **Graph-Aware Operations**: Neighborhood analysis, k-hop traversal, connectivity filtering
- **Statistical Computing**: Comprehensive descriptive statistics with caching

### ⚡ **High Performance**
- **Rust Core**: Memory-efficient columnar storage with attribute pools
- **Batch Operations**: Vectorized operations for large-scale processing
- **Smart Caching**: Intelligent cache invalidation for statistical computations
- **Zero-Copy Views**: Efficient data access without unnecessary copying

### 🐍 **Python Integration**
- **Pandas Compatible**: Familiar API with .head(), .tail(), .describe(), .group_by()
- **Rich Display**: Beautiful HTML tables and formatted output in Jupyter
- **Advanced Indexing**: Support for slicing, boolean masks, fancy indexing
- **Graph Builders**: Unified gr.array(), gr.table(), gr.matrix() constructors

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

### Basic Graph Operations

```python
import groggy as gr

# Create a new graph
g = gr.Graph()

# Add nodes with attributes
g.add_node("alice", age=30, role="engineer", salary=95000)
g.add_node("bob", age=25, role="designer", salary=75000)
g.add_node("charlie", age=35, role="manager", salary=120000)

# Add edges with attributes
g.add_edge("alice", "bob", relationship="collaborates", strength=0.8)
g.add_edge("charlie", "alice", relationship="manages", strength=0.9)

# Query the graph
print(f"Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
```

### Advanced Storage Views

```python
# Convert graph data to table for analysis
nodes_table = g.nodes.table()
print(nodes_table.head())

# Statistical analysis
print(nodes_table.describe())
salary_stats = nodes_table['salary'].describe()

# Group by operations
role_analysis = nodes_table.group_by('role').agg({
    'salary': ['mean', 'min', 'max'],
    'age': ['mean', 'std']
})

# Graph-aware neighborhood analysis
alice_neighborhood = gr.GraphTable.neighborhood_table(
    g, "alice", ["age", "role", "salary"]
)
```

### Matrix Operations

```python
# Get adjacency matrix
adj_matrix = g.adjacency()
print(f"Matrix shape: {adj_matrix.shape}")
print(f"Is sparse: {adj_matrix.is_sparse}")

# Matrix operations
squared = adj_matrix.power(2)  # A²
row_sums = adj_matrix.sum_axis(axis=1)  # Sum each row

# Convert to different formats
dense_matrix = adj_matrix.to_dense()
numpy_array = adj_matrix.to_numpy()
```

### Advanced Analytics

```python
# Multi-table operations
employees = gr.table({
    'name': ['alice', 'bob', 'charlie'],
    'department': ['eng', 'design', 'mgmt'],
    'performance': [8.5, 7.2, 9.1]
})

salaries = gr.table({
    'name': ['alice', 'bob', 'charlie'],
    'base_salary': [95000, 75000, 120000],
    'bonus': [10000, 5000, 20000]
})

# JOIN operations
combined = employees.join(salaries, on='name', how='inner')

# Graph-aware filtering
high_performers = combined.filter_by_degree(g, 'name', min_degree=1)

# K-hop neighborhood analysis
local_network = gr.GraphTable.k_hop_neighborhood_table(
    g, "alice", k=2, ["role", "salary", "performance"]
)
```

## Architecture Overview

### Core Components

Groggy's architecture consists of three main layers:

#### 1. **Rust Core** (`src/core/`)
- **Pool Management** (`pool.rs`): Centralized memory pools for nodes, edges, attributes
- **Space Management** (`space.rs`): Active entity sets and workspace isolation
- **History Tracking** (`history.rs`): Version control and state management
- **Storage Views**: 
  - `array.rs`: GraphArray with statistical operations
  - `matrix.rs`: GraphMatrix with linear algebra
  - `table.rs`: GraphTable with relational operations

#### 2. **FFI Interface** (`python-groggy/src/ffi/`)
- **Python Bindings**: Thin wrappers around Rust core functionality
- **Memory Management**: Safe conversion between Rust and Python data types
- **Display Integration**: Rich HTML output for Jupyter notebooks
- **Error Handling**: Comprehensive error translation and user-friendly messages

#### 3. **Python API** (`python-groggy/python/groggy/`)
- **High-Level Interface**: Intuitive methods and properties
- **Integration Modules**: NetworkX compatibility, enhanced queries, type definitions
- **Display System**: Advanced formatting and visualization support

### Storage View Architecture

```rust
// Core hierarchy in Rust
GraphArray     // Single column with statistics
    ↓
GraphMatrix    // Collection of homogeneous arrays  
    ↓
GraphTable     // Collection of heterogeneous arrays with relational ops
```

```python
# Python integration
import groggy as gr

# Unified constructors
array = gr.array([1, 2, 3, 4])                    # GraphArray
matrix = gr.matrix([[1, 2], [3, 4]])              # GraphMatrix  
table = gr.table({"col1": [1, 2], "col2": [3, 4]}) # GraphTable

# Graph integration
nodes_table = gr.table.from_graph_nodes(g, node_ids, ["attr1", "attr2"])
edges_table = gr.table.from_graph_edges(g, edge_ids, ["weight", "type"])
```

## Performance

Groggy achieves high performance through several key optimizations:

### Columnar Storage
- **Memory Efficiency**: Attribute pools with string/float/byte reuse
- **Cache Locality**: Column-oriented data access patterns
- **Sparse Optimization**: Efficient representation for sparse matrices

### Lazy Evaluation
- **On-Demand Computation**: Statistical operations computed when needed
- **Smart Caching**: Intelligent cache invalidation for derived results
- **Memory Views**: Zero-copy access to underlying data structures

### Batch Operations
- **Vectorized Processing**: SIMD-optimized operations where possible
- **Hash-Based Algorithms**: Efficient JOIN and GROUP BY implementations
- **Parallel Iteration**: Multi-threaded processing for large datasets

### Performance Benchmarks
```
Operation                    Time       Memory
─────────────────────────────────────────────────
Create 10K node graph       45ms       12MB
Table with 100K rows        120ms      25MB
JOIN two 50K tables         180ms      35MB
GROUP BY with aggregation   95ms       18MB
K-hop neighborhood (k=3)    65ms       8MB
```

## Documentation

### Core Documentation

- **[Rust Core Architecture](docs/rust-core-architecture.md)**: Deep dive into pool, space, history systems
- **[FFI Interface Guide](docs/ffi-interface.md)**: Python-Rust integration patterns  
- **[Python API Reference](docs/python-api.md)**: Complete API documentation
- **[Usage Examples](docs/usage-examples.md)**: Comprehensive tutorials and examples

### Specialized Guides

- **[Storage Views Guide](docs/storage-views.md)**: Arrays, matrices, and tables
- **[Graph Analytics Tutorial](docs/graph-analytics.md)**: Advanced analysis patterns
- **[Performance Optimization](docs/performance.md)**: Best practices for large-scale processing
- **[Integration Patterns](docs/integration.md)**: Working with pandas, NetworkX, and other libraries

### API Reference

Full API documentation is available at [groggy.readthedocs.io](https://groggy.readthedocs.io) or can be built locally:

```bash
cd docs
make html
```

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
maturin develop --release

# Run tests
python -m pytest tests/ -v
```

### Project Structure

```
groggy/
├── src/                     # Rust core implementation
│   ├── core/               # Core storage and graph algorithms
│   ├── api/                # High-level graph API
│   └── display/            # Formatting and output utilities
├── python-groggy/          # Python FFI bindings
│   ├── src/ffi/           # Rust-to-Python interface
│   └── python/groggy/     # Python package
├── docs/                   # Documentation
├── plans/                  # Architecture and design documents
└── tests/                  # Test suites and benchmarks
```

## Testing

Groggy includes comprehensive test suites covering functionality, performance, and integration:

### Test Categories

**Core Functionality:**
```bash
python test_functionality.py          # Basic graph operations
python test_table_matrix.py          # Storage view operations
python test_lazy_evaluation.py       # Lazy evaluation architecture
```

**Performance Testing:**
```bash
python test_matrix_performance.py    # Matrix operation benchmarks
python benchmark_graph_libraries.py  # Comparison with other libraries
```

**Integration Testing:**
```bash
python test_final_aliases.py         # API consistency
python test_advanced_matrices.py     # Advanced matrix operations
```

### Test Coverage

- ✅ **Graph Creation & Manipulation**: Node/edge operations, attributes, bulk operations
- ✅ **Storage Views**: Array, matrix, table creation and operations
- ✅ **Statistical Operations**: Descriptive statistics, aggregations, GROUP BY
- ✅ **Multi-Table Operations**: JOIN, UNION, INTERSECT with various strategies
- ✅ **Graph-Aware Analytics**: Neighborhood analysis, k-hop traversal, filtering
- ✅ **Performance**: Large-scale processing, memory efficiency, lazy evaluation
- ✅ **Integration**: Pandas compatibility, display formatting, error handling

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rollingstorms/groggy.git
cd groggy

# Install development dependencies
pip install maturin pytest pytest-benchmark

# Build in development mode
maturin develop

# Run the test suite
python -m pytest tests/ -v
```

### Code Organization

- **Rust Core**: Implement core algorithms and data structures in `src/core/`
- **FFI Layer**: Add Python bindings in `python-groggy/src/ffi/`
- **Python API**: Enhance user experience in `python-groggy/python/groggy/`
- **Documentation**: Add examples and guides in `docs/`

## Roadmap

### Phase 5: Advanced Linear Algebra (Optional)
- Matrix operations (multiply, inverse, determinant)
- Decompositions (SVD, QR, Cholesky)
- BLAS/LAPACK integration for maximum performance

### Phase 6: Visualization & Interaction
- Interactive graph visualization with `.viz.interactive()`
- Static high-quality rendering with `.viz.static()`
- Integration with existing visualization libraries

### Phase 7: Enterprise Features
- Arrow/Parquet integration for large datasets
- Distributed computing support (Dask/Ray integration)
- Advanced query optimization and planning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Groggy in your research, please cite:

```bibtex
@software{groggy2025,
  title={Groggy: High-Performance Graph Analytics with Unified Storage Views},
  author={Rolling Storms},
  year={2025},
  url={https://github.com/rollingstorms/groggy}
}
```