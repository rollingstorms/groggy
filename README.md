# Groggy

**A high-performance graph and networked table library built in Rust with Python bindings**

<div align="center">
  <img src="img/groggy.svg" alt="Groggy Logo" width="300"/>
</div>

---

## 🚀 **What is Groggy?**

Groggy is a modern graph analytics library that combines **graph topology** with **tabular data operations**. Built with a high-performance Rust core and intuitive Python API, Groggy lets you seamlessly work with graph data using familiar table-like operations.

### **Key Features:**
- 🔥 **High-performance Rust core** with Python bindings
- 📊 **Unified data structures**: GraphArray, GraphTable, GraphMatrix
- 🎯 **Graph-aware analytics** with table operations  
- 🎨 **Beautiful display** with rich formatting
- ⚡ **Fast operations**: Node creation 500K+/sec, Edge creation 400K+/sec

---

## 📥 **Installation**

### From Source
```bash
git clone https://github.com/rollingstorms/groggy.git
cd groggy/python-groggy

# Install dependencies
pip install maturin

# Build and install
maturin develop --release
```

### Quick Test
```python
import groggy as gr
print("Groggy installed successfully! 🎉")
```

---

## 🚀 **Quick Start**

### **Basic Graph Operations**
```python
import groggy as gr

# Create a graph
g = gr.Graph()

# Add nodes with attributes (returns numeric IDs)
alice = g.add_node(name="Alice", age=30, dept="Engineering")
bob = g.add_node(name="Bob", age=25, dept="Design")  
charlie = g.add_node(name="Charlie", age=35, dept="Management")

# Add edges with attributes
g.add_edge(alice, bob, weight=0.8, type="collaborates")
g.add_edge(charlie, alice, weight=0.9, type="manages")

print(f"Graph: {g.node_count()} nodes, {g.edge_count()} edges")
```

### **Table-Style Data Access**
```python
# Get node data as a table
nodes_table = g.nodes.table()
print(nodes_table)
# ⊖⊖ gr.table
# ╭──────┬───────┬──────┬──────────────╮
# │    # │ name  │ age  │ dept         │
# │      │ str   │ i64  │ str          │
# ├──────┼───────┼──────┼──────────────┤
# │    0 │ Alice │ 30   │ Engineering  │
# │    1 │ Bob   │ 25   │ Design       │
# │    2 │ Charlie│ 35   │ Management   │
# ╰──────┴───────┴──────┴──────────────╯

# Statistical analysis on columns
age_column = nodes_table['age']
print(f"Average age: {age_column.mean()}")
print(f"Age range: {age_column.min()} - {age_column.max()}")

# Table-style operations
print(nodes_table.describe())  # Statistical summary
young_nodes = nodes_table[nodes_table['age'] < 30]  # Boolean filtering
```

### **Graph Analytics**
```python
# Connected components
components = g.analytics.connected_components()
print(f"Components: {len(components)}")

# Shortest paths  
path = g.analytics.shortest_path(alice, bob)
print(f"Shortest path: {path}")

# Graph traversal
bfs_result = g.analytics.bfs(alice)
dfs_result = g.analytics.dfs(alice)
```

### **Advanced Features**
```python
# Filtering with graph-aware operations
engineering = g.filter_nodes(gr.NodeFilter.attribute_filter("dept", "==", "Engineering"))
print(f"Engineering team: {engineering.node_count()} people")

# Adjacency matrix operations  
adj_matrix = g.adjacency()
print(f"Matrix shape: {adj_matrix.shape}")
print(f"Density: {adj_matrix.sum_axis(1)}")  # Row sums (node degrees)

# Export compatibility
import networkx as nx
import pandas as pd
nx_graph = g.to_networkx()           # NetworkX compatibility
df = nodes_table.to_pandas()         # Pandas DataFrame
numpy_matrix = adj_matrix.to_numpy() # NumPy array
```

---

## 🏗️ **Core Architecture**

### **Data Structures**
- **`Graph`**: Main graph container with nodes, edges, and attributes
- **`GraphArray`**: High-performance columnar arrays with statistics (like Pandas Series)
- **`GraphTable`**: Table operations on graph data (like Pandas DataFrame)  
- **`GraphMatrix`**: Matrix operations including adjacency matrices
- **`Subgraph`**: Filtered views of the main graph

### **Key Concepts**
- **Node/Edge IDs**: Groggy uses numeric IDs (not strings) returned from `add_node()`/`add_edge()`
- **Attributes**: Rich attribute system supporting strings, numbers, booleans
- **Lazy Views**: Data structures are views that only materialize when needed
- **Unified API**: Same operations work on graphs, tables, arrays, and matrices

---

## 📊 **Performance**

Groggy achieves excellent performance through its Rust core:

| Operation | Rate | Complexity |
|-----------|------|------------|
| **Node Creation** | 500,000+ nodes/sec | O(N) |
| **Edge Creation** | 400,000+ edges/sec | O(N) | 
| **Table Operations** | Linear scaling | O(N) |
| **Connected Components** | 6.5x faster than NetworkX | O(N+E) |

*Benchmarks run on representative workloads. Your performance may vary.*

---

## 🔧 **API Reference**

### **Graph Operations**
```python
# Graph creation
g = gr.Graph(directed=False)  # Undirected graph (default)
g = gr.Graph(directed=True)   # Directed graph

# Node operations
node_id = g.add_node(**attributes)      # Returns numeric ID
g.add_nodes(data_list)                  # Bulk node creation
g.set_node_attribute(node_id, "key", value)

# Edge operations  
edge_id = g.add_edge(source, target, **attributes)
g.add_edges(edge_list)                  # Bulk edge creation
g.set_edge_attribute(edge_id, "key", value)

# Graph properties
g.node_count()                          # Number of nodes
g.edge_count()                          # Number of edges  
g.density()                             # Graph density
g.is_connected()                        # Connectivity check
g.degree(node_id)                       # Node degree
```

### **Data Access**
```python
# Accessor objects
g.nodes                                 # NodesAccessor
g.edges                                 # EdgesAccessor

# Table access
nodes_table = g.nodes.table()           # All nodes as GraphTable
edges_table = g.edges.table()           # All edges as GraphTable
subgraph_table = subgraph.table()       # Subgraph data as GraphTable

# Array access (single columns)
ages = nodes_table['age']               # GraphArray of ages
weights = edges_table['weight']         # GraphArray of edge weights

# Node/edge data access
node_data = g.nodes[node_id]            # Dictionary of node attributes
edge_data = g.edges[edge_id]            # Dictionary of edge attributes
```

### **Filtering & Subgraphs**
```python
# Node filtering
young = g.filter_nodes(gr.NodeFilter.attribute_filter("age", "<", 30))
engineers = g.filter_nodes(gr.NodeFilter.attribute_filter("dept", "==", "Engineering"))

# Edge filtering  
strong = g.filter_edges(gr.EdgeFilter.attribute_filter("weight", ">", 0.5))

# Subgraph operations (all return Subgraph objects)
components = g.analytics.connected_components()
subgraph = g.nodes[:10]                 # First 10 nodes
filtered = g.filter_nodes(filter_obj)   # Filtered nodes
```

### **Analytics**
```python
# Graph algorithms (g.analytics.*)
g.analytics.connected_components()      # List of connected components
g.analytics.shortest_path(start, end)  # Shortest path between nodes
g.analytics.bfs(start_node)            # Breadth-first search
g.analytics.dfs(start_node)            # Depth-first search

# Matrix operations
adj = g.adjacency()                     # Adjacency matrix as GraphMatrix
adj[i, j]                              # Matrix element access
adj.sum_axis(0)                        # Column sums (in-degrees)
adj.sum_axis(1)                        # Row sums (out-degrees)
adj.to_numpy()                         # Convert to NumPy array
```

---

## 📚 **Documentation**

- **[User Guide](docs/user-guide/)**: Comprehensive tutorials and examples
- **[API Reference](docs/api/)**: Complete method documentation  
- **[Quickstart](docs/quickstart.rst)**: Get up and running quickly
- **[Architecture](docs/architecture/)**: Internal design and Rust core details

---

## 🧪 **Testing**

Groggy includes tests and validation scripts:

### **Rust Core Tests**
```bash
# Run Rust unit tests
cargo test

# Run specific test module
cargo test core::array
```

### **Documentation Validation**
```bash
# Run documentation validation (in documentation/development/)
python documentation/development/debug_documentation.py

# Quick validation test
python documentation/development/simple_validation_test.py

# Full validation suite  
python documentation/development/validation_test_suite.py
```

**Current validation results: 95%+ documented features working correctly** ✅

**Note**: Python integration tests are in development. Current testing relies on Rust unit tests and documentation validation scripts.

---

## 🛠️ **Development**

### **Project Structure**
```
groggy/
├── src/                    # Rust core library  
│   ├── core/              # Core data structures and algorithms (with unit tests)
│   ├── api/               # High-level graph API
│   └── display/           # Rich formatting and display
├── python-groggy/         # Python bindings and package
│   ├── src/ffi/          # Rust-to-Python FFI layer  
│   ├── python/groggy/    # Python package code
│   └── tests/            # Python tests (in development)
├── docs/                  # Sphinx documentation (RST)
├── documentation/         # Development docs (Markdown)
│   ├── development/      # Testing scripts, validation tools
│   ├── planning/         # Architecture plans  
│   ├── releases/         # Release notes
│   └── examples/         # Usage examples
└── notebooks/             # Jupyter notebooks for testing/demos
```

### **Building & Testing**
```bash
# Build development version
maturin develop

# Build release version  
maturin develop --release

# Run formatting
cargo fmt

# Run Rust tests
cargo test

# Run documentation validation
python documentation/development/debug_documentation.py
```

### **Contributing**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run validation scripts to ensure docs work
5. Submit a pull request

---

## ⚖️ **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 **Acknowledgments**

Groggy builds on the excellent work of:
- **Rust ecosystem**: Especially PyO3 for Python bindings
- **Graph libraries**: NetworkX, igraph, and others for inspiration  
- **Data science tools**: Pandas, NumPy for API design patterns

---

**Ready to get started? Try the [Quick Start](#-quick-start) above!** 🚀