# Groggy Documentation Navigation

**Complete guide to finding what you need in the Groggy documentation**

---

## Documentation Map

```
docs/
├── 📚 Getting Started (Start here!)
├── 📖 User Guides (Learn by doing)
├── 🧠 Concepts (Understand the design)
└── 📋 API Reference (Complete method docs)
```

---

## 📚 Getting Started

Perfect for new users. Read these in order:

1. **[index.md](index.md)** - Welcome & overview
   - Why Groggy?
   - Quick example
   - Next steps

2. **[install.md](install.md)** - Installation
   - Requirements
   - pip install
   - Build from source
   - Troubleshooting

3. **[quickstart.md](quickstart.md)** - 5-minute tutorial
   - First graph
   - Basic operations
   - Common patterns

4. **[about.md](about.md)** - Project philosophy
   - Design goals
   - Performance focus
   - Community

---

## 📖 User Guides

Comprehensive tutorials with working examples:

### Core Concepts
- **[graph-core.md](guide/graph-core.md)** - Foundation
  - Creating graphs
  - Adding nodes/edges
  - Basic operations

- **[accessors.md](guide/accessors.md)** - Data access patterns
  - `g.nodes` and `g.edges`
  - Filtering and queries
  - Attribute access

### Data Structures
- **[arrays.md](guide/arrays.md)** - Array operations
  - NodesArray, EdgesArray
  - NumArray operations
  - Array transformations

- **[tables.md](guide/tables.md)** - Tabular data
  - GraphTable, NodesTable, EdgesTable
  - Pandas integration
  - Export/import

- **[matrices.md](guide/matrices.md)** - Matrix operations
  - Adjacency matrices
  - Laplacian matrices
  - Custom matrices

### Advanced Features
- **[subgraphs.md](guide/subgraphs.md)** - Subgraph creation
  - Filtering nodes/edges
  - Induced subgraphs
  - Subgraph operations

- **[subgraph-arrays.md](guide/subgraph-arrays.md)** - Collections
  - Connected components
  - Community detection
  - Batch operations

- **[algorithms.md](guide/algorithms.md)** - Graph algorithms
  - Shortest paths
  - Centrality measures
  - Community detection

- **[pipeline.md](guide/pipeline.md)** - Pipeline API
  - `Pipeline` objects
  - `apply()` helper
  - Subgraph.apply()

- **[builder.md](guide/builder.md)** - Custom pipelines
  - Builder DSL
  - Step primitives
  - Executing custom algorithms

- **[neural.md](guide/neural.md)** - Neural networks
  - Graph embeddings
  - GNN integration
  - Deep learning

### Performance & Integration
- **[performance.md](guide/performance.md)** - Optimization
  - Best practices
  - Profiling
  - Memory management

- **[integration.md](guide/integration.md)** - Other libraries
  - NetworkX
  - igraph
  - PyTorch Geometric

---

## 🧠 Concepts

Understand how Groggy works:

- **[overview.md](concepts/overview.md)** - High-level design
  - Core principles
  - Architecture overview
  - Design patterns

- **[architecture.md](concepts/architecture.md)** - System design
  - Three-tier architecture
  - Core (Rust) layer
  - FFI layer
  - API (Python) layer

- **[connected-views.md](concepts/connected-views.md)** - Object transformations
  - Delegation chains
  - Type transformations
  - Method forwarding

- **[origins.md](concepts/origins.md)** - Project history
  - Motivation
  - Evolution
  - Future direction

---

## 📋 API Reference

Complete method documentation for all objects:

### Core Objects

- **[graph.md](api/graph.md)** - Graph class (64 methods)
  - Graph construction
  - Node/edge operations
  - Algorithms
  - Subgraph creation
  - Export/import

- **[subgraph.md](api/subgraph.md)** - Subgraph class (59 methods)
  - Filtered views
  - Subgraph operations
  - Conversion methods

- **[pipeline.md](api/pipeline.md)** - Pipeline and apply helpers
- **[builder.md](api/builder.md)** - Builder DSL API

- **[graphmatrix.md](api/graphmatrix.md)** - GraphMatrix class (93 methods)
  - Matrix operations
  - Decompositions
  - Embeddings
  - Transformations

### Accessors

- **[nodesaccessor.md](api/nodesaccessor.md)** - NodesAccessor (15 methods)
  - Node filtering
  - Attribute access
  - Node queries

- **[edgesaccessor.md](api/edgesaccessor.md)** - EdgesAccessor (16 methods)
  - Edge filtering
  - Attribute access
  - Edge queries

### Arrays

- **[nodesarray.md](api/nodesarray.md)** - NodesArray (13 methods)
  - Node collections
  - Array operations
  - Transformations

- **[edgesarray.md](api/edgesarray.md)** - EdgesArray (15 methods)
  - Edge collections
  - Array operations
  - Transformations

- **[subgrapharray.md](api/subgrapharray.md)** - SubgraphArray (14 methods)
  - Subgraph collections
  - Batch operations
  - Aggregations

- **[numarray.md](api/numarray.md)** - NumArray (20 methods)
  - Numeric arrays
  - Statistics
  - Transformations

### Tables

- **[graphtable.md](api/graphtable.md)** - GraphTable (22 methods)
  - Graph tabular view
  - Combined node/edge data
  - Aggregations

- **[nodestable.md](api/nodestable.md)** - NodesTable (33 methods)
  - Node data table
  - Filtering & sorting
  - Export operations

- **[edgestable.md](api/edgestable.md)** - EdgesTable (37 methods)
  - Edge data table
  - Filtering & sorting
  - Export operations

- **[basetable.md](api/basetable.md)** - BaseTable
  - Base table operations
  - Common methods

---

## 🔍 Finding What You Need

### By Task

**Want to...**

- **Get started?** → [Quickstart](quickstart.md)
- **Install Groggy?** → [Installation](install.md)
- **Create a graph?** → [Graph Core Guide](guide/graph-core.md)
- **Filter nodes/edges?** → [Accessors Guide](guide/accessors.md) + [NodesAccessor API](api/nodesaccessor.md)
- **Work with tables?** → [Tables Guide](guide/tables.md)
- **Use algorithms?** → [Algorithms Guide](guide/algorithms.md)
- **Optimize performance?** → [Performance Guide](guide/performance.md)
- **Integrate with NetworkX?** → [Integration Guide](guide/integration.md)
- **Understand the design?** → [Architecture](concepts/architecture.md)

### By Experience Level

**Beginner (Never used Groggy)**
1. [Index](index.md) - Overview
2. [Install](install.md) - Setup
3. [Quickstart](quickstart.md) - First graph
4. [Graph Core Guide](guide/graph-core.md) - Basics

**Intermediate (Know the basics)**
1. [Accessors Guide](guide/accessors.md) - Data access
2. [Tables Guide](guide/tables.md) - Tabular operations
3. [Subgraphs Guide](guide/subgraphs.md) - Filtering
4. [Algorithms Guide](guide/algorithms.md) - Analytics

**Advanced (Building complex systems)**
1. [Architecture](concepts/architecture.md) - System design
2. [Connected Views](concepts/connected-views.md) - Delegation
3. [Performance Guide](guide/performance.md) - Optimization
4. Complete [API Reference](api/graph.md) - All methods

### By Object Type

**Graph Objects**
- [Graph API](api/graph.md) - Main graph object
- [Subgraph API](api/subgraph.md) - Filtered views

**Accessor Objects**
- [NodesAccessor API](api/nodesaccessor.md) - `g.nodes`
- [EdgesAccessor API](api/edgesaccessor.md) - `g.edges`

**Array Objects**
- [NodesArray API](api/nodesarray.md) - Node collections
- [EdgesArray API](api/edgesarray.md) - Edge collections
- [SubgraphArray API](api/subgrapharray.md) - Subgraph collections
- [NumArray API](api/numarray.md) - Numeric arrays

**Table Objects**
- [GraphTable API](api/graphtable.md) - Full graph table
- [NodesTable API](api/nodestable.md) - Node table
- [EdgesTable API](api/edgestable.md) - Edge table
- [BaseTable API](api/basetable.md) - Base table

**Matrix Objects**
- [GraphMatrix API](api/graphmatrix.md) - Matrix operations

---

## 📊 Documentation Stats

- **Total Files**: 32
- **Total Lines**: 19,714
- **API Methods Documented**: 401
- **Coverage**: 100%

### File Breakdown
- Getting Started: 4 files (1,001 lines)
- User Guides: 11 files (6,491 lines)
- Concepts: 4 files (1,637 lines)
- API Reference: 13 files (10,585 lines)

---

## 🎯 Quick Reference

### Most Common Operations

```python
import groggy as gr

# Create graph
g = gr.Graph()

# Add nodes with attributes
n1 = g.add_node(name="Alice", age=29)
n2 = g.add_node(name="Bob", age=35)

# Add edges with attributes
e = g.add_edge(n1, n2, weight=5.0)

# Filter nodes → Subgraph
young = g.nodes[g.nodes["age"] < 30]

# Get table → DataFrame
df = g.nodes.table().to_pandas()

# Run algorithm
g.connected_components(inplace=True)

# Get matrix
A = g.adjacency_matrix()
```

**Where to learn more:**
- Basic operations: [Graph Core Guide](guide/graph-core.md)
- Filtering: [Accessors Guide](guide/accessors.md)
- Tables: [Tables Guide](guide/tables.md)
- Algorithms: [Algorithms Guide](guide/algorithms.md)
- Matrices: [Matrices Guide](guide/matrices.md)

---

## 💡 Tips for Navigation

1. **Start with guides, reference the API**
   - Guides teach concepts with examples
   - API docs list every method with details

2. **Use the object transformation graph**
   - See [Connected Views](concepts/connected-views.md)
   - Understand how objects transform

3. **Follow the delegation chains**
   - `Graph → Subgraph → Table → DataFrame`
   - Each step documented in API reference

4. **Check both guide and API docs**
   - Guide: "How to use X"
   - API: "What methods does X have?"

---

## 📝 Documentation Status

✅ **Complete** - All core features documented

See [DOCUMENTATION_STATUS.md](DOCUMENTATION_STATUS.md) for detailed status report.

---

**Need help?** Check the [User Guides](guide/graph-core.md) or [API Reference](api/graph.md)
