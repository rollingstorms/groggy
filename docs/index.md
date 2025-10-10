# Groggy

**A graph analytics library for Python with a Rust core**

---

## Welcome

Groggy is a modern graph analytics library that combines **graph topology** with **tabular data operations**. Built with a high-performance Rust core and intuitive Python API, Groggy lets you seamlessly work with graph data using familiar table-like operations.

## Quick Links

### :material-clock-fast: Get Started in 5 Minutes
Install Groggy and build your first graph
→ [Quickstart](quickstart.md)

### :material-book-open-variant: User Guide
Comprehensive tutorials and examples
→ [User Guide](guide/graph-core.md)

### :material-api: API Reference
Detailed API documentation with theory and examples
→ [API Reference](api/graph.md)

### :material-lightbulb: Core Concepts
Understand Groggy's architecture and design philosophy
→ [Concepts](concepts/overview.md)

---

## Why Groggy?

### Everything is Connected

At its core, **a graph is a network** - a collection of entities (nodes) and the relationships (edges) between them. But connections carry meaning: interactions, flows, dependencies, influence. When you map those connections, entire hidden structures reveal themselves.

### Graphs + Tables + Arrays + Matrices

Groggy takes this further: every node and edge can have attributes stored in an efficient columnar format. Your "graph" isn't just dots and lines — it's a rich, living dataset where you can:

- **Query** like a database
- **Transform** like pandas
- **Analyze** like NetworkX
- **Compute** like NumPy

### High Performance, Intuitive API

- **Rust core** for memory-safe, high-performance operations
- **Columnar storage** for efficient bulk attribute operations
- **Delegation chains** for expressive, composable workflows
- **Git-like versioning** for time-travel queries

---

## A Quick Taste

```python
# Example: Build → Inspect → Query → Algorithm → Views → Viz
# Goal: demonstrate connected views and common ops in ~20 lines.
# Remember: everything is a graph.

import groggy as gr

# ───────────────────────────────────────────────
# 1. Build a tiny graph
# ───────────────────────────────────────────────
g = gr.Graph()

# add nodes
alice = g.add_node(name="Alice", age=29)
bob   = g.add_node(name="Bob",   club="Purple", active=True, age=55)
carol = g.add_node(name="Carol", club="Blue",   active=True, age=31)

# add edges
g.add_edge(alice, bob,   weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob,   carol, weight=1)

# ───────────────────────────────────────────────
# 2. Inspect (Graph → Table)
# ───────────────────────────────────────────────
nodes_tbl = g.nodes.table()
edges_tbl = g.edges.table()

print("Nodes table:")
print(nodes_tbl.head())   # columns: [id, name, age, club, active, ...]
print("\nEdges table:")
print(edges_tbl.head())   # columns: [src, dst, weight, ...]

# ───────────────────────────────────────────────
# 3. Query / Slice → returns Subgraph
# ───────────────────────────────────────────────
purple_nodes  = g.nodes[g.nodes["club"] == "Purple"]
younger_nodes = g.nodes[g.nodes["age"]  <  30]

print(f"\nPurple nodes:  {len(purple_nodes)}")
print(f"Younger nodes: {len(younger_nodes)}")

# ───────────────────────────────────────────────
# 4. Algorithm → Connected Components
# ───────────────────────────────────────────────
g.connected_components(inplace=True, label="component")
num_components = len(g.nodes["component"].unique())
print(f"\nConnected components: {num_components}")

# ───────────────────────────────────────────────
# 5. Views → Array → Matrix
# ───────────────────────────────────────────────
ages = g.nodes["age"]
mean_age = ages.mean()
print(f"\nMean age: {mean_age:.1f}")

L = g.laplacian_matrix()
print(f"Laplacian shape: {L.shape}")

# ───────────────────────────────────────────────
# 6. Viz → view the graph, color by computed attribute
# ───────────────────────────────────────────────
print("\nRendering visualization...")
g.viz.show(node_color="component")
```

---

## Everything is a Graph

Even the Groggy library itself can be thought of as a graph:

- **Nodes** = Object types (Graph, Subgraph, Table, Array, Matrix)
- **Edges** = Methods that transform one type into another

```
Graph → connected_components() → SubgraphArray
SubgraphArray → table() → GraphTable
GraphTable → agg() → AggregationResult
```

This design makes it easy to learn: once you understand the transformation patterns, the entire API becomes intuitive.

---

## Next Steps

!!! tip "New to Groggy?"
    Start with the [Quickstart Guide](quickstart.md) to get up and running in minutes.

!!! info "Want to understand the design?"
    Read the [Concepts & Architecture](concepts/overview.md) to learn how Groggy works under the hood.

!!! example "Ready to build?"
    Jump into the [User Guide](guide/graph-core.md) for comprehensive tutorials.

---

## Community & Support

- **GitHub**: [rollingstorms/groggy](https://github.com/rollingstorms/groggy)
- **Issues**: [Report bugs or request features](https://github.com/rollingstorms/groggy/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/rollingstorms/groggy/discussions)

---

*Built with performance in mind. Designed for humans.*
