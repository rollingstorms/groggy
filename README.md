<div align="left"><img src="img/groggy.svg" alt="groggy Logo" width="600"/>
</div>

# groggy

**A graph analytics library for Python with a Rust core**

---

## 👀 **What is groggy?**

groggy is a modern graph analytics library that combines **graph topology** with **tabular data operations**. Built with a high-performance Rust core and intuitive Python API, groggy lets you seamlessly work with graph data using familiar table-like operations.

## **Quick Start:**

### Installation

```bash
pip install groggy
```

### Example Python Usage

```python
import groggy as gr
from groggy.algorithms.centrality import pagerank
from groggy.algorithms.community import label_propagation

# Create a new graph
g = gr.Graph()

# Add nodes and edges
alice = g.add_node(name="Alice", age=29)
bob   = g.add_node(name="Bob",   club="Purple", active=True, age=55)
carol = g.add_node(name="Carol", club="Blue",   active=True, age=31)
g.add_edge(alice, bob, weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob, carol, weight=1)

# Inspect the graph
print(g.nodes.table().head())
print(g.edges.table().head())

# Query the graph
purple_nodes = g.nodes[g.nodes["club"] == "Purple"]
younger_nodes = g.nodes[g.nodes["age"] < 30]

# Run a graph algorithm
g.connected_components(inplace=True, label='component')

# Pipeline-style algorithms with g.apply()
g.apply(pagerank(max_iter=10, output_attr="score"))
print(g.nodes.table().sort_by("score").tail(10))

g.apply(label_propagation(output_attr="label"))

# Viz. the graph
g.viz.show(node_color="label")
```

## **A Little Graph Theory:**

A graph is composed of nodes and edges…or is it vertices and edges? or maybe nodes and links?
Let's start over.

At its core, **a graph is a network** - a collection of entities (nodes) and the relationships (edges) between them. That's the first truth of graph theory: 

**Everything is connected.** 

The second truth is more interesting: 

**Connections carry meaning.**

Edges aren't just lines on a diagram — they represent interactions, flows, dependencies, or influence. And when you map those connections, entire hidden structures begin to reveal themselves:
- Communities of related entities
- Bridges between otherwise disconnected groups
- Patterns that point to anomalies, generalizations, or insights

groggy builds on these ideas and takes them further: every node and edge can have attributes that are stored in a equally efficient format.
That means your "graph" isn't just dots and lines — it's a rich, living dataset:
- Nodes can store labels, features, embeddings, or metadata
- Edges can carry weights, timestamps, permissions, or scores
- You can analyze both structure and data, as a graph or as a table, together.

Whether you're exploring dynamic networks, running graph algorithms, or building machine learning pipelines, groggy aims to be a modular, high-performance foundation. It's designed for thinking in graphs — not only visualizing them, but querying, transforming, and learning from them.

⸻

## 1) A first graph

Start tiny. Let the structure grow as needed.

```python
import groggy as gr

g = gr.Graph()

alice = g.add_node(label="Alice", age=29)  # any kwargs become attributes
bob   = g.add_node(label="Bob",   club="Red")

g.add_edge(alice, bob, weight=5)           # edges use the returned node IDs

print(len(g.nodes))   # 2   (accessor, not a method)
print(len(g.edges))   # 1
```

Nodes return an integer ID. Keep it, reuse it. Any keyword you invent becomes a stored attribute.

⸻

## 2) Exploring real data

Use a built-in generator to play with something non-toy:

```python
g = gr.generators.karate_club()
print(g.table())  # GraphTable summary (sizes of nodes/edges tables)
```

Grab a whole attribute column straight from the graph and peek:

```python
names = g["name"]   # BaseArray of node names
print(names.head(5))
```

⸻

## 3) Attributes are the heart and soul of the graph

You can define attributes at creation time (kwargs), or later with set_attrs.
Make up any attribute names; groggy persists them.

```python
# At creation:
carol = g.add_node(label="Carol", role="Analyst", active=True)

# Later, in bulk:
g.nodes.set_attrs({
    carol: {"club": "Blue", "age": 31}
})

# Edges too:
g.edges.set_attrs({
    0: {"weight": 0.7, "type": "friendship"}
})
```

⸻

## 4) From graphs to tables — and back

Under the hood is columnar storage you can inspect, export, and rebuild from.

```python
nodes = g.nodes.table()
edges = g.edges.table()

print(nodes.head())
print(edges.head())

# Round-trip: build a GraphTable explicitly, then convert back to a Graph
gt = gr.GraphTable(nodes=nodes, edges=edges)
g2 = gt.to_graph()
```

Export to files or Pandas:

```python
nodes.to_parquet("nodes.parquet")
edges.to_csv("edges.csv")

df = nodes.to_pandas()
```

⸻

## 5) Columns & boolean filters (Pandas-style)

Masks feel familiar:

```python
blue_nodes  = g.nodes[g.nodes["club"] == "Blue"]
older_nodes = g.nodes[g.nodes["age"]  >  30]
```

⸻

## 6) Subgraphs (the quick way)

Slices and selections give you instant working sets:

```python
g_small = g.nodes[:10]          # first 10 nodes → subgraph view
subset  = g.nodes[[1, 5, 9]]    # arbitrary indices
g_sub   = g.subgraph(nodes=subset.node_ids())  # explicit subgraph if you want it materialized
```

⸻

## 7) Delegation & chaining

Think in steps; write in steps. Keep it expressive and compact.

```python
(
    g.connected_components()
     .filter(lambda comp: len(comp.nodes) > 3)
     .table()
     .head()
)
```

We're keeping algorithms light in the README while the foundation settles, but the chain shows where it's headed.

⸻

## 8) Saving & loading

Save whole graphs as bundles, or export tables individually.

```python
# Full graph bundle
g.save_bundle("graph.bundle")
g2 = gr.GraphTable.load_bundle("graph.bundle")

# Tables as files
g.nodes.table().to_parquet("nodes.parquet")
g.edges.table().to_csv("edges.csv")
```

⸻

## 9) Core architecture (how it holds together)

Three tiers:
	•	**Rust Core** — memory-safe performance, columnar ops, and core graph/state machinery.
	•	**FFI Bridge** — a slim foreign-function interface that exposes Rust capabilities safely.
	•	**Python API** — the user-facing surface that reads like you think, with chains, masks, and friendly data interchange.

This separation keeps groggy fast, composable, and future-proof while still feeling Pythonic.

⸻

⸻

## Installation & Building

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
print("groggy installed successfully! 🎉")
```

⸻

## Development

### Project Structure
```
groggy/
├── src/                          # Rust core library
│   ├── algorithms/               # Algorithm registry, pipeline steps, builders
│   ├── api/                      # High-level graph API surface
│   ├── core/                     # Core data structures and engines
│   ├── temporal/                 # Temporal snapshots, index, context helpers
│   ├── viz/                      # Visualization adapters
│   └── …                         # (query, storage, traits, utils, etc.)
├── python-groggy/
│   ├── src/ffi/                  # PyO3 FFI bindings (marshalling only)
│   └── python/groggy/
│       ├── algorithms/           # Python algorithm shims & DSL
│       ├── builder.py            # Pipeline builder entry points
│       ├── graph.py              # Graph facade over the Rust core
│       └── …                     # Display helpers, viz, generators, widgets
├── docs/                         # MkDocs documentation site
├── notes/                        # Architecture plans, personas, experiments
└── tests/                        # Python integration + cross-language suites
```

### Building & Testing
```bash
# Build development version
maturin develop

# Build release version  
maturin develop --release

# Run formatting
cargo fmt

# Run Rust tests
cargo test

# Run Python tests
python tests/test_documentation_validation.py
```

⸻

## Contributing

groggy is open and evolving. We'd love your ideas, experiments, and critiques.
	•	Open an issue to propose features, UX improvements, or docs tweaks.
	•	Send a PR for bug fixes, performance wins, or new generators/datasets.
	•	Share how you're using groggy — real examples shape the roadmap.

Before contributing, please:
	•	Run the tests locally (unit + integration where applicable).
	•	Add/adjust docs for any user-visible change.
	•	Keep examples small and narrative-friendly (we teach by showing).

Thanks for being here. Let's build a graph engine that feels as natural as drawing a line between two ideas.

⸻

## License

MIT License - see [LICENSE](LICENSE) for details.

⸻

## Acknowledgments

groggy builds on the excellent work of:
- **Rust ecosystem**: Especially PyO3 for Python bindings
- **Graph libraries**: NetworkX, igraph, and others for inspiration  
- **Data science tools**: Pandas, NumPy for API design patterns
