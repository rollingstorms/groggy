# Groggy for Agents & LLMs

Groggy is designed for seamless integration with intelligent agents and LLMs, enabling:
- Dynamic graph construction and querying
- Fast, cold-start-friendly context setup
- Intuitive, explainable API for code generation and reasoning

---

## Agent/LLM Usage Patterns & Comprehensive Examples

### 1. Fast Graph Instantiation
```python
from groggy import Graph
G = Graph(directed=True)
G2 = Graph(directed=False, backend='rust')
```

### 2. Bulk Operations for Efficiency
```python
G.nodes.add({'n1': {'type': 'A'}, 'n2': {'type': 'B'}})
G.nodes.add(['n3', 'n4'])
G.edges.add([('n1', 'n2', {'weight': 1.0}), ('n2', 'n3')])
G.edges.add(('n3', 'n4'))
G.nodes.remove(['n4'])
G.edges.remove([('n3', 'n4')])
```

### 3. Attribute Management
```python
G.nodes.attr.set({'n1': {'score': 0.9, 'type': 'root'}, 'n2': {'score': 0.5}})
G.nodes.attr.set_type('score', float)
attrs = G.nodes.attr.get(['n1', 'n2'], ['score'])
G.edges.attr.set({('n1', 'n2'): {'weight': 2.0}})
G.edges.attr.set_type('weight', float)
edge_attrs = G.edges.attr.get([('n1', 'n2')], ['weight'])
```

### 4. Filtering & Subgraphs
```python
filtered_nodes = G.nodes.filter(type='A')
filtered_edges = G.edges.filter(weight__gt=0.5)
subG = G.subgraph(node_filter='type=="A"', edge_filter='weight>0.5')
subGs = G.subgraphs(node_filter='type')  # Split by attribute
```

### 5. Node & Edge Proxies
```python
node = G.nodes['n1']
degree = node.degree(direction='out')
neigh = node.neighbors(direction='in')
attrs = node.attrs()
node.set_attr({'score': 1.0})
attr_val = node.get_attr('score')
edge = G.edges[('n1', 'n2')]
endpoints = edge.endpoints()
edge.set_attr({'weight': 3.0})
```

### 6. Iteration, Indexing, and Metadata
```python
for node in G.nodes:
    print(node.attrs())
for edge in G.edges:
    print(edge.attrs())
all_node_ids = G.nodes.ids()
all_edge_ids = G.edges.ids()
print(len(G.nodes), len(G.edges))
info = G.info()
```

### 7. Algorithms for Reasoning & Analytics
```python
paths = G.shortest_path('n1', 'n2')
components = G.connected_components()
clustering = G.clustering_coefficient('n1')
pagerank = G.pagerank()
betweenness = G.betweenness_centrality()
labels = G.label_propagation_algorithm()
louvain = G.louvain()
modularity = G.modularity()
```

### 8. Provenance, Snapshots, Branching & State Tracking
```python
snap = G.snapshot()
G.save_state('experiment-1')
G2 = G.load_state('experiment-1')
G.create_branch('new-branch')
G.switch_branch('new-branch')
changes = G.show_changes('experiment-1')
entity_changes = G.show_entity_changes('n1', 'experiment-1')
G.track_attribute_changes('score', 'experiment-1')
```

### 9. Utilities & Interoperability
```python
from groggy.utils import create_random_graph, convert_networkx_graph
G3 = create_random_graph(100, 0.1, use_rust=True)
import networkx as nx
nxG = nx.Graph()
G4 = convert_networkx_graph(nxG)
nxG2 = G.to_networkx()
pd_df = G.to_pandas()
torch_data = G.to_pytorch()
```

### 10. Error Handling & Validation
```python
try:
    G.nodes.add({'n1': {'score': 'not_a_float'}})
except ValueError as e:
    print('Validation error:', e)

try:
    G.nodes.remove(['nonexistent'])
except KeyError:
    print('Node not found')
```

### 11. ML/Data Science & Multi-Agent Workflows
```python
# ML: graph for GNN training
graph = create_random_graph(1000, 0.05)
features = graph.nodes.attr.get(attr_names=['feature1', 'feature2'])
labels = graph.nodes.attr.get(attr_names=['label'])
# Multi-agent: split graph for parallel agent reasoning
subgraphs = graph.subgraphs(node_filter='role')
for subG in subgraphs:
    agent_result = agent.process(subG)
```

### 12. Best Practices for Agents/LLMs
- Use `.info()` for metadata/context
- Prefer batch ops for speed
- Leverage attribute/type validation
- Use subgraphs/snapshots for context isolation
- Track provenance for reproducibility
- Always handle exceptions gracefully
- Use high-level API for code generation
- Prefer Rust backend for large graphs

---

Groggy is built to make agent-based graph reasoning, planning, analytics, and ML/AI integration as frictionless and robust as possible.
G.save_state('experiment-1')
```

---

## Agent Integration Guidelines
- Use batch operations for speed and atomicity
- Leverage attribute manager for schema enforcement
- Use subgraphs and snapshots for context isolation
- Prefer high-level API for LLM code generation
- Use provenance/state features for reproducibility

---

## LLM/Agent Best Practices
- Always check `.info()` for graph/collection metadata
- Use `.filter()` and `.attr.get()` for efficient queries
- Avoid direct storage manipulation—use provided API
- For large graphs, use Rust backend for optimal speed

---

## Example: LLM Workflow
```python
from groggy import Graph
G = Graph()
G.nodes.add({'n1': {'role': 'root'}, 'n2': {'role': 'leaf'}})
subG = G.subgraph(node_filter='role=="root"')
paths = G.bfs('n1')
```

---

Groggy is built to make agent-based graph reasoning, planning, and analytics as frictionless as possible.
