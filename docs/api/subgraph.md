# Subgraph API Reference

**Type**: `groggy.Subgraph`

---

## Overview

An immutable view into a subset of a Graph without copying data.

**Primary Use Cases:**
- Filtering nodes/edges by conditions
- Working with portions of large graphs
- Creating temporary working sets without copying

**Related Objects:**
- `Graph`
- `SubgraphArray`
- `GraphTable`

---

## Complete Method Reference

The following methods are available on `Subgraph` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `adj()` | `GraphMatrix` | ✓ |
| `adjacency_list()` | `dict` | ✓ |
| `adjacency_matrix()` | `GraphMatrix` | ✓ |
| `bfs()` | `?` | ✗ |
| `calculate_similarity()` | `?` | ✗ |
| `child_meta_nodes()` | `list` | ✓ |
| `clustering_coefficient()` | `?` | ✗ |
| `collapse()` | `?` | ✗ |
| `connected_components()` | `ComponentsArray` | ✓ |
| `contains_edge()` | `?` | ✗ |
| `contains_node()` | `?` | ✗ |
| `degree()` | `NumArray` | ✓ |
| `density()` | `float` | ✓ |
| `dfs()` | `?` | ✗ |
| `edge_count()` | `int` | ✓ |
| `edge_endpoints()` | `?` | ✗ |
| `edge_ids()` | `NumArray` | ✓ |
| `edges()` | `EdgesAccessor` | ✓ |
| `edges_table()` | `EdgesTable` | ✓ |
| `entity_type()` | `str` | ✓ |
| `filter_edges()` | `?` | ✗ |
| `filter_nodes()` | `?` | ✗ |
| `get_edge_attribute()` | `?` | ✗ |
| `get_node_attribute()` | `?` | ✗ |
| `group_by()` | `?` | ✗ |
| `has_edge()` | `?` | ✗ |
| `has_edge_between()` | `?` | ✗ |
| `has_meta_nodes()` | `bool` | ✓ |
| `has_node()` | `?` | ✗ |
| `has_path()` | `?` | ✗ |
| `hierarchy_level()` | `int` | ✓ |
| `in_degree()` | `NumArray` | ✓ |
| `induced_subgraph()` | `?` | ✗ |
| `intersect_with()` | `?` | ✗ |
| `is_connected()` | `bool` | ✓ |
| `is_empty()` | `bool` | ✓ |
| `merge_with()` | `?` | ✗ |
| `meta_nodes()` | `list` | ✓ |
| `neighborhood()` | `SubgraphArray` | ✓ |
| `neighbors()` | `?` | ✗ |
| `node_count()` | `int` | ✓ |
| `node_ids()` | `NumArray` | ✓ |
| `nodes()` | `NodesAccessor` | ✓ |
| `out_degree()` | `NumArray` | ✓ |
| `parent_meta_node()` | `?` | ✓ |
| `sample()` | `Subgraph` | ✓ |
| `set_edge_attrs()` | `?` | ✗ |
| `set_node_attrs()` | `?` | ✗ |
| `shortest_path_subgraph()` | `?` | ✗ |
| `subgraph_from_edges()` | `?` | ✗ |
| `subtract_from()` | `?` | ✗ |
| `summary()` | `str` | ✓ |
| `table()` | `GraphTable` | ✓ |
| `to_edges()` | `EdgesAccessor` | ✓ |
| `to_graph()` | `Graph` | ✓ |
| `to_matrix()` | `GraphMatrix` | ✓ |
| `to_networkx()` | `?` | ✗ |
| `to_nodes()` | `NodesAccessor` | ✓ |
| `transitivity()` | `?` | ✗ |
| `viz()` | `VizAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Queries & Inspection

#### `node_count()`

Get the number of nodes in the subgraph.

**Returns:**
- `int`: Number of nodes

**Example:**
```python
g = gr.generators.karate_club()
young = g.nodes[g.nodes['age'] < 30]
print(young.node_count())
```

**Performance:** O(1) for views, O(n) if filter needs evaluation

---

#### `edge_count()`

Get the number of edges in the subgraph.

**Returns:**
- `int`: Number of edges (only edges between included nodes)

**Example:**
```python
g = gr.generators.karate_club()
core = g.nodes[:10]
print(core.edge_count())  # Edges within first 10 nodes
```

---

#### `node_ids()`

Get all node IDs in the subgraph.

**Returns:**
- `NumArray`: Array of node IDs

**Example:**
```python
young = g.nodes[g.nodes['age'] < 30]
ids = young.node_ids()
print(ids.to_list())
```

---

#### `edge_ids()`

Get all edge IDs in the subgraph.

**Returns:**
- `NumArray`: Array of edge IDs

**Example:**
```python
young = g.nodes[g.nodes['age'] < 30]
edge_ids = young.edge_ids()
print(f"{len(edge_ids)} edges in subgraph")
```

---

#### `degree()`

Calculate degree for each node in subgraph.

**Returns:**
- `NumArray`: Degree values (based on edges within subgraph only)

**Example:**
```python
sub = g.nodes[:10]
degrees = sub.degree()
print(f"Mean degree in subgraph: {degrees.mean():.2f}")
```

**Notes:** Only counts edges within the subgraph

---

#### `in_degree()`

Calculate in-degree for directed graphs.

**Returns:**
- `NumArray`: In-degree values

**Example:**
```python
g = gr.Graph(directed=True)
# ... build graph ...
sub = g.nodes[:10]
in_deg = sub.in_degree()
```

---

#### `out_degree()`

Calculate out-degree for directed graphs.

**Returns:**
- `NumArray`: Out-degree values

**Example:**
```python
sub = g.nodes[:10]
out_deg = sub.out_degree()
```

---

#### `density()`

Calculate density of the subgraph.

**Returns:**
- `float`: Density (0.0 to 1.0)

**Example:**
```python
young = g.nodes[g.nodes['age'] < 30]
print(f"Density: {young.density():.3f}")
```

**Formula:** Same as Graph: `2m / (n(n-1))` for undirected

---

#### `is_empty()`

Check if subgraph has no nodes.

**Returns:**
- `bool`: True if empty

**Example:**
```python
filtered = g.nodes[g.nodes['age'] > 1000]  # No matches
print(filtered.is_empty())  # True
```

---

#### `is_connected()`

Check if subgraph is connected.

**Returns:**
- `bool`: True if all nodes reachable from any node

**Example:**
```python
sub = g.nodes[:20]
if sub.is_connected():
    print("Subgraph is connected")
```

---

#### `adjacency_matrix()` / `adj()`

Get adjacency matrix for the subgraph.

**Returns:**
- `GraphMatrix`: Adjacency matrix

**Example:**
```python
sub = g.nodes[:10]
A = sub.adjacency_matrix()
print(A.shape())  # (10, 10)
```

**Notes:** `adj()` is an alias for `adjacency_matrix()`

---

#### `adjacency_list()`

Get adjacency list representation.

**Returns:**
- `dict`: Mapping of node_id → list of neighbor IDs

**Example:**
```python
sub = g.nodes[:5]
adj_list = sub.adjacency_list()
for node, neighbors in adj_list.items():
    print(f"Node {node}: {neighbors}")
```

---

#### `summary()`

Get a text summary of the subgraph.

**Returns:**
- `str`: Summary string

**Example:**
```python
sub = g.nodes[:10]
print(sub.summary())
# "Subgraph: 10 nodes, 15 edges, density=0.333"
```

---

### Algorithms

#### `connected_components()`

Find connected components within the subgraph.

**Returns:**
- `ComponentsArray`: Array of subgraphs, one per component

**Example:**
```python
sub = g.nodes[:20]
components = sub.connected_components()
print(f"{len(components)} components")

for i, comp in enumerate(components):
    print(f"Component {i}: {comp.node_count()} nodes")
```

---

#### `neighborhood(depth=1)`

Get k-hop neighborhoods for all nodes in subgraph.

**Parameters:**
- `depth` (int): Number of hops (default 1)

**Returns:**
- `SubgraphArray`: Array of neighborhood subgraphs

**Example:**
```python
sub = g.nodes[:5]
neighborhoods = sub.neighborhood(depth=2)

for i, nbh in enumerate(neighborhoods):
    print(f"Node {i}: {nbh.node_count()} nodes in 2-hop neighborhood")
```

---

### Transformations

#### `nodes`

Access nodes via NodesAccessor.

**Returns:**
- `NodesAccessor`: Accessor for node operations on this subgraph

**Example:**
```python
sub = g.nodes[:20]
young_in_sub = sub.nodes[sub.nodes['age'] < 30]
```

---

#### `edges`

Access edges via EdgesAccessor.

**Returns:**
- `EdgesAccessor`: Accessor for edge operations on this subgraph

**Example:**
```python
sub = g.nodes[:20]
heavy_edges = sub.edges[sub.edges['weight'] > 5.0]
```

---

#### `table()`

Convert subgraph to table representation.

**Returns:**
- `GraphTable`: Table containing subgraph data

**Example:**
```python
sub = g.nodes[:10]
tbl = sub.table()
df = tbl.to_pandas()
```

---

#### `to_graph()`

Materialize subgraph as a new independent Graph.

**Returns:**
- `Graph`: New graph containing copy of subgraph data

**Example:**
```python
young = g.nodes[g.nodes['age'] < 30]
young_graph = young.to_graph()

# Now independent from original
young_graph.add_node(age=25)  # Doesn't affect g
```

**Performance:** O(V + E) - copies data

**Notes:**
- Creates independent copy
- Use when you need to modify filtered data
- Keep as Subgraph (view) when possible for performance

---

#### `to_matrix()`

Convert subgraph to adjacency matrix.

**Returns:**
- `GraphMatrix`: Adjacency matrix

**Example:**
```python
sub = g.nodes[:10]
A = sub.to_matrix()
```

---

#### `sample(n)`

Randomly sample nodes from the subgraph.

**Parameters:**
- `n` (int): Number of nodes to sample

**Returns:**
- `Subgraph`: New subgraph with sampled nodes

**Example:**
```python
large_sub = g.nodes[:1000]
sample = large_sub.sample(100)
print(f"Sampled {sample.node_count()} nodes")
```

---

#### `to_nodes()` / `to_edges()`

Convert to NodesAccessor or EdgesAccessor.

**Returns:**
- `NodesAccessor` or `EdgesAccessor`: Accessor for elements

**Example:**
```python
sub = g.nodes[:10]
nodes = sub.to_nodes()
edges = sub.to_edges()
```

**Notes:** Aliases for `.nodes` and `.edges` properties

---

### Meta-Graph Operations

#### `has_meta_nodes()`

Check if subgraph contains meta-nodes (hierarchical graphs).

**Returns:**
- `bool`: True if meta-nodes present

**Example:**
```python
if sub.has_meta_nodes():
    print("This is a hierarchical subgraph")
```

---

#### `meta_nodes()`

Get list of meta-nodes in subgraph.

**Returns:**
- `list`: Meta-node IDs

**Example:**
```python
if sub.has_meta_nodes():
    meta = sub.meta_nodes()
    print(f"{len(meta)} meta-nodes")
```

---

#### `hierarchy_level()`

Get hierarchy level of this subgraph.

**Returns:**
- `int`: Level in hierarchy (0 = base level)

**Example:**
```python
level = sub.hierarchy_level()
print(f"At hierarchy level {level}")
```

---

#### `entity_type()`

Get the entity type identifier.

**Returns:**
- `str`: Entity type name

**Example:**
```python
etype = sub.entity_type()
print(f"Entity type: {etype}")
```

---

## Subgraph Creation

Subgraphs are typically created via filtering, not direct construction:

```python
# Via NodesAccessor slicing
sub = g.nodes[:10]
sub = g.nodes[g.nodes['age'] < 30]

# Via EdgesAccessor
sub = g.edges[g.edges['weight'] > 5.0]

# Via algorithms
components = g.connected_components()  # Returns SubgraphArray
largest = components[0]  # Individual Subgraph
```

See [Subgraphs Guide](../guide/subgraphs.md) for complete creation patterns.

---

## Views vs Materialization

**Subgraphs are views by default:**
- No data copying
- Filters evaluated on access
- Memory efficient
- Changes to Graph affect Subgraph

**Materialize when:**
- Need to modify filtered data
- Repeatedly accessing same subgraph
- Want independence from original graph

```python
# View (fast, no copy)
view = g.nodes[:100]

# Materialize (slow, independent)
copy = view.to_graph()
```

See [Performance Guide](../guide/performance.md) for optimization strategies.

---

## Additional Methods

The following specialized methods are available:

- **`edges_table()`** - Get EdgesTable for subgraph edges
- **`child_meta_nodes()`** - Get child meta-nodes in hierarchy
- **`parent_meta_node()`** - Get parent meta-node
- **`viz()`** - Access visualization methods

For full details, see the [User Guide](../guide/subgraphs.md).

---

## Object Transformations

`Subgraph` can transform into:

- **Subgraph → Graph**: `sub.to_graph()`
- **Subgraph → GraphTable**: `sub.table()`
- **Subgraph → NodesAccessor**: `sub.nodes`
- **Subgraph → EdgesAccessor**: `sub.edges`
- **Subgraph → GraphMatrix**: `sub.to_matrix()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/subgraphs.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How Subgraph works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
#### `bfs(start)`

Bfs.

**Parameters:**
- `start`: start

**Returns:**
- `None`: Return value

**Example:**
```python
obj.bfs(start=...)
```

---

#### `calculate_similarity(other)`

Calculate Similarity.

**Parameters:**
- `other`: other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.calculate_similarity(other=...)
```

---

#### `child_meta_nodes()`

Child Meta Nodes.

**Returns:**
- `list`: Return value

**Example:**
```python
obj.child_meta_nodes()
```

---

#### `clustering_coefficient()`

Clustering Coefficient.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.clustering_coefficient()
```

---

#### `collapse()`

Collapse.

**Returns:**
- `MetaNode`: Return value

**Example:**
```python
obj.collapse()
```

---

#### `contains_edge(edge_id)`

Contains Edge.

**Parameters:**
- `edge_id`: edge id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.contains_edge(edge_id=...)
```

---

#### `contains_node(node_id)`

Contains Node.

**Parameters:**
- `node_id`: node id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.contains_node(node_id=...)
```

---

#### `dfs(start)`

Dfs.

**Parameters:**
- `start`: start

**Returns:**
- `None`: Return value

**Example:**
```python
obj.dfs(start=...)
```

---

#### `edge_endpoints(edge_id)`

Edge Endpoints.

**Parameters:**
- `edge_id`: edge id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.edge_endpoints(edge_id=...)
```

---

#### `edges()`

Edges.

**Returns:**
- `EdgesAccessor`: Return value

**Example:**
```python
obj.edges()
```

---

#### `edges_table()`

Edges Table.

**Returns:**
- `EdgesTable`: Return value

**Example:**
```python
obj.edges_table()
```

---

#### `filter_edges(filter)`

Filter Edges.

**Parameters:**
- `filter`: filter

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter_edges(filter=...)
```

---

#### `filter_nodes(filter)`

Filter Nodes.

**Parameters:**
- `filter`: filter

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter_nodes(filter=...)
```

---

#### `get_edge_attribute(attr_name)`

Get Edge Attribute.

**Parameters:**
- `attr_name`: attr name

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_edge_attribute(attr_name=...)
```

---

#### `get_node_attribute(attr_name)`

Get Node Attribute.

**Parameters:**
- `attr_name`: attr name

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_node_attribute(attr_name=...)
```

---

#### `group_by(element_type)`

Group By.

**Parameters:**
- `element_type`: element type

**Returns:**
- `None`: Return value

**Example:**
```python
obj.group_by(element_type=...)
```

---

#### `has_edge(edge_id)`

Has Edge.

**Parameters:**
- `edge_id`: edge id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.has_edge(edge_id=...)
```

---

#### `has_edge_between(source, target)`

Has Edge Between.

**Parameters:**
- `source`: source
- `target`: target

**Returns:**
- `None`: Return value

**Example:**
```python
obj.has_edge_between(source=..., target=...)
```

---

#### `has_node(node_id)`

Has Node.

**Parameters:**
- `node_id`: node id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.has_node(node_id=...)
```

---

#### `has_path(node1_id, node2_id)`

Has Path.

**Parameters:**
- `node1_id`: node1 id
- `node2_id`: node2 id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.has_path(node1_id=..., node2_id=...)
```

---

#### `induced_subgraph(nodes)`

Induced Subgraph.

**Parameters:**
- `nodes`: nodes

**Returns:**
- `None`: Return value

**Example:**
```python
obj.induced_subgraph(nodes=...)
```

---

#### `intersect_with(_other)`

Intersect With.

**Parameters:**
- `_other`:  other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.intersect_with(_other=...)
```

---

#### `merge_with(_other)`

Merge With.

**Parameters:**
- `_other`:  other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.merge_with(_other=...)
```

---

#### `neighbors(node_id)`

Neighbors.

**Parameters:**
- `node_id`: node id

**Returns:**
- `None`: Return value

**Example:**
```python
obj.neighbors(node_id=...)
```

---

#### `nodes()`

Nodes.

**Returns:**
- `NodesAccessor`: Return value

**Example:**
```python
obj.nodes()
```

---

#### `parent_meta_node()`

Parent Meta Node.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.parent_meta_node()
```

---

#### `set_edge_attrs(attrs_dict)`

Set Edge Attrs.

**Parameters:**
- `attrs_dict`: attrs dict

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_edge_attrs(attrs_dict=...)
```

---

#### `set_node_attrs(attrs_dict)`

Set Node Attrs.

**Parameters:**
- `attrs_dict`: attrs dict

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_node_attrs(attrs_dict=...)
```

---

#### `shortest_path_subgraph(source, target)`

Shortest Path Subgraph.

**Parameters:**
- `source`: source
- `target`: target

**Returns:**
- `None`: Return value

**Example:**
```python
obj.shortest_path_subgraph(source=..., target=...)
```

---

#### `subgraph_from_edges(edges)`

Subgraph From Edges.

**Parameters:**
- `edges`: edges

**Returns:**
- `None`: Return value

**Example:**
```python
obj.subgraph_from_edges(edges=...)
```

---

#### `subtract_from(_other)`

Subtract From.

**Parameters:**
- `_other`:  other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.subtract_from(_other=...)
```

---

#### `to_edges()`

To Edges.

**Returns:**
- `EdgesAccessor`: Return value

**Example:**
```python
obj.to_edges()
```

---

#### `to_networkx()`

To Networkx.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_networkx()
```

---

#### `transitivity()`

Transitivity.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.transitivity()
```

---

#### `viz()`

Viz.

**Returns:**
- `VizAccessor`: Return value

**Example:**
```python
obj.viz()
```

---

