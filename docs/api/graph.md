# Graph API Reference

**Type**: `groggy.Graph`

---

## Overview

The core mutable graph object containing nodes, edges, and attributes.

**Primary Use Cases:**
- Building and modifying graph structures
- Running graph algorithms
- Querying and filtering graph data

**Related Objects:**
- `Subgraph`
- `GraphTable`
- `NodesAccessor`
- `EdgesAccessor`

---

## Complete Method Reference

The following methods are available on `Graph` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `add_edge()` | `int` | ✓ |
| `add_edges()` | `?` | ✗ |
| `add_graph()` | `?` | ✗ |
| `add_node()` | `int` | ✓ |
| `add_nodes()` | `list` | ✓ |
| `aggregate()` | `AggregationResult` | ✓ |
| `all_edge_attribute_names()` | `list` | ✓ |
| `all_node_attribute_names()` | `list` | ✓ |
| `bfs()` | `Subgraph` | ✓ |
| `branches()` | `list` | ✓ |
| `checkout_branch()` | `?` | ✗ |
| `commit()` | `int` | ✓ |
| `commit_history()` | `list` | ✓ |
| `contains_edge()` | `bool` | ✓ |
| `contains_node()` | `bool` | ✓ |
| `create_branch()` | `?` | ✓ |
| `density()` | `float` | ✓ |
| `dfs()` | `Subgraph` | ✓ |
| `edge_attribute_keys()` | `list` | ✓ |
| `edge_count()` | `int` | ✓ |
| `edge_endpoints()` | `tuple` | ✓ |
| `edge_ids()` | `NumArray` | ✓ |
| `edges()` | `EdgesAccessor` | ✓ |
| `filter_edges()` | `Subgraph` | ✓ |
| `filter_nodes()` | `Subgraph` | ✓ |
| `get_edge_attr()` | `float` | ✓ |
| `get_edge_attrs()` | `?` | ✗ |
| `get_node_attr()` | `str` | ✓ |
| `get_node_attrs()` | `?` | ✗ |
| `get_node_mapping()` | `dict` | ✓ |
| `group_by()` | `GroupedAggregationResult` | ✓ |
| `group_nodes_by_attribute()` | `GroupedAggregationResult` | ✓ |
| `has_edge()` | `bool` | ✓ |
| `has_edge_attribute()` | `bool` | ✓ |
| `has_node()` | `bool` | ✓ |
| `has_node_attribute()` | `bool` | ✓ |
| `has_uncommitted_changes()` | `bool` | ✓ |
| `historical_view()` | `?` | ✗ |
| `is_connected()` | `bool` | ✓ |
| `is_directed()` | `bool` | ✓ |
| `is_empty()` | `bool` | ✓ |
| `is_undirected()` | `bool` | ✓ |
| `laplacian_matrix()` | `GraphMatrix` | ✓ |
| `neighborhood()` | `NeighborhoodResult` | ✓ |
| `neighborhood_statistics()` | `NeighborhoodStats` | ✓ |
| `neighbors()` | `?` | ✗ |
| `node_attribute_keys()` | `list` | ✓ |
| `node_count()` | `int` | ✓ |
| `node_ids()` | `NumArray` | ✓ |
| `nodes()` | `NodesAccessor` | ✓ |
| `remove_edge()` | `?` | ✓ |
| `remove_edges()` | `?` | ✗ |
| `remove_node()` | `?` | ✓ |
| `remove_nodes()` | `?` | ✗ |
| `resolve_string_id_to_node()` | `?` | ✗ |
| `set_edge_attr()` | `?` | ✗ |
| `set_edge_attrs()` | `?` | ✗ |
| `set_node_attr()` | `?` | ✗ |
| `set_node_attrs()` | `?` | ✗ |
| `shortest_path()` | `Subgraph` | ✓ |
| `table()` | `GraphTable` | ✓ |
| `to_matrix()` | `GraphMatrix` | ✓ |
| `to_networkx()` | `Graph` | ✓ |
| `transition_matrix()` | `?` | ✗ |
| `view()` | `Subgraph` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creation & Construction

#### `add_node(**attrs)`

Add a single node with optional attributes.

**Parameters:**
- `**attrs` (dict): Arbitrary keyword arguments become node attributes

**Returns:**
- `int`: The node ID

**Example:**
```python
g = gr.Graph()
n0 = g.add_node(name="Alice", age=29)
n1 = g.add_node(name="Bob", age=55)
print(f"Created nodes: {n0}, {n1}")  # 0, 1
```

**Notes:**
- Node IDs are auto-incrementing integers starting at 0
- All attribute values must be JSON-serializable
- Duplicate IDs are not allowed

---

#### `add_nodes(nodes_data)`

Add multiple nodes in bulk.

**Parameters:**
- `nodes_data` (list[dict]): List of dictionaries, each representing a node with its attributes

**Returns:**
- `list[int]`: List of created node IDs

**Example:**
```python
nodes = [
    {"name": "Alice", "age": 29},
    {"name": "Bob", "age": 55},
    {"name": "Carol", "age": 31}
]
ids = g.add_nodes(nodes)
print(ids)  # [0, 1, 2]
```

**Performance:**
- O(n) where n is number of nodes
- Much faster than individual `add_node()` calls
- Single FFI crossing for entire batch

---

#### `add_edge(source, target, **attrs)`

Add a single edge between two nodes.

**Parameters:**
- `source` (int): Source node ID
- `target` (int): Target node ID
- `**attrs` (dict): Optional edge attributes

**Returns:**
- `int`: The edge ID

**Example:**
```python
g = gr.Graph()
n0 = g.add_node(name="Alice")
n1 = g.add_node(name="Bob")
e = g.add_edge(n0, n1, weight=5.0, type="friend")
print(f"Created edge: {e}")  # 0
```

**Raises:**
- `ValueError`: If source or target node doesn't exist

**Notes:**
- Edge IDs are auto-incrementing starting at 0
- Self-loops are allowed
- Multiple edges between same nodes are allowed

---

#### `add_edges(edges_data)`

Add multiple edges in bulk.

**Parameters:**
- `edges_data` (list[tuple | dict]): List of (source, target) tuples or dicts with 'source', 'target', and optional attributes

**Returns:**
- `list[int]`: List of created edge IDs (implementation-dependent)

**Example:**
```python
# As tuples
edges = [(0, 1), (1, 2), (0, 2)]
g.add_edges(edges)

# With attributes
edges = [
    {"source": 0, "target": 1, "weight": 5.0},
    {"source": 1, "target": 2, "weight": 2.0}
]
g.add_edges(edges)
```

**Performance:**
- O(m) where m is number of edges
- Significantly faster than individual `add_edge()` calls

---

### Queries & Inspection

#### `node_count()`

Get the number of nodes in the graph.

**Returns:**
- `int`: Number of nodes

**Example:**
```python
g = gr.generators.karate_club()
print(g.node_count())  # 34
```

**Performance:** O(1)

---

#### `edge_count()`

Get the number of edges in the graph.

**Returns:**
- `int`: Number of edges

**Example:**
```python
g = gr.generators.karate_club()
print(g.edge_count())  # 78
```

**Performance:** O(1)

---

#### `contains_node(node_id)` / `has_node(node_id)`

Check if a node exists in the graph.

**Parameters:**
- `node_id` (int): Node ID to check

**Returns:**
- `bool`: True if node exists

**Example:**
```python
g = gr.Graph()
n = g.add_node(name="Alice")
print(g.has_node(n))     # True
print(g.has_node(999))   # False
```

**Notes:** Both methods are aliases with identical behavior

---

#### `contains_edge(edge_id)` / `has_edge(edge_id)`

Check if an edge exists in the graph.

**Parameters:**
- `edge_id` (int): Edge ID to check

**Returns:**
- `bool`: True if edge exists

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
e = g.add_edge(n0, n1)
print(g.has_edge(e))    # True
print(g.has_edge(999))  # False
```

---

#### `has_node_attribute(attr_name)`

Check if any node has the specified attribute.

**Parameters:**
- `attr_name` (str): Attribute name to check

**Returns:**
- `bool`: True if at least one node has this attribute

**Example:**
```python
g = gr.Graph()
g.add_node(name="Alice", age=29)
g.add_node(name="Bob")  # No age
print(g.has_node_attribute("age"))   # True
print(g.has_node_attribute("city"))  # False
```

---

#### `has_edge_attribute(attr_name)`

Check if any edge has the specified attribute.

**Parameters:**
- `attr_name` (str): Attribute name to check

**Returns:**
- `bool`: True if at least one edge has this attribute

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
g.add_edge(n0, n1, weight=5.0)
print(g.has_edge_attribute("weight"))  # True
print(g.has_edge_attribute("type"))    # False
```

---

#### `node_ids()`

Get all node IDs as an array.

**Returns:**
- `NumArray`: Array of all node IDs

**Example:**
```python
g = gr.Graph()
g.add_nodes([{}, {}, {}])
ids = g.node_ids()
print(ids.to_list())  # [0, 1, 2]
```

**Performance:** O(n)

---

#### `edge_ids()`

Get all edge IDs as an array.

**Returns:**
- `NumArray`: Array of all edge IDs

**Example:**
```python
g = gr.Graph()
n0, n1, n2 = g.add_nodes([{}, {}, {}])
g.add_edge(n0, n1)
g.add_edge(n1, n2)
ids = g.edge_ids()
print(ids.to_list())  # [0, 1]
```

**Performance:** O(m)

---

#### `get_node_attr(node_id, attr_name)`

Get a single attribute value from a node.

**Parameters:**
- `node_id` (int): Node ID
- `attr_name` (str): Attribute name

**Returns:**
- `Any`: Attribute value (type varies)

**Example:**
```python
g = gr.Graph()
n = g.add_node(name="Alice", age=29)
name = g.get_node_attr(n, "name")
age = g.get_node_attr(n, "age")
print(name, age)  # Alice 29
```

**Raises:**
- `KeyError`: If node doesn't exist or doesn't have that attribute

---

#### `get_edge_attr(edge_id, attr_name)`

Get a single attribute value from an edge.

**Parameters:**
- `edge_id` (int): Edge ID
- `attr_name` (str): Attribute name

**Returns:**
- `Any`: Attribute value

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
e = g.add_edge(n0, n1, weight=5.0)
weight = g.get_edge_attr(e, "weight")
print(weight)  # 5.0
```

**Raises:**
- `KeyError`: If edge doesn't exist or doesn't have that attribute

---

#### `all_node_attribute_names()`

Get all attribute names used by any node.

**Returns:**
- `list[str]`: List of unique attribute names

**Example:**
```python
g = gr.Graph()
g.add_node(name="Alice", age=29)
g.add_node(name="Bob", city="NYC")
attrs = g.all_node_attribute_names()
print(sorted(attrs))  # ['age', 'city', 'name']
```

---

#### `all_edge_attribute_names()`

Get all attribute names used by any edge.

**Returns:**
- `list[str]`: List of unique attribute names

**Example:**
```python
g = gr.Graph()
n0, n1, n2 = g.add_nodes([{}, {}, {}])
g.add_edge(n0, n1, weight=5.0)
g.add_edge(n1, n2, type="friend")
attrs = g.all_edge_attribute_names()
print(sorted(attrs))  # ['type', 'weight']
```

---

#### `edge_endpoints(edge_id)`

Get the source and target nodes of an edge.

**Parameters:**
- `edge_id` (int): Edge ID

**Returns:**
- `tuple[int, int]`: (source, target) node IDs

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
e = g.add_edge(n0, n1)
src, tgt = g.edge_endpoints(e)
print(f"{src} -> {tgt}")  # 0 -> 1
```

---

#### `density()`

Calculate graph density (ratio of actual to possible edges).

**Returns:**
- `float`: Density value between 0.0 and 1.0

**Example:**
```python
g = gr.generators.complete_graph(10)
print(g.density())  # 1.0

g = gr.generators.path_graph(10)
print(g.density())  # ~0.09 (9 edges, 45 possible)
```

**Formula:**
- Undirected: `2m / (n(n-1))`
- Directed: `m / (n(n-1))`

---

#### `is_empty()`

Check if graph has no nodes.

**Returns:**
- `bool`: True if graph has zero nodes

**Example:**
```python
g = gr.Graph()
print(g.is_empty())  # True
g.add_node()
print(g.is_empty())  # False
```

---

#### `is_connected()`

Check if graph is connected (all nodes reachable from any node).

**Returns:**
- `bool`: True if graph is connected

**Example:**
```python
g = gr.generators.path_graph(10)
print(g.is_connected())  # True

g.add_node()  # Isolated node
print(g.is_connected())  # False
```

**Performance:** O(V + E) - runs BFS internally

---

#### `is_directed()`

Check if graph is directed.

**Returns:**
- `bool`: True if directed

**Example:**
```python
g = gr.Graph(directed=True)
print(g.is_directed())  # True
```

---

#### `is_undirected()`

Check if graph is undirected.

**Returns:**
- `bool`: True if undirected

**Example:**
```python
g = gr.Graph()  # Default undirected
print(g.is_undirected())  # True
```

---

### Algorithms

#### `bfs(start)`

Breadth-first search from a starting node.

**Parameters:**
- `start` (int): Starting node ID

**Returns:**
- `Subgraph`: Subgraph containing all reachable nodes in BFS order

**Example:**
```python
g = gr.generators.karate_club()
reachable = g.bfs(start=0)
print(f"BFS from node 0 reached {reachable.node_count()} nodes")
```

**Performance:** O(V + E)

---

#### `dfs(start)`

Depth-first search from a starting node.

**Parameters:**
- `start` (int): Starting node ID

**Returns:**
- `Subgraph`: Subgraph containing all reachable nodes in DFS order

**Example:**
```python
g = gr.generators.karate_club()
reachable = g.dfs(start=0)
print(f"DFS from node 0 reached {reachable.node_count()} nodes")
```

**Performance:** O(V + E)

---

#### `shortest_path(source, target=None, weight=None)`

Find shortest path(s) from source node.

**Parameters:**
- `source` (int): Starting node ID
- `target` (int, optional): End node ID. If None, finds paths to all nodes
- `weight` (str, optional): Edge attribute to use as weight

**Returns:**
- `Subgraph`: Subgraph containing nodes on shortest paths

**Example:**
```python
# Unweighted shortest paths from node 0
paths = g.shortest_path(source=0)

# Weighted shortest path
g_weighted = gr.Graph()
# ... add edges with weight attribute ...
path = g_weighted.shortest_path(source=0, target=5, weight="weight")
```

**Performance:**
- Unweighted: O(V + E) via BFS
- Weighted: O((V + E) log V) via Dijkstra

---

#### `neighborhood(node, depth=1)`

Get k-hop neighborhood around a node.

**Parameters:**
- `node` (int): Center node ID
- `depth` (int): Number of hops (default 1)

**Returns:**
- `NeighborhoodResult` or `Subgraph`: Nodes within k hops

**Example:**
```python
# 2-hop neighborhood around node 0
nbh = g.neighborhood(node=0, depth=2)
print(f"2-hop neighborhood: {nbh.node_count()} nodes")
```

**Performance:** O(V + E) worst case, often much faster

---

### Transformations

#### `nodes`

Access nodes via NodesAccessor.

**Returns:**
- `NodesAccessor`: Accessor for node operations

**Example:**
```python
g = gr.generators.karate_club()

# Access via property
ages = g.nodes["age"]
young = g.nodes[g.nodes["age"] < 30]
first_five = g.nodes[:5]
```

**See:** [NodesAccessor API](nodesaccessor.md)

---

#### `edges`

Access edges via EdgesAccessor.

**Returns:**
- `EdgesAccessor`: Accessor for edge operations

**Example:**
```python
g = gr.generators.karate_club()

# Access via property
weights = g.edges["weight"]
heavy = g.edges[g.edges["weight"] > 5.0]
```

**See:** [EdgesAccessor API](edgesaccessor.md)

---

#### `table()`

Convert graph to tabular representation.

**Returns:**
- `GraphTable`: Table containing all graph data

**Example:**
```python
g = gr.generators.karate_club()
tbl = g.table()
print(tbl.head())

# Export to pandas
df = tbl.to_pandas()
```

**See:** [GraphTable API](graphtable.md)

---

#### `to_matrix()`

Convert graph to matrix representation.

**Returns:**
- `GraphMatrix`: Adjacency matrix

**Example:**
```python
g = gr.generators.karate_club()
A = g.to_matrix()
print(A.shape())  # (34, 34)
```

**Notes:** Alias for `adjacency_matrix()`

---

#### `laplacian_matrix()`

Get the graph Laplacian matrix.

**Returns:**
- `GraphMatrix`: Laplacian matrix (D - A)

**Example:**
```python
g = gr.generators.karate_club()
L = g.laplacian_matrix()

# Use for spectral methods
# eigenvalues, eigenvectors = scipy.linalg.eigh(L.data())
```

**Formula:** L = D - A, where D is degree matrix, A is adjacency

---

#### `to_networkx()`

Convert to NetworkX graph.

**Returns:**
- `networkx.Graph` or `networkx.DiGraph`: NetworkX representation

**Example:**
```python
import networkx as nx

g = gr.generators.karate_club()
G_nx = g.to_networkx()

# Use NetworkX algorithms
pagerank = nx.pagerank(G_nx)
```

---

### State Management

#### `remove_node(node_id)`

Remove a node and all its edges.

**Parameters:**
- `node_id` (int): Node ID to remove

**Returns:**
- `None`

**Example:**
```python
g = gr.Graph()
n = g.add_node(name="Alice")
g.remove_node(n)
print(g.node_count())  # 0
```

**Notes:** Also removes all edges connected to this node

---

#### `remove_edge(edge_id)`

Remove an edge.

**Parameters:**
- `edge_id` (int): Edge ID to remove

**Returns:**
- `None`

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
e = g.add_edge(n0, n1)
g.remove_edge(e)
print(g.edge_count())  # 0
```

---

### Filtering

#### `filter_nodes(condition)`

Filter to subgraph containing only matching nodes.

**Parameters:**
- `condition`: Node filter condition

**Returns:**
- `Subgraph`: Filtered subgraph

**Example:**
```python
g = gr.generators.karate_club()
young = g.filter_nodes(lambda n: g.get_node_attr(n, "age") < 30)
```

**Notes:** Prefer using `g.nodes[condition]` syntax

---

#### `filter_edges(condition)`

Filter to subgraph containing only matching edges.

**Parameters:**
- `condition`: Edge filter condition

**Returns:**
- `Subgraph`: Filtered subgraph

**Example:**
```python
g = gr.Graph()
# ... build graph with edge weights ...
heavy = g.filter_edges(lambda e: g.get_edge_attr(e, "weight") > 5.0)
```

**Notes:** Prefer using `g.edges[condition]` syntax

---

### Version Control

#### `commit(message="")`

Create a commit snapshot of current graph state.

**Parameters:**
- `message` (str, optional): Commit message

**Returns:**
- `int`: Commit ID

**Example:**
```python
g = gr.Graph()
g.add_nodes([{}, {}, {}])
commit_id = g.commit("Added initial nodes")
print(f"Commit: {commit_id}")
```

---

#### `commit_history()`

Get list of all commits.

**Returns:**
- `list`: List of commit records

**Example:**
```python
g = gr.Graph()
g.add_node()
g.commit("First")
g.add_node()
g.commit("Second")

history = g.commit_history()
print(f"{len(history)} commits")
```

---

#### `branches()`

Get list of all branches.

**Returns:**
- `list[str]`: Branch names

**Example:**
```python
g = gr.Graph()
g.create_branch("experiment")
print(g.branches())  # ['main', 'experiment']
```

---

#### `has_uncommitted_changes()`

Check if graph has uncommitted changes.

**Returns:**
- `bool`: True if there are uncommitted changes

**Example:**
```python
g = gr.Graph()
g.add_node()
print(g.has_uncommitted_changes())  # True
g.commit("Save")
print(g.has_uncommitted_changes())  # False
```

---

## Additional Methods

**Note:** The following methods are available but less commonly used. See method table above for complete list.

- `aggregate()` - Aggregate node/edge attributes
- `group_by()` / `group_nodes_by_attribute()` - Group nodes by attribute values
- `get_node_mapping()` - Get internal node ID mapping
- `neighborhood_statistics()` - Statistics about neighborhoods
- `view()` - Create a view of the graph

For full details on these methods, see the [User Guide](../guide/graph-core.md) or explore via `help(Graph)` in Python.

---

## Object Transformations

`Graph` can transform into:

- **Graph → Subgraph**: `g.nodes[condition]`, `g.subgraph(nodes=[...])`
- **Graph → GraphTable**: `g.table()`
- **Graph → NodesAccessor**: `g.nodes`
- **Graph → EdgesAccessor**: `g.edges`
- **Graph → ComponentsArray**: `g.connected_components()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/graph-core.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How Graph works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
#### `add_graph(other)`

Add Graph.

**Parameters:**
- `other`: other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.add_graph(other=...)
```

---

#### `aggregate()`

Aggregate.

**Returns:**
- `AggregationResult`: Return value

**Example:**
```python
obj.aggregate()
```

---

#### `checkout_branch()`

Checkout Branch.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.checkout_branch()
```

---

#### `create_branch()`

Create Branch.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.create_branch()
```

---

#### `edge_attribute_keys()`

Edge Attribute Keys.

**Returns:**
- `list`: Return value

**Example:**
```python
obj.edge_attribute_keys()
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

#### `get_edge_attrs()`

Get Edge Attrs.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_edge_attrs()
```

---

#### `get_node_attrs()`

Get Node Attrs.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_node_attrs()
```

---

#### `get_node_mapping()`

Get Node Mapping.

**Returns:**
- `dict`: Return value

**Example:**
```python
obj.get_node_mapping()
```

---

#### `group_by()`

Group By.

**Returns:**
- `GroupedAggregationResult`: Return value

**Example:**
```python
obj.group_by()
```

---

#### `group_nodes_by_attribute()`

Group Nodes By Attribute.

**Returns:**
- `GroupedAggregationResult`: Return value

**Example:**
```python
obj.group_nodes_by_attribute()
```

---

#### `has_edge()`

Has Edge.

**Returns:**
- `bool`: Return value

**Example:**
```python
obj.has_edge()
```

---

#### `has_node()`

Has Node.

**Returns:**
- `bool`: Return value

**Example:**
```python
obj.has_node()
```

---

#### `historical_view()`

Historical View.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.historical_view()
```

---

#### `neighbors()`

Neighbors.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.neighbors()
```

---

#### `node_attribute_keys()`

Node Attribute Keys.

**Returns:**
- `list`: Return value

**Example:**
```python
obj.node_attribute_keys()
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

#### `remove_edges()`

Remove Edges.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.remove_edges()
```

---

#### `remove_nodes()`

Remove Nodes.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.remove_nodes()
```

---

#### `resolve_string_id_to_node()`

Resolve String Id To Node.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.resolve_string_id_to_node()
```

---

#### `set_edge_attr()`

Set Edge Attr.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_edge_attr()
```

---

#### `set_edge_attrs()`

Set Edge Attrs.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_edge_attrs()
```

---

#### `set_node_attr()`

Set Node Attr.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_node_attr()
```

---

#### `set_node_attrs()`

Set Node Attrs.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set_node_attrs()
```

---

#### `transition_matrix()`

Transition Matrix.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.transition_matrix()
```

---

#### `view()`

View.

**Returns:**
- `Subgraph`: Return value

**Example:**
```python
obj.view()
```

---

