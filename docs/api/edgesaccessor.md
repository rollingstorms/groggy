# EdgesAccessor API Reference

**Type**: `groggy.EdgesAccessor`

---

## Overview

Accessor for edge-level operations and filtering on graphs and subgraphs.

**Primary Use Cases:**
- Filtering edges by attributes
- Accessing edge properties
- Creating edge-based subgraphs

**Related Objects:**
- `Graph`
- `Subgraph`
- `EdgesTable`
- `EdgesArray`

---

## Complete Method Reference

The following methods are available on `EdgesAccessor` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `all()` | `Subgraph` | ✓ |
| `array()` | `EdgesArray` | ✓ |
| `attribute_names()` | `list` | ✓ |
| `attributes()` | `list` | ✓ |
| `base()` | `EdgesAccessor` | ✓ |
| `filter()` | `?` | ✗ |
| `group_by()` | `SubgraphArray` | ✓ |
| `ids()` | `NumArray` | ✓ |
| `matrix()` | `GraphMatrix` | ✓ |
| `meta()` | `EdgesAccessor` | ✓ |
| `set_attrs()` | `?` | ✗ |
| `sources()` | `NumArray` | ✓ |
| `table()` | `EdgesTable` | ✓ |
| `targets()` | `NumArray` | ✓ |
| `viz()` | `VizAccessor` | ✓ |
| `weight_matrix()` | `GraphMatrix` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Accessing Edges

EdgesAccessor supports multiple access patterns similar to NodesAccessor:

#### Attribute Access: `g.edges["attribute"]`

Get a column of edge attribute values.

**Returns:**
- `BaseArray`: Array of attribute values

**Example:**
```python
g = gr.generators.karate_club()

# Get edge weights
weights = g.edges["weight"]
print(weights.head())

# Get edge types
types = g.edges["type"]
```

**Notes:**
- Returns all values for that attribute across all edges
- Missing values handled gracefully

---

#### Filtering: `g.edges[condition]`

Filter edges by condition to create subgraph.

**Returns:**
- `Subgraph`: Subgraph containing matching edges (and their endpoints)

**Example:**
```python
# Filter by weight
heavy = g.edges[g.edges["weight"] > 5.0]
print(f"{heavy.edge_count()} heavy edges")

# Multiple conditions
important = g.edges[
    (g.edges["weight"] > 3.0) &
    (g.edges["type"] == "critical")
]

# Filter by source/target
from_node_0 = g.edges[g.edges.sources() == 0]
```

**Notes:**
- Creates a view (no copying)
- Includes endpoint nodes in resulting subgraph
- Can filter on edge attributes or topological properties

---

#### Slicing: `g.edges[start:end]` or `g.edges[indices]`

Access edges by position or ID list.

**Returns:**
- `Subgraph`: Subgraph with selected edges

**Example:**
```python
# First 10 edges
first_ten_edges = g.edges[:10]

# Specific range
middle_edges = g.edges[10:20]

# By ID list
specific_edges = g.edges[[0, 5, 10]]

# Every other edge
every_other = g.edges[::2]
```

---

### Core Methods

#### `ids()`

Get array of all edge IDs.

**Returns:**
- `NumArray`: Edge IDs

**Example:**
```python
g = gr.generators.karate_club()
edge_ids = g.edges.ids()
print(f"{len(edge_ids)} edges")
```

**Performance:** O(m) where m is edge count

---

#### `sources()`

Get source node IDs for all edges.

**Returns:**
- `NumArray`: Source node IDs

**Example:**
```python
srcs = g.edges.sources()
print(srcs.head())
```

**Notes:**
- For directed graphs: actual source nodes
- For undirected: arbitrary endpoint designation

---

#### `targets()`

Get target node IDs for all edges.

**Returns:**
- `NumArray`: Target node IDs

**Example:**
```python
tgts = g.edges.targets()
print(tgts.head())

# Build edge list
import pandas as pd
edge_list = pd.DataFrame({
    'source': g.edges.sources().to_list(),
    'target': g.edges.targets().to_list()
})
```

---

#### `all()`

Get subgraph containing all edges.

**Returns:**
- `Subgraph`: Subgraph with all edges

**Example:**
```python
all_edges = g.edges.all()
print(all_edges.edge_count())  # Same as g.edge_count()
```

---

#### `attribute_names()` / `attributes()`

Get list of all edge attribute names.

**Returns:**
- `list[str]`: Attribute names

**Example:**
```python
g = gr.Graph()
n0, n1 = g.add_node(), g.add_node()
g.add_edge(n0, n1, weight=5.0, type="friend")

attrs = g.edges.attribute_names()
print(sorted(attrs))  # ['type', 'weight']
```

**Notes:** Both methods are aliases

---

#### `attributes()`

Get list of all edge attribute names (property/method).

**Returns:**
- `list[str]`: Attribute names

**Example:**
```python
g = gr.generators.karate_club()

# As property
attrs = g.edges.attributes()
print(sorted(attrs))  # All edge attribute names

# Same as attribute_names()
assert g.edges.attributes() == g.edges.attribute_names()
```

---

#### `table()`

Convert edges to tabular format.

**Returns:**
- `EdgesTable`: Table of all edge data

**Example:**
```python
edges_table = g.edges.table()

# View first rows
print(edges_table.head())

# Export to pandas
df = edges_table.to_pandas()
print(df[['source', 'target', 'weight']])
```

---

#### `array()`

Convert to EdgesArray representation.

**Returns:**
- `EdgesArray`: Array-based view of edges

**Example:**
```python
edges_arr = g.edges.array()
print(type(edges_arr))  # EdgesArray
```

---

#### `matrix()`

Create adjacency matrix from edges.

**Returns:**
- `GraphMatrix`: Adjacency matrix (binary)

**Example:**
```python
A = g.edges.matrix()
print(A.shape())  # (num_nodes, num_nodes)
```

**Notes:** Same as `g.adjacency_matrix()`

---

#### `weight_matrix()`

Create weighted adjacency matrix.

**Returns:**
- `GraphMatrix`: Matrix with edge weights

**Example:**
```python
g = gr.Graph()
n0, n1, n2 = g.add_nodes([{}, {}, {}])
g.add_edge(n0, n1, weight=5.0)
g.add_edge(n1, n2, weight=2.0)
g.add_edge(n0, n2, weight=1.0)

# Weighted matrix
W = g.edges.weight_matrix()
print(W.data())  # Contains actual weights, not just 0/1
```

**Notes:**
- Uses `weight` attribute by default
- Missing weights may be treated as 0 or 1

---

### Filtering

#### `filter(predicate)`

Filter edges by a predicate function or expression.

**Parameters:**
- `predicate` (callable or str): Filter expression

**Returns:**
- `Subgraph`: Filtered edges with their endpoints

**Example:**
```python
# Using callable
heavy = g.edges.filter(lambda edge: edge["weight"] > 5.0)

# Using string expression (if supported)
heavy = g.edges.filter("weight > 5.0")
```

**Notes:** Method signature requires predicate parameter

---

### Grouping and Aggregation

#### `group_by(attribute)`

Group edges by attribute value.

**Parameters:**
- `attribute` (str): Attribute name to group by

**Returns:**
- `SubgraphArray`: Array of subgraphs, one per group

**Example:**
```python
g = gr.Graph()
n0, n1, n2, n3 = g.add_nodes([{}, {}, {}, {}])
g.add_edge(n0, n1, type="friend", weight=5)
g.add_edge(n1, n2, type="coworker", weight=3)
g.add_edge(n2, n3, type="friend", weight=4)

# Group by edge type
by_type = g.edges.group_by("type")
print(f"{len(by_type)} edge types")

for group in by_type:
    print(f"Type: {group.edge_count()} edges")
```

**Use cases:**
- Analyzing edge types separately
- Computing per-group statistics
- Multi-relational graphs

---

### Attribute Updates

#### `set_attrs(attr_dict)`

Bulk update edge attributes.

**Parameters:**
- `attr_dict` (dict): Mapping of edge_id → {attr: value}

**Returns:**
- `None` (modifies graph in-place)

**Example:**
```python
g = gr.Graph()
n0, n1, n2 = g.add_nodes([{}, {}, {}])
e0 = g.add_edge(n0, n1)
e1 = g.add_edge(n1, n2)

# Bulk set attributes
g.edges.set_attrs({
    e0: {"weight": 5.0, "type": "strong"},
    e1: {"weight": 1.0, "type": "weak"}
})

# Verify
print(g.edges["weight"].to_list())  # [5.0, 1.0]
```

**Performance:** O(m) where m is number of updates

---

### Hierarchical Operations

#### `meta()`

Access meta-edges (for hierarchical graphs).

**Returns:**
- `EdgesAccessor`: Accessor for meta-edges only

**Example:**
```python
if g.has_meta_edges():
    meta_edges = g.edges.meta()
    print(f"{len(meta_edges.ids())} meta-edges")
```

---

#### `base()`

Access base-level edges (non-meta).

**Returns:**
- `EdgesAccessor`: Accessor for base edges only

**Example:**
```python
base_edges = g.edges.base()
print(f"{len(base_edges.ids())} base edges")
```

---

### Visualization

#### `viz()`

Access visualization methods.

**Returns:**
- `VizAccessor`: Visualization accessor

**Example:**
```python
viz = g.edges.viz()
# Use viz methods for plotting
```

---

## Usage Patterns

### Pattern 1: Filter by Weight Threshold

```python
# Keep only strong connections
strong = g.edges[g.edges["weight"] > 5.0]
print(f"Strong subgraph: {strong.node_count()} nodes, {strong.edge_count()} edges")

# Analyze filtered network
density = strong.density()
```

### Pattern 2: Edge Type Analysis

```python
# Group by type
by_type = g.edges.group_by("type")

# Analyze each type
for edge_type in by_type:
    weights = edge_type.edges["weight"]
    print(f"Type avg weight: {weights.mean():.2f}")
```

### Pattern 3: Extract Edge List

```python
# Get edge list with attributes
sources = g.edges.sources().to_list()
targets = g.edges.targets().to_list()
weights = g.edges["weight"].to_list()

# Create structured edge list
import pandas as pd
edge_df = pd.DataFrame({
    'src': sources,
    'tgt': targets,
    'weight': weights
})
```

### Pattern 4: Filter by Topology

```python
# Edges from high-degree nodes
degrees = g.degree()
high_degree_nodes = set(g.nodes[degrees > 10].node_ids().to_list())

# Filter edges
srcs = g.edges.sources()
edges_from_hubs = g.edges[[
    s in high_degree_nodes for s in srcs.to_list()
]]
```

### Pattern 5: Weight Normalization

```python
# Get weights
weights = g.edges["weight"]
max_weight = weights.max()

# Normalize and set back
g.edges.set_attrs({
    int(eid): {"normalized_weight": float(w / max_weight)}
    for eid, w in zip(g.edges.ids(), weights)
})

# Now use normalized weights
normalized = g.edges["normalized_weight"]
```

### Pattern 6: Edge Direction Analysis

```python
# For directed graphs
srcs = g.edges.sources()
tgts = g.edges.targets()

# Count edges per source
from collections import Counter
out_edges = Counter(srcs.to_list())
print(f"Node 0 has {out_edges[0]} outgoing edges")

# Find reciprocal edges
edge_set = set(zip(srcs.to_list(), tgts.to_list()))
reciprocal = [
    (s, t) for s, t in edge_set
    if (t, s) in edge_set
]
print(f"{len(reciprocal)} reciprocal edge pairs")
```

---

## Access Syntax Summary

| Syntax | Returns | Example |
|--------|---------|---------|
| `g.edges["attr"]` | `BaseArray` | `weights = g.edges["weight"]` |
| `g.edges[condition]` | `Subgraph` | `heavy = g.edges[g.edges["weight"] > 5]` |
| `g.edges[:10]` | `Subgraph` | `first_ten = g.edges[:10]` |
| `g.edges[[0,5,10]]` | `Subgraph` | `specific = g.edges[[0,5,10]]` |
| `g.edges.ids()` | `NumArray` | `all_ids = g.edges.ids()` |
| `g.edges.sources()` | `NumArray` | `srcs = g.edges.sources()` |
| `g.edges.targets()` | `NumArray` | `tgts = g.edges.targets()` |
| `g.edges.table()` | `EdgesTable` | `tbl = g.edges.table()` |

---

## Performance Considerations

**Efficient:**
- Attribute access: `g.edges["weight"]` - O(m) columnar scan
- Filtering: `g.edges[condition]` - Creates view, lazy evaluation
- Slicing: `g.edges[:10]` - O(1) view creation
- Source/target access: O(m) optimized array operations

**Less Efficient:**
- Individual edge lookups in loops - prefer bulk operations
- Repeated attribute access - cache the array
- Converting to table repeatedly - convert once, reuse

**Best Practices:**
```python
# ✅ Good: bulk operations
weights = g.edges["weight"]
avg_weight = weights.mean()

# ❌ Avoid: loops over individual edges
total = 0
for eid in g.edges.ids():
    total += g.get_edge_attr(eid, "weight")

# ✅ Good: vectorized edge list construction
edge_list = list(zip(
    g.edges.sources().to_list(),
    g.edges.targets().to_list()
))

# ❌ Avoid: loop-based construction
edge_list = []
for eid in g.edges.ids():
    src, tgt = g.edge_endpoints(eid)
    edge_list.append((src, tgt))
```

---

## Differences from NodesAccessor

While similar in design, EdgesAccessor has unique methods:

| Feature | NodesAccessor | EdgesAccessor |
|---------|---------------|---------------|
| IDs | `nodes.ids()` | `edges.ids()` |
| Topology | N/A | `edges.sources()`, `edges.targets()` |
| Matrix | `nodes.matrix()` (features) | `edges.matrix()` (adjacency) |
| Weighted matrix | N/A | `edges.weight_matrix()` |
| Grouping | `nodes.group_by(attr)` | `edges.group_by(attr)` |

**Key difference:** Edge filtering includes endpoint nodes in resulting subgraph


---

## Object Transformations

`EdgesAccessor` can transform into:

- **EdgesAccessor → Subgraph**: `g.edges[condition]`
- **EdgesAccessor → EdgesTable**: `g.edges.table()`
- **EdgesAccessor → EdgesArray**: `g.edges.ids()`
- **EdgesAccessor → BaseArray**: `g.edges["attribute"]`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/accessors.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How EdgesAccessor works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
