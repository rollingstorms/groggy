# NodesAccessor API Reference

**Type**: `groggy.NodesAccessor`

---

## Overview

Accessor for node-level operations and filtering on graphs and subgraphs.

**Primary Use Cases:**
- Filtering nodes by attributes
- Accessing node properties
- Creating node-based subgraphs

**Related Objects:**
- `Graph`
- `Subgraph`
- `NodesTable`
- `NodesArray`

---

## Complete Method Reference

The following methods are available on `NodesAccessor` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `all()` | `Subgraph` | ✓ |
| `array()` | `NodesArray` | ✓ |
| `attribute_names()` | `list` | ✓ |
| `attributes()` | `list` | ✓ |
| `base()` | `NodesAccessor` | ✓ |
| `filter()` | `?` | ✗ |
| `get_meta_node()` | `?` | ✗ |
| `group_by()` | `SubgraphArray` | ✓ |
| `ids()` | `NumArray` | ✓ |
| `matrix()` | `GraphMatrix` | ✓ |
| `meta()` | `NodesAccessor` | ✓ |
| `set_attrs()` | `?` | ✗ |
| `subgraphs()` | `NumArray` | ✓ |
| `table()` | `NodesTable` | ✓ |
| `viz()` | `VizAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Accessing Nodes

NodesAccessor supports multiple access patterns via special syntax:

#### Attribute Access: `g.nodes["attribute"]`

Get a column of node attribute values.

**Returns:**
- `BaseArray`: Array of attribute values

**Example:**
```python
g = gr.generators.karate_club()

# Get single attribute
ages = g.nodes["age"]
print(ages.head())

# Type of result
print(type(ages))  # BaseArray
```

**Notes:**
- Returns all values for that attribute across all nodes
- Missing values handled gracefully (may be None/NaN)

---

#### Filtering: `g.nodes[condition]`

Filter nodes by condition to create subgraph.

**Returns:**
- `Subgraph`: Filtered subgraph containing matching nodes

**Example:**
```python
# Filter by attribute
young = g.nodes[g.nodes["age"] < 30]
print(f"{young.node_count()} young nodes")

# Multiple conditions
active_young = g.nodes[
    (g.nodes["age"] < 30) &
    (g.nodes["active"] == True)
]

# Filter by degree
high_degree = g.nodes[g.degree() > 5]
```

**Notes:**
- Creates a view (no copying)
- Conditions can use any node attribute
- Can combine with graph metrics (degree, etc.)

---

#### Slicing: `g.nodes[start:end]` or `g.nodes[indices]`

Access nodes by position or ID list.

**Returns:**
- `Subgraph`: Subgraph with selected nodes

**Example:**
```python
# First 10 nodes
first_ten = g.nodes[:10]

# Specific range
middle = g.nodes[10:20]

# By ID list
specific = g.nodes[[0, 5, 10, 15]]

# Every other node
every_other = g.nodes[::2]
```

---

### Core Methods

#### `ids()`

Get array of all node IDs.

**Returns:**
- `NumArray`: Node IDs

**Example:**
```python
g = gr.generators.karate_club()
node_ids = g.nodes.ids()
print(node_ids.to_list())
```

**Performance:** O(n)

---

#### `all()`

Get subgraph containing all nodes.

**Returns:**
- `Subgraph`: Subgraph with all nodes

**Example:**
```python
all_nodes = g.nodes.all()
print(all_nodes.node_count())  # Same as g.node_count()
```

**Notes:** Useful for consistency in pipelines

---

#### `attribute_names()` / `attributes()`

Get list of all node attribute names.

**Returns:**
- `list[str]`: Attribute names

**Example:**
```python
g = gr.Graph()
g.add_node(name="Alice", age=29)
g.add_node(name="Bob", city="NYC")

attrs = g.nodes.attribute_names()
print(sorted(attrs))  # ['age', 'city', 'name']
```

**Notes:** Both methods are aliases

---

#### `attributes()`

Get list of all node attribute names (property/method).

**Returns:**
- `list[str]`: Attribute names

**Example:**
```python
g = gr.generators.karate_club()

# As property
attrs = g.nodes.attributes()
print(sorted(attrs))  # All node attribute names

# Same as attribute_names()
assert g.nodes.attributes() == g.nodes.attribute_names()
```

---

#### `table()`

Convert nodes to tabular format.

**Returns:**
- `NodesTable`: Table of all node data

**Example:**
```python
g = gr.generators.karate_club()
nodes_table = g.nodes.table()

# View first rows
print(nodes_table.head())

# Export to pandas
df = nodes_table.to_pandas()
```

---

#### `array()`

Convert to NodesArray representation.

**Returns:**
- `NodesArray`: Array-based view of nodes

**Example:**
```python
nodes_arr = g.nodes.array()
print(type(nodes_arr))  # NodesArray
```

---

#### `matrix()`

Create matrix from node attributes.

**Returns:**
- `GraphMatrix`: Matrix representation

**Example:**
```python
# Create feature matrix from node attributes
M = g.nodes.matrix()
print(M.shape())
```

**Notes:** Useful for creating feature matrices for ML

---

### Filtering

#### `filter(predicate)`

Filter nodes by a predicate function or expression.

**Parameters:**
- `predicate` (callable or str): Filter expression

**Returns:**
- `Subgraph`: Filtered nodes

**Example:**
```python
# Using callable
filtered = g.nodes.filter(lambda node: node["age"] > 25)

# Using string expression (if supported)
filtered = g.nodes.filter("age > 25")
```

**Notes:** Method signature requires predicate parameter

---

### Grouping and Aggregation

#### `group_by(attribute)`

Group nodes by attribute value.

**Parameters:**
- `attribute` (str): Attribute name to group by

**Returns:**
- `SubgraphArray`: Array of subgraphs, one per group

**Example:**
```python
g = gr.Graph()
g.add_node(city="NYC", age=29)
g.add_node(city="NYC", age=31)
g.add_node(city="LA", age=25)

# Group by city
by_city = g.nodes.group_by("city")
print(f"{len(by_city)} cities")

for city_group in by_city:
    print(f"City group: {city_group.node_count()} nodes")
```

**Use cases:**
- Analyzing groups separately
- Per-group statistics
- Stratified sampling

---

### Attribute Updates

#### `set_attrs(attr_dict)`

Bulk update node attributes.

**Parameters:**
- `attr_dict` (dict): Mapping of node_id → {attr: value}

**Returns:**
- `None` (modifies graph in-place)

**Example:**
```python
g = gr.Graph()
n0, n1, n2 = g.add_nodes([{}, {}, {}])

# Bulk set attributes
g.nodes.set_attrs({
    n0: {"label": "A", "score": 0.9},
    n1: {"label": "B", "score": 0.7},
    n2: {"label": "C", "score": 0.8}
})

# Verify
print(g.nodes["label"].to_list())  # ['A', 'B', 'C']
```

**Performance:** O(n) where n is number of updates

**Notes:**
- Updates existing attributes or adds new ones
- Atomic operation (all or nothing)
- More efficient than individual set operations

---

### Hierarchical Operations

#### `meta()`

Access meta-nodes (for hierarchical graphs).

**Returns:**
- `NodesAccessor`: Accessor for meta-nodes only

**Example:**
```python
if g.has_meta_nodes():
    meta_nodes = g.nodes.meta()
    print(f"{len(meta_nodes.ids())} meta-nodes")
```

---

#### `get_meta_node(node_id)`

Get the meta-node that contains a specific node.

**Parameters:**
- `node_id` (int): Node ID to look up

**Returns:**
- `MetaNode`: Meta-node containing this node

**Example:**
```python
# Get meta-node for node 0
meta = g.nodes.get_meta_node(0)
print(f"Node 0 is in meta-node {meta.id}")
```

**Notes:** Only works in hierarchical graphs with meta-nodes

---

#### `base()`

Access base-level nodes (non-meta).

**Returns:**
- `NodesAccessor`: Accessor for base nodes only

**Example:**
```python
base_nodes = g.nodes.base()
print(f"{len(base_nodes.ids())} base nodes")
```

---

#### `subgraphs()`

Get subgraph membership for nodes.

**Returns:**
- `NumArray`: Subgraph IDs for each node

**Example:**
```python
membership = g.nodes.subgraphs()
print(membership.head())
```

**Notes:** Relevant for hierarchical/multi-level graphs

---

### Visualization

#### `viz()`

Access visualization methods.

**Returns:**
- `VizAccessor`: Visualization accessor

**Example:**
```python
viz = g.nodes.viz()
# Use viz methods for plotting
```

**See:** Visualization guide for available methods

---

## Usage Patterns

### Pattern 1: Filter → Analyze

```python
# Filter to interesting nodes
young = g.nodes[g.nodes["age"] < 30]

# Analyze filtered set
avg_connections = young.degree().mean()
print(f"Young nodes average degree: {avg_connections:.2f}")
```

### Pattern 2: Group → Aggregate

```python
# Group by attribute
by_department = g.nodes.group_by("department")

# Analyze each group
for dept in by_department:
    avg_age = dept.nodes["age"].mean()
    print(f"Department {dept}: avg age {avg_age:.1f}")
```

### Pattern 3: Attribute Access

```python
# Get multiple attributes
ages = g.nodes["age"]
scores = g.nodes["score"]
names = g.nodes["name"]

# Use in computation
import numpy as np
features = np.column_stack([
    ages.to_list(),
    scores.to_list()
])
```

### Pattern 4: Bulk Updates

```python
# Compute values
degrees = g.degree()

# Set as attributes
g.nodes.set_attrs({
    int(nid): {"degree": int(deg)}
    for nid, deg in zip(g.nodes.ids(), degrees)
})

# Now queryable
high_degree = g.nodes[g.nodes["degree"] > 10]
```

### Pattern 5: Chained Operations

```python
# Complex pipeline
result = (
    g.nodes[g.nodes["active"] == True]  # Filter active
     .group_by("department")             # Group by dept
     .sample(5)                          # Sample 5 groups
     .table()                            # Convert to table
     .to_pandas()                        # Export to pandas
)
```

---

## Access Syntax Summary

| Syntax | Returns | Example |
|--------|---------|---------|
| `g.nodes["attr"]` | `BaseArray` | `ages = g.nodes["age"]` |
| `g.nodes[condition]` | `Subgraph` | `young = g.nodes[g.nodes["age"] < 30]` |
| `g.nodes[:10]` | `Subgraph` | `first_ten = g.nodes[:10]` |
| `g.nodes[[0,5,10]]` | `Subgraph` | `specific = g.nodes[[0,5,10]]` |
| `g.nodes.ids()` | `NumArray` | `all_ids = g.nodes.ids()` |
| `g.nodes.table()` | `NodesTable` | `tbl = g.nodes.table()` |

---

## Performance Considerations

**Efficient:**
- Attribute access: `g.nodes["age"]` - O(n) columnar scan
- Filtering: `g.nodes[condition]` - Creates view, lazy evaluation
- Slicing: `g.nodes[:10]` - O(1) view creation

**Less Efficient:**
- Individual node lookups in loops - prefer bulk operations
- Repeated filtering with same condition - cache the subgraph
- Converting to table repeatedly - convert once, reuse

**Best Practices:**
```python
# ✅ Good: bulk operations
ages = g.nodes["age"]
mean_age = ages.mean()

# ❌ Avoid: loops over individual nodes
total = 0
for nid in g.nodes.ids():
    total += g.get_node_attr(nid, "age")
```


---

## Object Transformations

`NodesAccessor` can transform into:

- **NodesAccessor → Subgraph**: `g.nodes[condition]`
- **NodesAccessor → NodesTable**: `g.nodes.table()`
- **NodesAccessor → NodesArray**: `g.nodes.ids()`
- **NodesAccessor → BaseArray**: `g.nodes["attribute"]`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/accessors.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How NodesAccessor works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
