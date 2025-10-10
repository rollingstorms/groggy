# EdgesArray API Reference

**Type**: `groggy.EdgesArray`

---

## Overview

Array of edge IDs with edge-specific operations.

**Primary Use Cases:**
- Working with collections of edge IDs
- Edge set operations
- Batch edge queries

**Related Objects:**
- `NumArray`
- `EdgesAccessor`

---

## Complete Method Reference

The following methods are available on `EdgesArray` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `contains()` | `?` | ✗ |
| `filter()` | `?` | ✗ |
| `filter_by_size()` | `EdgesArray` | ✓ |
| `filter_by_weight()` | `EdgesArray` | ✓ |
| `first()` | `EdgesAccessor` | ✓ |
| `interactive()` | `?` | ✗ |
| `is_empty()` | `bool` | ✓ |
| `iter()` | `EdgesArrayIterator` | ✓ |
| `last()` | `EdgesAccessor` | ✓ |
| `nodes()` | `NodesArray` | ✓ |
| `stats()` | `dict` | ✓ |
| `table()` | `TableArray` | ✓ |
| `to_list()` | `list` | ✓ |
| `total_edge_count()` | `int` | ✓ |
| `union()` | `EdgesAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating EdgesArray

EdgesArray is typically returned from edge grouping operations:

```python
import groggy as gr

g = gr.generators.karate_club()

# From edge grouping
by_type = g.edges.group_by("type")  # → EdgesArray
print(type(by_type))  # EdgesArray

# Each element is an EdgesAccessor (edge collection)
for edges in by_type:
    print(f"{edges.edge_count()} edges in group")
```

**Key Concept:** EdgesArray is an array of edge collections, where each element is an EdgesAccessor representing a group of edges.

---

### Core Methods

#### `first()`

Get first edge collection.

**Returns:**
- `EdgesAccessor`: First group of edges

**Example:**
```python
by_type = g.edges.group_by("type")
first_group = by_type.first()
print(f"First group: {first_group.edge_count()} edges")
```

**Performance:** O(1)

---

#### `last()`

Get last edge collection.

**Returns:**
- `EdgesAccessor`: Last group of edges

**Example:**
```python
by_type = g.edges.group_by("type")
last_group = by_type.last()
print(f"Last group: {last_group.edge_count()} edges")
```

**Performance:** O(1)

---

#### `is_empty()`

Check if array has no groups.

**Returns:**
- `bool`: True if no groups

**Example:**
```python
by_type = g.edges.group_by("type")
if by_type.is_empty():
    print("No groups found")
else:
    print(f"{len(by_type)} edge groups")
```

**Performance:** O(1)

---

#### `iter()`

Iterate over edge collections.

**Returns:**
- Iterator over EdgesAccessor objects

**Example:**
```python
by_type = g.edges.group_by("type")

for edges_group in by_type.iter():
    print(f"Group: {edges_group.edge_count()} edges")
    # Access attributes
    if edges_group.edge_count() > 0:
        weights = edges_group["weight"]
        print(f"  Avg weight: {weights.mean():.2f}")
```

**Performance:** O(1) per iteration

---

### Aggregation Methods

#### `total_edge_count()`

Get total edges across all groups.

**Returns:**
- `int`: Total edge count

**Example:**
```python
by_type = g.edges.group_by("type")
total = by_type.total_edge_count()
print(f"Total edges: {total}")

# Should equal original graph
assert total == g.edge_count()
```

**Performance:** O(k) where k is number of groups

---

#### `union()`

Combine all edge groups into single collection.

**Returns:**
- `EdgesAccessor`: Union of all groups

**Example:**
```python
by_type = g.edges.group_by("type")
all_edges = by_type.union()

# Should contain all edges
print(f"Union: {all_edges.edge_count()} edges")
assert all_edges.edge_count() == g.edge_count()
```

**Performance:** O(k) where k is number of groups

---

#### `stats()`

Get statistics about the array.

**Returns:**
- `dict`: Statistics including group count, sizes, etc.

**Example:**
```python
by_type = g.edges.group_by("type")
stats = by_type.stats()
print(stats)
# {'num_groups': 3, 'total_edges': 78, 'avg_size': 26.0, ...}
```

---

### Filtering Methods

#### `filter_by_size(min_size, max_size=None)`

Filter groups by edge count.

**Parameters:**
- `min_size` (int): Minimum edges in group
- `max_size` (int, optional): Maximum edges in group

**Returns:**
- `EdgesArray`: Filtered array

**Example:**
```python
by_type = g.edges.group_by("type")

# Only large groups
large = by_type.filter_by_size(min_size=10)
print(f"{len(large)} groups with ≥10 edges")

# Size range
medium = by_type.filter_by_size(min_size=5, max_size=20)
```

**Performance:** O(k) where k is number of groups

---

#### `filter_by_weight(threshold, operator='>')`

Filter groups by average weight.

**Parameters:**
- `threshold` (float): Weight threshold
- `operator` (str): Comparison operator ('>', '<', '>=', '<=', '==')

**Returns:**
- `EdgesArray`: Filtered array

**Example:**
```python
by_type = g.edges.group_by("type")

# Groups with high average weight
heavy = by_type.filter_by_weight(threshold=5.0, operator='>')
print(f"{len(heavy)} groups with avg weight > 5.0")

# Groups with low average weight
light = by_type.filter_by_weight(threshold=2.0, operator='<')
```

---

### Conversion Methods

#### `to_list()`

Convert to Python list of EdgesAccessor objects.

**Returns:**
- `list[EdgesAccessor]`: List of edge groups

**Example:**
```python
by_type = g.edges.group_by("type")
groups = by_type.to_list()

for i, group in enumerate(groups):
    print(f"Group {i}: {group.edge_count()} edges")
```

**Performance:** O(k) where k is number of groups

---

#### `table()`

Convert all groups to table array.

**Returns:**
- `TableArray`: Array of tables, one per group

**Example:**
```python
by_type = g.edges.group_by("type")
tables = by_type.table()

# Export each group
for i, tbl in enumerate(tables):
    tbl.to_csv(f"edge_group_{i}.csv")
```

---

#### `nodes()`

Get node groups corresponding to edge groups.

**Returns:**
- `NodesArray`: Array of node collections (endpoints)

**Example:**
```python
by_type = g.edges.group_by("type")
node_groups = by_type.nodes()

# Analyze nodes involved in each edge type
for i, nodes in enumerate(node_groups):
    print(f"Type {i}: {nodes.node_count()} nodes involved")
```

**Notes:** Returns nodes that are endpoints of edges in each group

---

### Additional Methods

#### `contains(item)`

Check if array contains specific item.

**Parameters:**
- `item`: Item to check for

**Returns:**
- `bool`: True if item exists

**Example:**
```python
by_type = g.edges.group_by("type")
if by_type.contains(some_group):
    print("Group found")
```

**Status:** Requires item parameter

---

#### `filter(predicate)`

Filter groups by predicate function.

**Parameters:**
- `predicate`: Filter function

**Returns:**
- `EdgesArray`: Filtered array

**Example:**
```python
by_type = g.edges.group_by("type")
large = by_type.filter(lambda g: g.edge_count() > 10)
```

**Status:** Requires predicate parameter

---

#### `interactive()`

Launch interactive visualization (not yet implemented).

**Returns:**
- `str`: HTML/visualization output

**Example:**
```python
by_type = g.edges.group_by("type")
# Not yet implemented - use workaround:
by_type.table().interactive_viz()
```

**Status:** Not yet implemented - use `.table().interactive_viz()` instead

---

### Indexing & Slicing

EdgesArray supports indexing and slicing:

**Example:**
```python
by_type = g.edges.group_by("type")

# Get specific group
group_0 = by_type[0]  # First group (EdgesAccessor)
print(f"Group 0: {group_0.edge_count()} edges")

# Slice
first_three = by_type[:3]  # First 3 groups (EdgesArray)

# Negative indexing
last_group = by_type[-1]
```

---

## Usage Patterns

### Pattern 1: Edge Type Analysis

```python
by_type = g.edges.group_by("type")

print(f"Total edge types: {len(by_type)}")
print(f"Total edges: {by_type.total_edge_count()}")

for i, edges_group in enumerate(by_type):
    weights = edges_group["weight"]
    print(f"\nType {i}:")
    print(f"  Count: {edges_group.edge_count()}")
    print(f"  Avg weight: {weights.mean():.2f}")
    print(f"  Weight range: {weights.min():.1f}-{weights.max():.1f}")
```

### Pattern 2: Filter and Analyze

```python
by_type = g.edges.group_by("type")

# Only significant edge types (≥10 edges)
significant = by_type.filter_by_size(min_size=10)

for edges in significant:
    edge_type = edges["type"].first()
    print(f"\n{edge_type}:")
    print(f"  Edges: {edges.edge_count()}")

    # Analyze endpoint nodes
    nodes = edges.nodes()
    print(f"  Nodes involved: {nodes.node_count()}")
    print(f"  Avg degree: {nodes.degree().mean():.2f}")
```

### Pattern 3: Weight-Based Filtering

```python
by_type = g.edges.group_by("type")

# Heavy edge types
heavy_types = by_type.filter_by_weight(threshold=5.0, operator='>')

print(f"{len(heavy_types)} heavy edge types:")
for edges in heavy_types:
    edge_type = edges["type"].first()
    weights = edges["weight"]
    print(f"  {edge_type}: avg={weights.mean():.2f}, max={weights.max():.2f}")
```

### Pattern 4: Export by Type

```python
by_type = g.edges.group_by("type")

# Export each edge type separately
for i, edges_group in enumerate(by_type):
    edge_type = edges_group["type"].first() if edges_group.edge_count() > 0 else f"type_{i}"

    # Create edge list
    df = edges_group.table().to_pandas()
    df.to_csv(f"edges_{edge_type}.csv", index=False)

    print(f"Exported {edge_type}: {len(df)} edges")
```

### Pattern 5: Cross-Type Comparison

```python
by_type = g.edges.group_by("type")

groups = by_type.to_list()
if len(groups) >= 2:
    type_a = groups[0]
    type_b = groups[1]

    print("Type comparison:")
    print(f"  Type A: {type_a.edge_count()} edges")
    print(f"    Avg weight: {type_a['weight'].mean():.2f}")
    print(f"    Nodes: {type_a.nodes().node_count()}")

    print(f"  Type B: {type_b.edge_count()} edges")
    print(f"    Avg weight: {type_b['weight'].mean():.2f}")
    print(f"    Nodes: {type_b.nodes().node_count()}")
```

### Pattern 6: Hierarchical Edge Analysis

```python
# Group by primary type
by_primary = g.edges.group_by("primary_type")

for primary_edges in by_primary:
    primary = primary_edges["primary_type"].first()
    print(f"\n{primary}:")

    # Sub-group by secondary type
    by_secondary = primary_edges.group_by("secondary_type")
    for secondary_edges in by_secondary:
        secondary = secondary_edges["secondary_type"].first()
        print(f"  {secondary}: {secondary_edges.edge_count()} edges")
```

---

## Quick Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `first()` | `EdgesAccessor` | First group |
| `last()` | `EdgesAccessor` | Last group |
| `is_empty()` | `bool` | Check if empty |
| `iter()` | Iterator | Iterate groups |
| `total_edge_count()` | `int` | Total edges |
| `union()` | `EdgesAccessor` | Combine all groups |
| `stats()` | `dict` | Array statistics |
| `filter_by_size(min, max)` | `EdgesArray` | Filter by edge count |
| `filter_by_weight(thresh, op)` | `EdgesArray` | Filter by avg weight |
| `to_list()` | `list` | Convert to list |
| `table()` | `TableArray` | Convert to tables |
| `nodes()` | `NodesArray` | Get endpoint nodes |
| `[i]` | `EdgesAccessor` | Get group by index |
| `[:n]` | `EdgesArray` | Slice groups |


---

## Object Transformations

`EdgesArray` can transform into:

- **EdgesArray → Subgraph**: `g.edges[edge_array]`
- **EdgesArray → ndarray**: `edge_array.to_numpy()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/arrays.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How EdgesArray works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
