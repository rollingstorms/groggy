# NodesArray API Reference

**Type**: `groggy.NodesArray`

---

## Overview

Array of node IDs with node-specific operations.

**Primary Use Cases:**
- Working with collections of node IDs
- Node set operations
- Batch node queries

**Related Objects:**
- `NumArray`
- `NodesAccessor`

---

## Complete Method Reference

The following methods are available on `NodesArray` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `contains()` | `?` | ✗ |
| `filter()` | `?` | ✗ |
| `filter_by_size()` | `?` | ✗ |
| `first()` | `NodesAccessor` | ✓ |
| `interactive()` | `?` | ✗ |
| `is_empty()` | `bool` | ✓ |
| `iter()` | `NodesArrayIterator` | ✓ |
| `last()` | `NodesAccessor` | ✓ |
| `stats()` | `dict` | ✓ |
| `table()` | `TableArray` | ✓ |
| `to_list()` | `list` | ✓ |
| `total_node_count()` | `int` | ✓ |
| `union()` | `NodesAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating NodesArray

NodesArray is typically returned from grouping operations:

```python
import groggy as gr

g = gr.generators.karate_club()

# From node grouping
by_club = g.nodes.group_by("club")  # → NodesArray
print(type(by_club))  # NodesArray

# Each element is a NodesAccessor (node collection)
for nodes in by_club:
    print(f"{nodes.node_count()} nodes in group")
```

**Key Concept:** NodesArray is an array of node collections, where each element is a NodesAccessor representing a group of nodes.

---

### Core Methods

#### `first()`

Get first node collection.

**Returns:**
- `NodesAccessor`: First group of nodes

**Example:**
```python
by_club = g.nodes.group_by("club")
first_group = by_club.first()
print(f"First group: {first_group.node_count()} nodes")
```

**Performance:** O(1)

---

#### `last()`

Get last node collection.

**Returns:**
- `NodesAccessor`: Last group of nodes

**Example:**
```python
by_club = g.nodes.group_by("club")
last_group = by_club.last()
print(f"Last group: {last_group.node_count()} nodes")
```

**Performance:** O(1)

---

#### `is_empty()`

Check if array has no groups.

**Returns:**
- `bool`: True if no groups

**Example:**
```python
by_club = g.nodes.group_by("club")
if by_club.is_empty():
    print("No groups found")
else:
    print(f"{len(by_club)} groups")
```

**Performance:** O(1)

---

#### `iter()`

Iterate over node collections.

**Returns:**
- Iterator over NodesAccessor objects

**Example:**
```python
by_club = g.nodes.group_by("club")

for nodes_group in by_club.iter():
    print(f"Group: {nodes_group.node_count()} nodes")
    # Access attributes
    if nodes_group.node_count() > 0:
        attrs = nodes_group.attribute_names()
        print(f"  Attributes: {attrs}")
```

**Performance:** O(1) per iteration

---

### Aggregation Methods

#### `total_node_count()`

Get total nodes across all groups.

**Returns:**
- `int`: Total node count

**Example:**
```python
by_club = g.nodes.group_by("club")
total = by_club.total_node_count()
print(f"Total nodes: {total}")

# Should equal original graph
assert total == g.node_count()
```

**Performance:** O(k) where k is number of groups

---

#### `union()`

Combine all node groups into single collection.

**Returns:**
- `NodesAccessor`: Union of all groups

**Example:**
```python
by_club = g.nodes.group_by("club")
all_nodes = by_club.union()

# Should contain all nodes
print(f"Union: {all_nodes.node_count()} nodes")
assert all_nodes.node_count() == g.node_count()
```

**Performance:** O(k) where k is number of groups

---

#### `stats()`

Get statistics about the array.

**Returns:**
- `dict`: Statistics including group count, sizes, etc.

**Example:**
```python
by_club = g.nodes.group_by("club")
stats = by_club.stats()
print(stats)
# {'num_groups': 2, 'total_nodes': 34, 'avg_size': 17.0, ...}
```

---

### Filtering & Membership

#### `contains(item)`

Check if array contains a specific item.

**Parameters:**
- `item`: Item to check for

**Returns:**
- `bool`: True if item exists

**Example:**
```python
by_club = g.nodes.group_by("club")
# Check if specific group exists
if by_club.contains(some_group):
    print("Group found")
```

**Notes:** Requires item parameter

---

#### `filter(predicate)`

Filter groups by predicate.

**Parameters:**
- `predicate` (callable): Filter function

**Returns:**
- `NodesArray`: Filtered array

**Example:**
```python
by_club = g.nodes.group_by("club")
# Only large groups
large = by_club.filter(lambda g: g.node_count() > 10)
```

**Notes:** Requires predicate parameter

---

#### `filter_by_size(min_size, max_size=None)`

Filter groups by node count.

**Parameters:**
- `min_size` (int): Minimum nodes in group
- `max_size` (int, optional): Maximum nodes in group

**Returns:**
- `NodesArray`: Filtered array

**Example:**
```python
by_club = g.nodes.group_by("club")
# Only groups with 5-20 nodes
medium = by_club.filter_by_size(min_size=5, max_size=20)
```

**Notes:** Requires min_size parameter

---

### Conversion Methods

#### `to_list()`

Convert to Python list of NodesAccessor objects.

**Returns:**
- `list[NodesAccessor]`: List of node groups

**Example:**
```python
by_club = g.nodes.group_by("club")
groups = by_club.to_list()

for i, group in enumerate(groups):
    print(f"Group {i}: {group.node_count()} nodes")
```

**Performance:** O(k) where k is number of groups

---

#### `table()`

Convert all groups to table array.

**Returns:**
- `TableArray`: Array of tables, one per group

**Example:**
```python
by_club = g.nodes.group_by("club")
tables = by_club.table()

# Export each group
for i, tbl in enumerate(tables):
    tbl.to_csv(f"group_{i}.csv")
```

---

#### `interactive()`

Launch interactive visualization (not yet implemented).

**Returns:**
- `str`: HTML/visualization output

**Example:**
```python
by_club = g.nodes.group_by("club")
# Not yet implemented - use workaround:
by_club.table().interactive_viz()
```

**Notes:** Not yet implemented - use `.table().interactive_viz()` instead

---

### Indexing & Slicing

NodesArray supports indexing and slicing:

**Example:**
```python
by_club = g.nodes.group_by("club")

# Get specific group
group_0 = by_club[0]  # First group (NodesAccessor)
print(f"Group 0: {group_0.node_count()} nodes")

# Slice
first_three = by_club[:3]  # First 3 groups (NodesArray)

# Negative indexing
last_group = by_club[-1]
```

---

## Usage Patterns

### Pattern 1: Group Statistics

```python
by_club = g.nodes.group_by("club")

print(f"Total groups: {len(by_club)}")
print(f"Total nodes: {by_club.total_node_count()}")

for i, group in enumerate(by_club):
    print(f"\nGroup {i}:")
    print(f"  Size: {group.node_count()}")
    print(f"  Avg degree: {group.degree().mean():.2f}")
```

### Pattern 2: Per-Group Analysis

```python
by_department = g.nodes.group_by("department")

for dept_nodes in by_department:
    # Get department name from first node
    if dept_nodes.node_count() > 0:
        dept_name = dept_nodes["department"].first()

        # Analyze this department
        ages = dept_nodes["age"]
        print(f"\n{dept_name}:")
        print(f"  Size: {dept_nodes.node_count()}")
        print(f"  Avg age: {ages.mean():.1f}")
        print(f"  Age range: {ages.min():.0f}-{ages.max():.0f}")
```

### Pattern 3: Export Groups Separately

```python
by_city = g.nodes.group_by("city")

# Export each city to separate file
for i, city_nodes in enumerate(by_city):
    city_table = city_nodes.table()
    city_name = city_nodes["city"].first() if city_nodes.node_count() > 0 else f"unknown_{i}"
    city_table.to_csv(f"city_{city_name}.csv")
```

### Pattern 4: Filter Groups by Size

```python
by_type = g.nodes.group_by("type")

# Get only large groups
large_groups = []
for group in by_type:
    if group.node_count() >= 10:
        large_groups.append(group)

print(f"{len(large_groups)} groups with ≥10 nodes")

# Analyze large groups
for group in large_groups:
    print(f"Large group: {group.node_count()} nodes")
```

### Pattern 5: Compare Groups

```python
by_club = g.nodes.group_by("club")

groups_list = by_club.to_list()
if len(groups_list) >= 2:
    group_a = groups_list[0]
    group_b = groups_list[1]

    print("Group comparison:")
    print(f"  Group A: {group_a.node_count()} nodes, avg degree {group_a.degree().mean():.2f}")
    print(f"  Group B: {group_b.node_count()} nodes, avg degree {group_b.degree().mean():.2f}")

    # Inter-group edges
    edges_between = g.edges[
        (g.edges.sources().to_list() in group_a.node_ids().to_list()) &
        (g.edges.targets().to_list() in group_b.node_ids().to_list())
    ]
    print(f"  Edges between: {edges_between.edge_count()}")
```

### Pattern 6: Hierarchical Grouping

```python
# First group by country
by_country = g.nodes.group_by("country")

# Then group each country by city
for country_nodes in by_country:
    country = country_nodes["country"].first()
    print(f"\n{country}:")

    by_city = country_nodes.group_by("city")
    for city_nodes in by_city:
        city = city_nodes["city"].first()
        print(f"  {city}: {city_nodes.node_count()} nodes")
```

---

## Quick Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `first()` | `NodesAccessor` | First group |
| `last()` | `NodesAccessor` | Last group |
| `is_empty()` | `bool` | Check if empty |
| `iter()` | Iterator | Iterate groups |
| `total_node_count()` | `int` | Total nodes |
| `union()` | `NodesAccessor` | Combine all groups |
| `stats()` | `dict` | Array statistics |
| `to_list()` | `list` | Convert to list |
| `table()` | `TableArray` | Convert to tables |
| `[i]` | `NodesAccessor` | Get group by index |
| `[:n]` | `NodesArray` | Slice groups |


---

## Object Transformations

`NodesArray` can transform into:

- **NodesArray → Subgraph**: `g.nodes[node_array]`
- **NodesArray → ndarray**: `node_array.to_numpy()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/arrays.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How NodesArray works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains
