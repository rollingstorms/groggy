# NumArray API Reference

**Type**: `groggy.NumArray`

---

## Overview

Numeric array with mathematical operations and statistics.

**Primary Use Cases:**
- Numerical computations on graph attributes
- Statistical analysis
- Vector operations

**Related Objects:**
- `BaseArray`
- `NodesArray`
- `EdgesArray`

---

## Complete Method Reference

The following methods are available on `NumArray` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `contains()` | `?` | ✗ |
| `count()` | `int` | ✓ |
| `dtype()` | `str` | ✓ |
| `first()` | `int` | ✓ |
| `is_empty()` | `bool` | ✓ |
| `last()` | `int` | ✓ |
| `max()` | `float` | ✓ |
| `mean()` | `float` | ✓ |
| `min()` | `float` | ✓ |
| `nunique()` | `int` | ✓ |
| `reshape()` | `?` | ✗ |
| `std()` | `float` | ✓ |
| `sum()` | `float` | ✓ |
| `to_list()` | `list` | ✓ |
| `to_type()` | `?` | ✗ |
| `unique()` | `NumArray` | ✓ |
| `var()` | `float` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating NumArray

NumArrays are typically returned from graph operations, not constructed directly:

```python
import groggy as gr

g = gr.generators.karate_club()

# From node/edge operations
node_ids = g.nodes.ids()        # NumArray
degrees = g.degree()             # NumArray
edge_ids = g.edges.ids()         # NumArray

# Manual creation
arr = gr.num_array([1, 2, 3, 4, 5])  # NumArray
```

---

### Statistical Methods

#### `mean()`

Calculate arithmetic mean.

**Returns:**
- `float`: Mean value

**Example:**
```python
degrees = g.degree()
avg_degree = degrees.mean()
print(f"Average degree: {avg_degree:.2f}")
```

**Performance:** O(n)

---

#### `sum()`

Calculate sum of all elements.

**Returns:**
- `float`: Sum

**Example:**
```python
degrees = g.degree()
total_degree = degrees.sum()
print(f"Total degree: {total_degree}")

# For undirected graphs: total_degree = 2 * edge_count
```

**Performance:** O(n)

---

#### `min()`

Find minimum value.

**Returns:**
- `float`: Minimum value

**Example:**
```python
degrees = g.degree()
min_degree = degrees.min()
print(f"Minimum degree: {min_degree}")
```

**Performance:** O(n)

---

#### `max()`

Find maximum value.

**Returns:**
- `float`: Maximum value

**Example:**
```python
degrees = g.degree()
max_degree = degrees.max()
print(f"Maximum degree: {max_degree}")
print(f"Hub node(s) have degree {max_degree}")
```

**Performance:** O(n)

---

#### `std()`

Calculate standard deviation.

**Returns:**
- `float`: Standard deviation

**Example:**
```python
degrees = g.degree()
degree_std = degrees.std()
print(f"Degree std dev: {degree_std:.2f}")
```

**Performance:** O(n)

**Notes:** Uses sample standard deviation (n-1 denominator)

---

#### `var()`

Calculate variance.

**Returns:**
- `float`: Variance

**Example:**
```python
degrees = g.degree()
degree_var = degrees.var()
print(f"Degree variance: {degree_var:.2f}")
```

**Performance:** O(n)

**Notes:** Uses sample variance (n-1 denominator)

---

### Data Access

#### `to_list()`

Convert to Python list.

**Returns:**
- `list`: Python list of values

**Example:**
```python
degrees = g.degree()
degree_list = degrees.to_list()
print(degree_list[:5])  # [4, 3, 5, 2, 1]
```

**Performance:** O(n)

---

#### `first()`

Get first element.

**Returns:**
- Scalar value (type varies)

**Example:**
```python
node_ids = g.nodes.ids()
first_id = node_ids.first()
print(f"First node ID: {first_id}")
```

**Performance:** O(1)

---

#### `last()`

Get last element.

**Returns:**
- Scalar value (type varies)

**Example:**
```python
node_ids = g.nodes.ids()
last_id = node_ids.last()
print(f"Last node ID: {last_id}")
```

**Performance:** O(1)

---

#### `count()` / `len()`

Get number of elements.

**Returns:**
- `int`: Number of elements

**Example:**
```python
degrees = g.degree()
n = degrees.count()
print(f"{n} nodes")

# Also works with len()
n = len(degrees)
```

**Performance:** O(1)

---

### Properties

#### `dtype()`

Get data type of array.

**Returns:**
- `str`: Type name ("int", "float", etc.)

**Example:**
```python
node_ids = g.nodes.ids()
print(node_ids.dtype())  # "int"

ages = g.nodes["age"]
print(ages.dtype())  # May be "float" or "int"
```

---

#### `is_empty()`

Check if array has no elements.

**Returns:**
- `bool`: True if empty

**Example:**
```python
filtered = g.nodes[g.nodes["age"] > 1000]
ids = filtered.node_ids()
if ids.is_empty():
    print("No matching nodes")
```

**Performance:** O(1)

---

### Unique Values

#### `unique()`

Get unique values.

**Returns:**
- `NumArray`: Array of unique values

**Example:**
```python
# Node attributes
cities = g.nodes["city"]
unique_cities = cities.unique()
print(f"{len(unique_cities)} unique cities")
print(unique_cities.to_list())
```

**Performance:** O(n log n) - sorts internally

---

#### `nunique()`

Count unique values.

**Returns:**
- `int`: Number of unique values

**Example:**
```python
cities = g.nodes["city"]
n_cities = cities.nunique()
print(f"{n_cities} different cities")
```

**Performance:** O(n log n)

---

### Array Operations

#### Comparison Operations

NumArray supports comparison operators:

**Example:**
```python
degrees = g.degree()

# Boolean mask
high_degree_mask = degrees > 5
low_degree_mask = degrees < 2
range_mask = (degrees >= 3) & (degrees <= 7)

# Use in filtering
high_degree_nodes = g.nodes[degrees > 5]
```

**Operators:**
- `>`, `>=`, `<`, `<=`, `==`, `!=`
- `&` (and), `|` (or) for combining conditions

---

#### Arithmetic Operations

**Example:**
```python
degrees = g.degree()

# Scalar operations
normalized = degrees / degrees.max()

# Array operations (if supported)
# combined = arr1 + arr2
```

---

### Conversion

#### `to_numpy()`

Convert to NumPy array (if implemented).

**Returns:**
- `numpy.ndarray`: NumPy array

**Example:**
```python
import numpy as np

degrees = g.degree()
np_degrees = np.array(degrees.to_list())

# Or if to_numpy() exists:
# np_degrees = degrees.to_numpy()
```

---

### Display Methods

#### `head(n=5)`

Show first n elements.

**Parameters:**
- `n` (int): Number of elements to show (default 5)

**Returns:**
- Display output (varies by environment)

**Example:**
```python
degrees = g.degree()
degrees.head()      # First 5
degrees.head(10)    # First 10
```

---

#### `tail(n=5)`

Show last n elements.

**Parameters:**
- `n` (int): Number of elements to show (default 5)

**Returns:**
- Display output

**Example:**
```python
degrees = g.degree()
degrees.tail()      # Last 5
degrees.tail(10)    # Last 10
```

---

## Usage Patterns

### Pattern 1: Basic Statistics

```python
degrees = g.degree()

stats = {
    'mean': degrees.mean(),
    'std': degrees.std(),
    'min': degrees.min(),
    'max': degrees.max(),
    'count': len(degrees)
}

print(f"Degree stats: {stats}")
```

### Pattern 2: Filtering

```python
degrees = g.degree()

# Create boolean mask
high_degree = degrees > degrees.mean()

# Use in node filtering
hubs = g.nodes[high_degree]
print(f"{hubs.node_count()} hub nodes")
```

### Pattern 3: Normalization

```python
values = g.nodes["score"]

# Z-score normalization
mean_val = values.mean()
std_val = values.std()

# Convert to list for computation
vals_list = values.to_list()
normalized = [(v - mean_val) / std_val for v in vals_list]
```

### Pattern 4: Binning/Categorization

```python
degrees = g.degree()

# Categorize by degree
degree_list = degrees.to_list()
categories = []
for d in degree_list:
    if d < 5:
        categories.append("low")
    elif d < 15:
        categories.append("medium")
    else:
        categories.append("high")

# Set as attribute
g.nodes.set_attrs({
    int(nid): {"degree_category": cat}
    for nid, cat in zip(g.nodes.ids(), categories)
})
```

### Pattern 5: Value Counts

```python
cities = g.nodes["city"]

# Count occurrences
from collections import Counter
city_counts = Counter(cities.to_list())

print("Nodes per city:")
for city, count in city_counts.most_common():
    print(f"  {city}: {count}")
```

### Pattern 6: Correlation Analysis

```python
import numpy as np

ages = g.nodes["age"].to_list()
scores = g.nodes["score"].to_list()
degrees = g.degree().to_list()

# Correlation matrix
data = np.column_stack([ages, scores, degrees])
corr = np.corrcoef(data.T)

print("Correlation matrix:")
print(corr)
```

---

## Quick Reference

### Statistics

| Method | Returns | Description |
|--------|---------|-------------|
| `mean()` | `float` | Arithmetic mean |
| `sum()` | `float` | Sum of values |
| `min()` | `float` | Minimum value |
| `max()` | `float` | Maximum value |
| `std()` | `float` | Standard deviation |
| `var()` | `float` | Variance |

### Data Access

| Method | Returns | Description |
|--------|---------|-------------|
| `to_list()` | `list` | Convert to Python list |
| `first()` | Scalar | First element |
| `last()` | Scalar | Last element |
| `count()` | `int` | Number of elements |

### Unique Values

| Method | Returns | Description |
|--------|---------|-------------|
| `unique()` | `NumArray` | Unique values |
| `nunique()` | `int` | Count of unique values |

---

## Performance Considerations

**Efficient Operations:**
- `mean()`, `sum()`, `min()`, `max()` - O(n) single pass
- `count()`, `first()`, `last()` - O(1)
- `to_list()` - O(n) direct copy

**Moderate Cost:**
- `unique()`, `nunique()` - O(n log n) due to sorting
- `std()`, `var()` - O(n) but requires two passes

**Best Practices:**
```python
# ✅ Good: compute once, use multiple times
degrees = g.degree()
mean_deg = degrees.mean()
std_deg = degrees.std()

# ❌ Avoid: recomputing same array
for i in range(10):
    mean_deg = g.degree().mean()  # Recomputes each time

# ✅ Good: filter in one pass
high = g.nodes[g.degree() > 5]

# ❌ Avoid: multiple conversions
degrees_list = g.degree().to_list()
mean_deg = sum(degrees_list) / len(degrees_list)  # Use .mean() instead
```

---

## Comparison with NumPy

NumArray provides a subset of NumPy functionality optimized for graph data:

| Feature | NumArray | NumPy |
|---------|----------|-------|
| Basic stats | ✅ `mean()`, `std()`, etc. | ✅ Full suite |
| Element access | ✅ `first()`, `last()` | ✅ Indexing `arr[0]` |
| Unique values | ✅ `unique()`, `nunique()` | ✅ `np.unique()` |
| Broadcasting | ❌ Limited | ✅ Full support |
| Linear algebra | ❌ Use GraphMatrix | ✅ Full support |
| Conversion | ✅ `to_list()` | ✅ `.tolist()` |

**When to convert to NumPy:**
- Need advanced operations (FFT, linear algebra, etc.)
- Integration with NumPy-based libraries
- Broadcasting arithmetic

**When to stay with NumArray:**
- Simple statistics
- Filtering graph elements
- Integration with other Groggy objects


---

## Object Transformations

`NumArray` can transform into:

- **NumArray → ndarray**: `num_array.to_numpy()`
- **NumArray → scalar**: `num_array.mean()`, `num_array.sum()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/arrays.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How NumArray works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains

## Additional Methods

#### `contains(item)`

Contains.

**Parameters:**
- `item`: item

**Returns:**
- `None`: Return value

**Example:**
```python
obj.contains(item=...)
```

---

#### `reshape(rows, cols)`

Reshape.

**Parameters:**
- `rows`: rows
- `cols`: cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.reshape(rows=..., cols=...)
```

---

#### `to_type(dtype)`

To Type.

**Parameters:**
- `dtype`: dtype

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_type(dtype=...)
```

---

