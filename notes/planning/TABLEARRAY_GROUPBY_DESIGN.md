# TableArray & SubgraphArray GroupBy/Iteration API Design

## Problem Statement

Users need intuitive, chainable APIs for:
1. Grouping tables by columns
2. Iterating over groups
3. Applying aggregations to groups
4. Working with arrays of tables/subgraphs

Example use case:
```python
gt = gr.from_csv(
    nodes_filepath='nodes.csv',
    edges_filepath='edges.csv',
    node_id_column='object_name',
    source_id_column='object_name',
    target_id_column='result_type'
)

# Goal: Calculate mean success rate per object
result = gt.edges.group_by('object_name')['success'].mean()
```

## Current State

- `BaseTable`, `NodesTable`, `EdgesTable`, `GraphTable` exist in Rust FFI
- `TableArray` and `SubgraphArray` exist but lack group_by
- No clear iteration/aggregation pattern

## Proposed Design

### Core Principle: **Implicit Delegation**

When you call `.group_by()` on a Table, it returns a `TableArray` where each group is a separate table. Operations on `TableArray` automatically apply to each table and collect results.

### API Examples

#### 1. Basic Group-By with Aggregation

```python
# Group edges by 'object_name', then calculate mean of 'success' column
result = gt.edges.group_by('object_name')['success'].mean()

# Returns: BaseTable with columns ['object_name', 'success']
# Each row is one group's mean
```

**Chain breakdown:**
1. `gt.edges` → `EdgesTable`
2. `.group_by('object_name')` → `TableArray` (one table per unique object_name)
3. `['success']` → `ArrayArray` (extract 'success' column from each table)
4. `.mean()` → `BaseArray` (mean of each array)
5. Auto-package → `BaseTable` with group keys + results

#### 2. Multiple Aggregations

```python
# Multiple columns, multiple aggregations
result = gt.edges.group_by('object_name').agg({
    'success': 'mean',
    'duration': ['min', 'max', 'mean'],
    'count': 'sum'
})

# Returns: BaseTable with columns
# ['object_name', 'success_mean', 'duration_min', 'duration_max', 'duration_mean', 'count_sum']
```

#### 3. Explicit Iteration

```python
# Iterate over groups
for group_key, group_table in gt.edges.group_by('object_name'):
    print(f"Group {group_key}: {len(group_table)} rows")
    print(group_table.head())
```

#### 4. Multi-Column Grouping

```python
# Group by multiple columns
result = gt.edges.group_by(['object_name', 'result_type'])['success'].mean()

# Returns: BaseTable with ['object_name', 'result_type', 'success']
```

#### 5. Graph/Subgraph Operations

```python
# Group subgraphs, operate on each
components = g.connected_components()  # SubgraphArray

# Get node count per component
node_counts = components.map(lambda sg: len(sg.nodes))  # BaseArray

# Get average degree per component
avg_degrees = components['degree'].mean()  # BaseArray

# Filter components by size
large_components = components.filter(lambda sg: len(sg.nodes) > 10)
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Priority)

#### A. `TableArray` class

```python
class TableArray:
    """Array of tables, typically from group_by operation."""

    def __init__(self, tables: List[BaseTable], keys: Optional[List[Any]] = None):
        self.tables = tables
        self.keys = keys or list(range(len(tables)))

    def __getitem__(self, column: str) -> 'ArrayArray':
        """Extract column from each table, returns ArrayArray."""
        return ArrayArray([table[column] for table in self.tables], self.keys)

    def __iter__(self):
        """Iterate over (key, table) pairs."""
        return iter(zip(self.keys, self.tables))

    def __len__(self):
        return len(self.tables)

    def agg(self, spec: Dict[str, Union[str, List[str]]]) -> BaseTable:
        """Apply aggregations to each table, collect as table."""
        # Implementation details below
        pass

    def map(self, func) -> BaseArray:
        """Apply function to each table, collect results as array."""
        return BaseArray([func(table) for table in self.tables])

    def filter(self, func) -> 'TableArray':
        """Filter tables by predicate."""
        filtered = [(k, t) for k, t in zip(self.keys, self.tables) if func(t)]
        return TableArray([t for _, t in filtered], [k for k, _ in filtered])
```

#### B. `ArrayArray` class

```python
class ArrayArray:
    """Array of arrays, intermediate for column extraction from TableArray."""

    def __init__(self, arrays: List[BaseArray], keys: Optional[List[Any]] = None):
        self.arrays = arrays
        self.keys = keys or list(range(len(arrays)))

    def mean(self) -> BaseArray:
        """Calculate mean of each array."""
        return BaseArray([arr.mean() for arr in self.arrays])

    def sum(self) -> BaseArray:
        return BaseArray([arr.sum() for arr in self.arrays])

    def min(self) -> BaseArray:
        return BaseArray([arr.min() for arr in self.arrays])

    def max(self) -> BaseArray:
        return BaseArray([arr.max() for arr in self.arrays])

    def count(self) -> BaseArray:
        return BaseArray([len(arr) for arr in self.arrays])

    def std(self) -> BaseArray:
        return BaseArray([arr.std() for arr in self.arrays])
```

#### C. `BaseTable.group_by()` method

```python
# Add to BaseTable
def group_by(self, columns: Union[str, List[str]]) -> TableArray:
    """
    Group table by column(s), return TableArray.

    Args:
        columns: Column name(s) to group by

    Returns:
        TableArray: One table per unique combination of group key values

    Example:
        >>> gt.edges.group_by('object_name')
        TableArray[34 groups]
    """
    if isinstance(columns, str):
        columns = [columns]

    # Convert to pandas for grouping (temporary until Rust impl)
    df = self.to_pandas()

    tables = []
    keys = []

    for key, group_df in df.groupby(columns):
        # Convert back to BaseTable
        group_table = BaseTable.from_pandas(group_df)
        tables.append(group_table)
        keys.append(key)

    return TableArray(tables, keys)
```

### Phase 2: Aggregation Sugar

#### Auto-packaging to Table

When an `ArrayArray` aggregation produces a `BaseArray`, automatically package it with group keys into a `BaseTable`:

```python
# In ArrayArray methods:
def mean(self) -> Union[BaseArray, BaseTable]:
    """Calculate mean, auto-package with keys if available."""
    means = BaseArray([arr.mean() for arr in self.arrays])

    if self.keys is not None:
        # Package as table with group keys
        return BaseTable.from_dict({
            'group_key': self.keys,
            'mean': means.to_list()
        })

    return means
```

### Phase 3: Rust Implementation

Move to Rust for performance:
- `src/storage/table/group_by.rs`
- Columnar group-by with hash-based grouping
- Zero-copy slicing for groups
- Parallel aggregation

### Phase 4: SubgraphArray Extensions

Apply same patterns to `SubgraphArray`:

```python
components = g.connected_components()  # SubgraphArray

# Extract node attributes across all components
all_degrees = components['degree']  # TableArray or ArrayArray

# Aggregate per component
component_sizes = components.map(lambda sg: len(sg.nodes))
component_avg_degree = components['degree'].mean()

# Filter
large = components.filter(lambda sg: len(sg) > 100)
```

## SubgraphArray Design Details

### Core Concept

`SubgraphArray` is an array of subgraphs, typically returned from graph algorithms. It should support:
1. Column/attribute access across all subgraphs
2. Aggregation operations
3. Filtering and mapping
4. Iteration

### SubgraphArray Class

```python
class SubgraphArray:
    """Array of subgraphs from algorithms like connected_components, sample, etc."""

    def __init__(self, subgraphs: List[Subgraph], keys: Optional[List[Any]] = None):
        """
        Args:
            subgraphs: List of Subgraph objects
            keys: Optional keys/labels for each subgraph (e.g., component IDs)
        """
        self.subgraphs = subgraphs
        self.keys = keys or list(range(len(subgraphs)))

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, key):
        """
        Access subgraph by index or extract attribute from all subgraphs.

        Examples:
            components[0]        → First subgraph
            components['degree'] → ArrayArray of degree values from all subgraphs
        """
        if isinstance(key, int):
            return self.subgraphs[key]
        elif isinstance(key, str):
            # Extract node attribute from all subgraphs
            return self._extract_attribute(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def _extract_attribute(self, attr: str) -> 'ArrayArray':
        """
        Extract a node attribute from all subgraphs.

        Returns ArrayArray where each array is the attribute values
        from one subgraph.
        """
        arrays = []
        for sg in self.subgraphs:
            # Get attribute array from subgraph's nodes
            attr_array = sg.nodes[attr]  # Assumes nodes has __getitem__
            arrays.append(attr_array)

        return ArrayArray(arrays, self.keys)

    def __iter__(self):
        """Iterate over (key, subgraph) pairs."""
        return iter(zip(self.keys, self.subgraphs))

    # --- Aggregation Methods ---

    def map(self, func) -> BaseArray:
        """
        Apply function to each subgraph, return array of results.

        Example:
            sizes = components.map(lambda sg: len(sg.nodes))
            avg_degrees = components.map(lambda sg: sg['degree'].mean())
        """
        return BaseArray([func(sg) for sg in self.subgraphs])

    def filter(self, func) -> 'SubgraphArray':
        """
        Filter subgraphs by predicate.

        Example:
            large = components.filter(lambda sg: len(sg.nodes) > 10)
        """
        filtered = [(k, sg) for k, sg in zip(self.keys, self.subgraphs) if func(sg)]
        return SubgraphArray([sg for _, sg in filtered], [k for k, _ in filtered])

    def sample(self, n: int) -> 'SubgraphArray':
        """
        Random sample of n subgraphs.

        Example:
            components.sample(5)
        """
        import random
        indices = random.sample(range(len(self)), min(n, len(self)))
        return SubgraphArray(
            [self.subgraphs[i] for i in indices],
            [self.keys[i] for i in indices]
        )

    # --- Graph Operations ---

    def neighborhood(self, depth: int = 1) -> 'SubgraphArray':
        """
        Expand each subgraph by depth-hop neighborhood.

        Example:
            expanded = components.sample(5).neighborhood(depth=2)
        """
        expanded = [sg.neighborhood(depth) for sg in self.subgraphs]
        return SubgraphArray(expanded, self.keys)

    def merge(self) -> 'Graph':
        """
        Merge all subgraphs into a single graph.

        Example:
            filtered_graph = components.filter(lambda sg: len(sg) > 10).merge()
        """
        from .graph import Graph
        merged = Graph()
        for sg in self.subgraphs:
            merged.add_subgraph(sg)  # Assumes add_subgraph method
        return merged

    # --- Table Access ---

    def nodes_table(self) -> 'TableArray':
        """
        Get node tables from all subgraphs.

        Returns TableArray where each table is nodes from one subgraph.
        """
        tables = [sg.nodes.table() for sg in self.subgraphs]
        return TableArray(tables, self.keys)

    def edges_table(self) -> 'TableArray':
        """
        Get edge tables from all subgraphs.

        Returns TableArray where each table is edges from one subgraph.
        """
        tables = [sg.edges.table() for sg in self.subgraphs]
        return TableArray(tables, self.keys)

    # --- Summary Methods ---

    def summary(self) -> BaseTable:
        """
        Summary statistics for each subgraph.

        Returns:
            BaseTable with columns: [key, node_count, edge_count, density, ...]
        """
        data = {
            'key': self.keys,
            'node_count': [len(sg.nodes) for sg in self.subgraphs],
            'edge_count': [len(sg.edges) for sg in self.subgraphs],
            'density': [sg.density() for sg in self.subgraphs],
            'avg_degree': [sg['degree'].mean() if len(sg.nodes) > 0 else 0
                          for sg in self.subgraphs]
        }
        return BaseTable.from_dict(data)

    def describe(self):
        """Print summary statistics and return self for chaining."""
        summary = self.summary()
        print(f"SubgraphArray: {len(self)} subgraphs")
        print(summary.head(10))
        return self

    # --- Delegation (from original design) ---

    def table(self) -> BaseTable:
        """
        Convert to table with one row per subgraph.

        Equivalent to .summary()
        """
        return self.summary()

    def agg(self, spec: Dict[str, Union[str, List[str]]]) -> BaseTable:
        """
        Apply aggregations to node attributes across subgraphs.

        Example:
            components.agg({
                'degree': ['mean', 'max'],
                'pagerank': 'mean'
            })

        Returns table with: [key, degree_mean, degree_max, pagerank_mean]
        """
        result = {'key': self.keys}

        for attr, aggs in spec.items():
            if isinstance(aggs, str):
                aggs = [aggs]

            for agg_func in aggs:
                col_name = f"{attr}_{agg_func}"
                values = []

                for sg in self.subgraphs:
                    attr_array = sg.nodes[attr]
                    if agg_func == 'mean':
                        values.append(attr_array.mean())
                    elif agg_func == 'sum':
                        values.append(attr_array.sum())
                    elif agg_func == 'min':
                        values.append(attr_array.min())
                    elif agg_func == 'max':
                        values.append(attr_array.max())
                    elif agg_func == 'std':
                        values.append(attr_array.std())
                    elif agg_func == 'count':
                        values.append(len(attr_array))
                    else:
                        raise ValueError(f"Unknown aggregation: {agg_func}")

                result[col_name] = values

        return BaseTable.from_dict(result)
```

### SubgraphArray Usage Examples

#### Example 1: Component Analysis

```python
# Get connected components
components = g.connected_components()  # SubgraphArray
print(f"Found {len(components)} components")

# Summary statistics
components.describe()
# Output:
# SubgraphArray: 10 components
#    key  node_count  edge_count  density  avg_degree
# 0    0         234         567    0.021       4.846
# 1    1          12          15    0.227       2.500
# ...

# Filter to large components
large = components.filter(lambda sg: len(sg.nodes) > 100)

# Get average degree per component
avg_degrees = components['degree'].mean()
# Returns: BaseArray([4.846, 2.500, ...])

# Multiple aggregations
stats = components.agg({
    'degree': ['mean', 'max', 'std'],
    'pagerank': 'mean'
})
print(stats)
# Output:
#    key  degree_mean  degree_max  degree_std  pagerank_mean
# 0    0        4.846          23       3.421          0.004
# 1    1        2.500           5       1.234          0.083
```

#### Example 2: Sampling and Neighborhood Expansion

```python
# Sample 5 random components
sample = components.sample(5)

# Expand each by 2-hop neighborhood
expanded = sample.neighborhood(depth=2)

# Get node counts before and after
print("Before:", sample.map(lambda sg: len(sg.nodes)))
print("After:", expanded.map(lambda sg: len(sg.nodes)))

# Merge back into single graph
expanded_graph = expanded.merge()
```

#### Example 3: Chaining with Algorithms

```python
# Complex chain: components → sample → neighborhood → table → agg
result = (
    g.connected_components()          # SubgraphArray
     .sample(5)                       # Sample 5 components
     .neighborhood(depth=2)           # Expand each
     .agg({'degree': 'mean',          # Aggregate
           'pagerank': ['min', 'max']})
)
print(result)
# Output: Table with [key, degree_mean, pagerank_min, pagerank_max]
```

#### Example 4: Iterating Over Subgraphs

```python
# Explicit iteration
for component_id, subgraph in components:
    print(f"Component {component_id}:")
    print(f"  Nodes: {len(subgraph.nodes)}")
    print(f"  Edges: {len(subgraph.edges)}")
    print(f"  Avg degree: {subgraph['degree'].mean():.2f}")

    # Visualize large components
    if len(subgraph.nodes) > 50:
        subgraph.viz()
```

#### Example 5: Advanced Filtering

```python
# Filter by multiple criteria
interesting = components.filter(
    lambda sg: (
        len(sg.nodes) > 10 and
        len(sg.nodes) < 1000 and
        sg.density() > 0.1
    )
)

# Chain with more operations
hub_components = (
    interesting
    .filter(lambda sg: sg['degree'].max() > 50)  # Has a hub node
    .sample(3)                                    # Sample 3
)
```

#### Example 6: Table-Based Analysis

```python
# Get all node tables as TableArray
node_tables = components.nodes_table()

# Group by attributes within each component
for comp_id, table in node_tables:
    grouped = table.group_by('type')['value'].mean()
    print(f"Component {comp_id}:")
    print(grouped)
```

### Integration with Existing Delegation Chain

The key insight is that `SubgraphArray` fits naturally into Groggy's existing delegation pattern:

```python
# Current chain example from README
chain_result = (
    g.connected_components()          # SubgraphArray
     .sample(5)                       # SubgraphArray (5 components)
     .neighborhood(depth=2)           # SubgraphArray (expanded)
     .table()                         # BaseTable (summary)
     .agg({"weight": "mean"})         # BaseTable (aggregated)
)
```

### SubgraphArray vs TableArray

| Feature | SubgraphArray | TableArray |
|---------|---------------|------------|
| Contains | Subgraph objects | Table objects |
| From | Algorithms (components, sample) | group_by(), splits |
| `[attr]` returns | ArrayArray (node attrs) | ArrayArray (column) |
| `.map(func)` | Apply to subgraph | Apply to table |
| `.filter(func)` | Filter subgraphs | Filter tables |
| `.agg(spec)` | Aggregate node attrs | Aggregate columns |
| `.merge()` | Merge to Graph | Concat to Table |
| Graph ops | ✓ (neighborhood, etc.) | ✗ |

### Implementation Notes

1. **Memory Efficiency**: SubgraphArray should hold references/views, not copies
2. **Lazy Evaluation**: Consider lazy operations for large graphs
3. **Parallel Operations**: map/filter can be parallelized
4. **Attribute Access**: `sg['attr']` should access node attributes by default
5. **Edge Attributes**: Add `edges[attr]` accessor for edge attributes

## Design Decisions

### 1. Implicit vs Explicit Iteration

**Decision: Implicit by default, explicit available**

```python
# Implicit (automatic delegation)
result = gt.edges.group_by('object_name')['success'].mean()

# Explicit (manual control)
for key, group in gt.edges.group_by('object_name'):
    print(f"{key}: {group['success'].mean()}")
```

### 2. Return Types

| Operation | Input | Output | Notes |
|-----------|-------|--------|-------|
| `table.group_by(col)` | Table | TableArray | Groups |
| `table_array[col]` | TableArray | ArrayArray | Column extraction |
| `array_array.mean()` | ArrayArray | BaseArray | Scalar per group |
| Auto-package | BaseArray + keys | BaseTable | When keys present |

### 3. Naming Convention

- `TableArray` - array of tables (from group_by, splits, etc.)
- `ArrayArray` - array of arrays (intermediate, from column extraction)
- `SubgraphArray` - array of subgraphs (from algorithms, filters)

### 4. Pandas Compatibility

Match pandas patterns where intuitive:
```python
# Pandas
df.groupby('col')['value'].mean()

# Groggy
table.group_by('col')['value'].mean()
```

## Migration Path

1. **Now**: Document proposed API
2. **Week 1**: Implement `TableArray` and `ArrayArray` in Python
3. **Week 2**: Add `group_by()` to BaseTable (pandas-backed initially)
4. **Week 3**: Add aggregation auto-packaging
5. **Week 4**: Begin Rust migration for performance

## Open Questions

1. **Should `['column']` on TableArray return ArrayArray or TableArray?**
   - **Recommendation**: `ArrayArray` for aggregation chains
   - Alternative: `TableArray` with single column for further table ops

2. **How to handle missing columns in some groups?**
   - Fill with None/NaN
   - Skip groups
   - Error

3. **Should we support SQL-like syntax?**
   ```python
   gt.edges.query("SELECT object_name, AVG(success) GROUP BY object_name")
   ```
   - **Recommendation**: No, keep Pythonic

4. **Lazy vs Eager evaluation?**
   - Start eager for simplicity
   - Move to lazy later for optimization

## Summary

This design provides:
- ✅ Intuitive, pandas-like group-by syntax
- ✅ Chainable operations
- ✅ Automatic result packaging
- ✅ Explicit iteration when needed
- ✅ Consistent patterns across Table/Subgraph
- ✅ Clear migration path to Rust

Next step: Implement Phase 1 in Python, validate with user examples.
