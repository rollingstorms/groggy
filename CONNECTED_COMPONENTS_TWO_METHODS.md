# Connected Components: Two Methods Explained

## The Two Methods

### Method 1: `sg.connected_components()` (Direct Core Method)
```python
components = g.view().connected_components()
```

### Method 2: `sg.apply(community.connected_components())` (Algorithm Wrapper)
```python
result = g.apply(community.connected_components(output_attr='component'))
```

## Key Differences

| Aspect | Method 1 (Direct) | Method 2 (Wrapper) |
|--------|-------------------|-------------------|
| **Returns** | `ComponentsArray` (array of subgraphs) | `Subgraph` (with component attribute) |
| **Result Structure** | Separate subgraph per component | Same graph with component IDs |
| **Performance** | ⚡ 0.47ms (faster) | 0.94ms (2x slower, adds attributes) |
| **Use Case** | Analyze components separately | Add component info to nodes |

## Method 1: sg.connected_components() Details

**Returns**: `ComponentsArray` - A lazy array where each element is a component subgraph

**What you get:**
- Array of subgraphs, one per component
- Can iterate: `for comp in components: ...`
- Can get sizes: `components.sizes()` → `[(3, 2), (2, 1), ...]`
- Can get largest: `components.largest_component()`
- Can filter by size: `components.filter(lambda c: c.node_count() > 10)`

**Example:**
```python
components = g.view().connected_components()
print(f"Found {len(components)} components")

# Get largest component
largest = components.largest_component()
print(f"Largest: {largest.node_count()} nodes")

# Iterate over components
for i, comp in enumerate(components.to_list()):
    print(f"Component {i}: {comp.node_count()} nodes")
```

**When to use:**
- ✅ You want to analyze each component separately
- ✅ You need the largest component for further analysis
- ✅ You want to filter components by size
- ✅ Performance is critical
- ✅ You want component subgraphs, not just IDs

## Method 2: sg.apply(community.connected_components()) Details

**Returns**: `Subgraph` - The same graph with a 'component' attribute added to nodes

**What you get:**
- The full graph (same nodes and edges)
- Each node has a 'component' attribute with its component ID
- Can access IDs: `result.nodes['component']`
- Can chain with other algorithms
- Can use in visualization, grouping, etc.

**Example:**
```python
result = g.apply(community.connected_components(output_attr='component'))

# It's the full graph
print(f"Nodes: {result.node_count()}, Edges: {result.edge_count()}")

# Access component IDs
component_ids = result.nodes['component']
print(f"Component IDs: {list(component_ids)}")

# Use for visualization coloring
result.viz(node_color='component')  # Color by component

# Chain with other algorithms
result_with_pr = result.apply(pagerank(output_attr='pr'))
```

**When to use:**
- ✅ You need component IDs as node attributes
- ✅ You want to color/label nodes by component in visualization
- ✅ You're chaining multiple algorithms together
- ✅ You need to group/aggregate by component
- ✅ You want to use component info in queries/filters

## Performance Comparison (500 nodes, 2500 edges)

```
Direct method:     0.47ms (baseline)
Algorithm wrapper: 0.94ms (2x slower)
```

**Why is the wrapper slower?**
1. Calls the same core TraversalEngine
2. Adds overhead: Converting result to attributes
3. Sets 'component' attribute on every node
4. Returns full subgraph with attributes

**Is 2x overhead acceptable?**
YES! Because you get:
- Component IDs as attributes (useful for viz, grouping)
- Ability to chain with other algorithms
- Full subgraph API
- Integration with rest of groggy ecosystem

## Real-World Usage Examples

### Example 1: Get Largest Component (Use Method 1)
```python
# Fast and direct
components = g.view().connected_components()
largest = components.largest_component()

# Now work with just the largest component
pr_scores = largest.apply(pagerank(output_attr='score'))
```

### Example 2: Color Nodes by Component (Use Method 2)
```python
# Add component attribute
g_with_comp = g.apply(community.connected_components(output_attr='component'))

# Use in visualization
g_with_comp.viz(
    node_color='component',  # Color by component
    layout='force_directed'
)
```

### Example 3: Filter Small Components (Use Method 1)
```python
# Get components and filter
components = g.view().connected_components()

# Keep only components with 10+ nodes
large_comps = [c for c in components.to_list() if c.node_count() >= 10]

# Or use filter
large = components.filter(lambda c: c.node_count() >= 10)
```

### Example 4: Component Size Analysis (Use Method 1)
```python
components = g.view().connected_components()

# Get size distribution
sizes = [c.node_count() for c in components.to_list()]
print(f"Component sizes: {sorted(sizes, reverse=True)}")

# Get statistics
print(f"Total components: {len(components)}")
print(f"Largest: {max(sizes)} nodes")
print(f"Average: {sum(sizes) / len(sizes):.1f} nodes")
```

## Which Method Should You Use?

### Use Method 1 (`sg.connected_components()`) when:
- You need to **work with components as separate graphs**
- You want **the largest component** for analysis
- You need to **filter components by size**
- You want to **iterate over each component**
- **Performance matters** (it's 2x faster)

### Use Method 2 (`sg.apply(community.connected_components())`) when:
- You need **component IDs as node attributes**
- You're **visualizing and want to color by component**
- You're **chaining multiple algorithms**
- You want to **group/aggregate by component**
- You need **component info in queries/filters**

## Under the Hood

Both methods use the **same core TraversalEngine implementation**! They just return results differently:

**Method 1 flow:**
1. TraversalEngine.connected_components_for_nodes()
2. Returns ComponentsArray (lazy array of subgraphs)
3. Each component is a separate subgraph object

**Method 2 flow:**
1. TraversalEngine.connected_components_for_nodes()
2. Convert result to component ID mapping
3. Set 'component' attribute on all nodes
4. Return full subgraph with attributes

## Conclusion

Both methods are valid and useful for different scenarios:

- **Method 1 is faster and better for component-centric analysis**
- **Method 2 is more flexible for attribute-based workflows**

Choose based on your use case:
- Analyzing components → Method 1
- Visualizing/coloring → Method 2
- Chaining algorithms → Method 2
- Performance critical → Method 1
