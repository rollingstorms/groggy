# AttrOps API

`AttrOps` provides attribute I/O operations: loading node and edge attributes, saving results, and managing the algorithm's attribute space.

## Overview

Access AttrOps through `sG.builder.attr`:

```python
@algorithm
def example(sG):
    # Load attribute
    weights = sG.builder.attr.load("weight", default=1.0)
    
    # Process
    result = weights * 2.0
    
    # Save (or just return)
    return result
```

## Loading Attributes

### `load(name, default=0.0)`

Load a node attribute from the graph.

```python
values = sG.builder.attr.load("pagerank", default=0.0)
```

**Parameters:**
- `name`: Attribute name (str)
- `default`: Default value for nodes without the attribute (default: 0.0)

**Returns:** VarHandle (node values)

**Behavior:**
- If attribute exists: loads values from graph
- If attribute missing: returns default value for all nodes
- If some nodes missing attribute: uses default for those nodes

**Examples:**

**Load existing attribute:**
```python
@algorithm
def normalize_pagerank(sG):
    pr = sG.builder.attr.load("pagerank", default=0.0)
    return pr.normalize()
```

**Use default for missing:**
```python
@algorithm
def increment_counter(sG):
    counter = sG.builder.attr.load("visit_count", default=0.0)
    return counter + 1.0
```

**Load and process:**
```python
@algorithm
def scale_attribute(sG, attr_name, scale_factor):
    values = sG.builder.attr.load(attr_name, default=0.0)
    return values * scale_factor
```

### `load_edge(name, default=0.0)`

Load an edge attribute from the graph.

```python
weights = sG.builder.attr.load_edge("weight", default=1.0)
```

**Parameters:**
- `name`: Edge attribute name (str)
- `default`: Default value for edges without the attribute (default: 0.0)

**Returns:** VarHandle (edge values)

**Example - Weighted PageRank:**
```python
@algorithm
def weighted_pagerank(sG, damping=0.85, max_iter=100):
    # Load edge weights
    weights = sG.builder.attr.load_edge("weight", default=1.0)
    
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.loop(max_iter):
        contrib = ranks / (deg + 1e-9)
        
        # Weighted neighbor aggregation
        neighbor_sum = sG.builder.graph_ops.neighbor_agg(
            contrib, agg="sum", weights=weights
        )
        
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / sG.N
        )
    
    return ranks.normalize()
```

**Example - Edge filtering:**
```python
@algorithm
def strong_edges_only(sG, threshold=0.5):
    weights = sG.builder.attr.load_edge("weight", default=0.0)
    is_strong = weights > threshold
    
    # Filter to strong edges
    strong_sg = sG.builder.graph_ops.filter_edges(is_strong)
    return is_strong
```

### `load_optional(name)`

Load attribute if it exists, return None if missing.

```python
values = sG.builder.attr.load_optional("pagerank")
if values is not None:
    # Attribute exists
    pass
```

**Parameters:**
- `name`: Attribute name (str)

**Returns:** VarHandle if attribute exists, None otherwise

**Example - Conditional processing:**
```python
@algorithm
def use_cache_if_available(sG):
    cached = sG.builder.attr.load_optional("cached_result")
    
    if cached is not None:
        return cached
    
    # Compute from scratch
    result = sG.nodes(1.0)
    # ... expensive computation ...
    return result
```

**Note:** This is pseudocode. The builder doesn't support Python conditionals directly. Use for checking in decorator logic, not in algorithm body.

## Saving Attributes

### `attach(name, values)`

Attach computed values as a node attribute (saved to output).

```python
sG.builder.attr.attach("pagerank", ranks)
```

**Parameters:**
- `name`: Attribute name (str)
- `values`: VarHandle (values to save)

**Returns:** None

**Behavior:**
- Values are attached to the result subgraph
- Accessible after `subgraph.apply(algo)` via `result.nodes.data("pagerank")`

**Example:**
```python
@algorithm
def multi_output(sG):
    # Compute multiple metrics
    deg = sG.builder.graph_ops.degree()
    pr = pagerank_computation(sG)
    
    # Attach all results
    sG.builder.attr.attach("degree", deg)
    sG.builder.attr.attach("pagerank", pr)
    sG.builder.attr.attach("combined", deg + pr)
    
    # Return primary result (also attached automatically)
    return pr
```

**Decorator auto-attach:**
```python
@algorithm
def my_algo(sG):
    result = sG.nodes(1.0)
    return result  # Automatically attached as "output"
```

### `attach_edge(name, values)`

Attach computed values as an edge attribute.

```python
sG.builder.attr.attach_edge("flow", edge_flows)
```

**Parameters:**
- `name`: Edge attribute name (str)
- `values`: VarHandle (edge values to save)

**Returns:** None

**Example - Edge scores:**
```python
@algorithm
def edge_importance(sG):
    # Load source and target node values
    node_values = sG.builder.attr.load("pagerank", default=0.0)
    
    # Compute edge scores
    # (This is simplified - real impl would need source/target access)
    edge_scores = node_values.reduce("mean")  # Placeholder
    
    sG.builder.attr.attach_edge("importance", edge_scores)
    return edge_scores
```

## Attribute Metadata

### `has_attribute(name)`

Check if attribute exists (for decorator logic).

```python
exists = sG.builder.attr.has_attribute("pagerank")
```

**Parameters:**
- `name`: Attribute name (str)

**Returns:** bool

**Note:** This is for checking in Python code before building, not in the algorithm IR itself.

### `list_attributes()`

List all available node attributes.

```python
attrs = sG.builder.attr.list_attributes()
# Returns: ['pagerank', 'degree', 'community', ...]
```

**Returns:** List of attribute names

**Example - Dynamic processing:**
```python
def process_all_numeric_attributes(subgraph):
    attrs = subgraph.builder.attr.list_attributes()
    results = {}
    
    for attr in attrs:
        @algorithm
        def normalize(sG, attr_name=attr):
            values = sG.builder.attr.load(attr_name)
            return values.normalize()
        
        algo = normalize()
        results[attr] = subgraph.apply(algo)
    
    return results
```

### `attribute_type(name)`

Get attribute type (node or edge).

```python
attr_type = sG.builder.attr.attribute_type("weight")
# Returns: "node" or "edge"
```

**Parameters:**
- `name`: Attribute name (str)

**Returns:** str ("node" or "edge")

## Attribute Initialization

### Init from graph handle

The `sG.nodes()` method creates initialized node values:

```python
# Uniform values
uniform = sG.nodes(1.0)

# Unique IDs (0, 1, 2, ...)
ids = sG.nodes(unique=True)

# Computed value
inv_n = sG.nodes(1.0 / sG.N)
```

These are NOT loading from attributes - they create new values.

**See:** Graph/node accessors in the main guide for `sG.nodes()` details.

## Common Patterns

### Load-Process-Save

```python
@algorithm
def process_attribute(sG, attr_name):
    # Load
    values = sG.builder.attr.load(attr_name, default=0.0)
    
    # Process
    processed = values * 2.0 + 1.0
    
    # Save (via return - decorator handles attach)
    return processed
```

### Multi-Attribute Computation

```python
@algorithm
def combined_score(sG, weights=(0.5, 0.3, 0.2)):
    # Load multiple attributes
    pr = sG.builder.attr.load("pagerank", default=0.0)
    deg = sG.builder.attr.load("degree", default=0.0)
    cc = sG.builder.attr.load("clustering", default=0.0)
    
    # Normalize each
    pr_norm = pr.normalize()
    deg_norm = deg.normalize()
    cc_norm = cc.normalize()
    
    # Weighted combination
    score = weights[0] * pr_norm + weights[1] * deg_norm + weights[2] * cc_norm
    
    return score
```

### Conditional Default

```python
@algorithm
def use_attribute_or_compute(sG, attr_name):
    # Try to load
    values = sG.builder.attr.load(attr_name, default=-1.0)
    
    # Check if loaded (values != -1)
    has_data = values > -1.0
    
    # If missing, compute
    computed = sG.builder.graph_ops.degree()
    
    # Use loaded if available, else computed
    result = has_data.where(values, computed)
    return result
```

### Incremental Update

```python
@algorithm
def accumulate(sG, attr_name, increment):
    # Load current value
    current = sG.builder.attr.load(attr_name, default=0.0)
    
    # Increment
    updated = current + increment
    
    # Return (will be attached)
    return updated
```

## Edge Attribute Patterns

### Weighted Aggregation

```python
@algorithm
def weighted_neighbor_sum(sG, value_attr, weight_attr):
    # Load node values and edge weights
    values = sG.builder.attr.load(value_attr, default=0.0)
    weights = sG.builder.attr.load_edge(weight_attr, default=1.0)
    
    # Weighted aggregation
    result = sG.builder.graph_ops.neighbor_agg(
        values, agg="sum", weights=weights
    )
    
    return result
```

### Edge Value Computation

```python
@algorithm
def compute_edge_similarity(sG, feature_attr):
    # Load node features
    features = sG.builder.attr.load(feature_attr, default=0.0)
    
    # Compute edge similarity (simplified - needs source/target)
    # Real implementation would compute per-edge similarity
    
    # Placeholder
    similarity = features.reduce("mean")
    
    sG.builder.attr.attach_edge("similarity", similarity)
    return similarity
```

## Performance Notes

### Attribute Loading

Loading attributes is efficient (zero-copy where possible):

```python
# Fast - direct reference
values = sG.builder.attr.load("pagerank")

# Also fast - default only used if attribute missing
values = sG.builder.attr.load("pagerank", default=0.0)
```

### Multiple Loads

Loading the same attribute multiple times is fine:

```python
# These share underlying data:
a = sG.builder.attr.load("weight")
b = sG.builder.attr.load("weight")
```

### Attach Overhead

Attaching attributes is a metadata operation (minimal overhead):

```python
# Fast - just marks variable for output
sG.builder.attr.attach("result", values)
```

## Limitations

### No Dynamic Attribute Names

Attribute names must be known at algorithm definition time:

```python
# ❌ This doesn't work:
for attr in some_list:
    values = sG.builder.attr.load(attr)  # Can't be dynamic in IR

# ✅ Do this instead:
# Load all attributes in decorator logic, pass as parameters
```

### No Attribute Deletion

Can't delete attributes in the builder:

```python
# ❌ No delete operation
# Attributes persist from input to output
```

### Edge Attributes in Aggregation

Edge attributes can only be used as weights in aggregation:

```python
# ✅ Works:
weighted_sum = sG.builder.graph_ops.neighbor_agg(values, weights=edge_attr)

# ❌ Can't do arbitrary edge operations:
# edge_result = edge_attr * 2.0  # Edge VarHandles not fully supported
```

## Migration from Old API

### Before (explicit attach):

```python
builder = AlgorithmBuilder("my_algo")
values = builder.init_nodes(1.0)
result = builder.core.add(values, 1.0)
builder.attach_as("output", result)
algo = builder.build()
```

### After (decorator auto-attach):

```python
@algorithm
def my_algo(sG):
    values = sG.nodes(1.0)
    result = values + 1.0
    return result  # Automatically attached
```

### Multiple outputs:

**Before:**
```python
builder.attach_as("metric1", m1)
builder.attach_as("metric2", m2)
```

**After:**
```python
sG.builder.attr.attach("metric1", m1)
sG.builder.attr.attach("metric2", m2)
return m1  # Primary output
```

## Examples

### Attribute Normalization

```python
@algorithm
def normalize_attribute(sG, attr_name):
    """Normalize an attribute to [0, 1] range."""
    values = sG.builder.attr.load(attr_name, default=0.0)
    
    min_val = values.reduce("min")
    max_val = values.reduce("max")
    
    normalized = (values - min_val) / (max_val - min_val + 1e-9)
    return normalized
```

### Attribute Combination

```python
@algorithm
def rank_fusion(sG, alpha=0.5):
    """Combine PageRank and degree centrality."""
    pr = sG.builder.attr.load("pagerank", default=0.0)
    deg = sG.builder.attr.load("degree", default=0.0)
    
    # Normalize both
    pr_norm = pr / pr.reduce("max")
    deg_norm = deg / deg.reduce("max")
    
    # Fuse
    fused = alpha * pr_norm + (1 - alpha) * deg_norm
    
    return fused
```

### Attribute Propagation

```python
@algorithm
def propagate_attribute(sG, attr_name, decay=0.9, max_iter=10):
    """Propagate attribute values through graph."""
    initial = sG.builder.attr.load(attr_name, default=0.0)
    values = initial
    
    with sG.builder.iter.loop(max_iter):
        neighbor_avg = sG.builder.graph_ops.neighbor_agg(values, "mean")
        values = sG.builder.var("values",
            decay * neighbor_avg + (1 - decay) * initial
        )
    
    return values
```

### Weighted Degree with Attributes

```python
@algorithm
def weighted_degree(sG, weight_attr="weight"):
    """Compute weighted degree from edge attribute."""
    weights = sG.builder.attr.load_edge(weight_attr, default=1.0)
    
    # Sum edge weights for each node
    ones = sG.nodes(1.0)
    weighted_deg = sG.builder.graph_ops.neighbor_agg(
        ones, agg="sum", weights=weights
    )
    
    return weighted_deg
```

## See Also

- [CoreOps API](core.md) - Operations on attribute values
- [GraphOps API](graph.md) - Using attributes in neighbor aggregation
- [Custom Metrics Tutorial](../tutorials/04_custom_metrics.md) - Building algorithms with attributes
