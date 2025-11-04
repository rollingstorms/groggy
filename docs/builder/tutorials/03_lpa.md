# Tutorial 3: Label Propagation Algorithm

In this tutorial, you'll learn how to implement algorithms with **asynchronous updates** using mode-based aggregation. We'll implement the Label Propagation Algorithm (LPA) for community detection.

## What You'll Learn

- Mode aggregation (most common value)
- Asynchronous updates with `neighbor_mode_update`
- When to use async vs sync updates
- Community detection basics

## Prerequisites

- Complete [Tutorial 2: PageRank](02_pagerank.md)
- Understanding of community detection
- Familiarity with iterative algorithms

## What is Label Propagation?

Label Propagation is a simple yet effective community detection algorithm:
1. Each node starts with a unique label
2. In each iteration, nodes adopt the most common label among their neighbors
3. After a few iterations, densely connected groups converge to the same label

**Key insight:** Nodes in the same community will reinforce each other's labels.

## Sync vs Async Updates

Before we dive in, it's important to understand two update strategies:

### Synchronous Updates (like PageRank)
- Compute all new values based on old values
- Update all nodes at once
- More predictable, easier to reason about
- May converge slower

### Asynchronous Updates (LPA)
- Update nodes one at a time or in random order
- Each update immediately affects subsequent updates
- Often converges faster
- More realistic for some scenarios (e.g., gossip protocols)

LPA typically uses **asynchronous updates** because labels spread faster through the network.

## Step 1: Basic Structure

```python
from groggy.builder import algorithm

@algorithm("community")
def label_propagation(sG, max_iter=100):
    """Detect communities using label propagation."""
    pass  # We'll fill this in
```

## Step 2: Initialize Unique Labels

Each node starts with a unique label (its node ID):

```python
@algorithm("community")
def label_propagation(sG, max_iter=100):
    """Detect communities using label propagation."""
    # Initialize each node with unique label (node index)
    labels = G.nodes(unique=True)
    
    return labels
```

**What's happening:**
- `G.nodes(unique=True)` assigns each node its index (0, 1, 2, ...)
- This gives every node a unique starting label

## Step 3: Mode Aggregation

The core of LPA is finding the most common label among neighbors:

```python
@algorithm("community")
def label_propagation(sG, max_iter=100):
    """Detect communities using label propagation."""
    # Initialize unique labels
    labels = G.nodes(unique=True)
    
    # Iterate
    with G.builder.iter.loop(max_iter):
        # For each node, adopt the most common neighbor label
        new_labels = G.builder.graph.neighbor_agg(labels, "mode")
        
        # Update labels
        labels = G.builder.var("labels", new_labels)
    
    return labels
```

**Key concept:**

### Mode Aggregation
```python
new_labels = G.builder.graph.neighbor_agg(labels, "mode")
```

`"mode"` finds the most frequent value among neighbors. If there's a tie, it picks one (deterministically based on value).

**Other aggregation options:**
- `"sum"`: Add neighbor values (used in PageRank)
- `"mean"`: Average neighbor values
- `"max"`: Maximum neighbor value
- `"min"`: Minimum neighbor value
- `"mode"`: Most common neighbor value (LPA)

## Step 4: Async Updates (Advanced)

The above version uses synchronous updates (all nodes update at once). For true asynchronous LPA, we use a special operation:

```python
@algorithm("community")
def label_propagation(sG, max_iter=100):
    """Detect communities using label propagation (async updates)."""
    # Initialize unique labels
    labels = G.nodes(unique=True)
    
    # Iterate with asynchronous updates
    with G.builder.iter.loop(max_iter):
        # Asynchronously update each node to most common neighbor label
        labels = G.builder.graph.neighbor_mode_update(labels)
    
    return labels
```

**What's different:**

### `neighbor_mode_update()`
```python
labels = G.builder.graph.neighbor_mode_update(labels)
```

This is a fused operation that:
1. For each node (in some order), finds the mode of neighbor labels
2. Immediately updates that node's label
3. Later nodes see the updated labels

This is more efficient than separate aggregation + reassignment.

**Note:** The node order affects results! Groggy uses a deterministic order for reproducibility.

## Complete Example: Karate Club

Let's apply LPA to the famous Zachary's Karate Club network:

```python
from groggy.builder import algorithm
import groggy as gg

@algorithm("community")
def label_propagation(sG, max_iter=100):
    """
    Detect communities using label propagation with async updates.
    
    Args:
        G: Graph handle (injected by decorator)
        max_iter: Maximum iterations (default: 100)
    
    Returns:
        Community labels for each node
    """
    # Start with unique labels
    labels = G.nodes(unique=True)
    
    # Iterate with asynchronous mode updates
    with G.builder.iter.loop(max_iter):
        labels = G.builder.graph.neighbor_mode_update(labels)
    
    return labels

# Load Karate Club network (34 nodes, 2 known communities)
G = gg.Graph()
# Add edges for the karate club network
# (In practice, you'd load from a file or dataset)
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]
G.add_edges_from(edges)

# Run label propagation
lpa = label_propagation(max_iter=10)
result = G.all().apply(lpa)

# Analyze communities
communities = result.nodes()["community"]

# Group nodes by community
from collections import defaultdict
comm_groups = defaultdict(list)
for node, comm in communities.items():
    comm_groups[comm].append(node)

print(f"Found {len(comm_groups)} communities:")
for comm_id, members in sorted(comm_groups.items()):
    print(f"  Community {comm_id}: {sorted(members)}")
```

Expected output (may vary slightly due to ties):
```
Found 2 communities:
  Community 0: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
  Community 32: [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
```

The algorithm found the two main factions in the karate club!

## How It Works: Step by Step

Let's trace what happens in a small example:

**Initial graph:**
```
    1 --- 2 --- 3
    |     |     |
    4 --- 5 --- 6
```

**Iteration 0 (initial labels):**
```
Node: 1 2 3 4 5 6
Label: 1 2 3 4 5 6
```

**Iteration 1 (each node looks at neighbors):**
```
Node 1: neighbors are {2, 4} → labels {2, 4} → mode = 2 (or 4, tie) → label = 2
Node 2: neighbors are {1, 3, 5} → labels {1, 3, 5} → mode = 1 (using async update from node 1!) → label = 1
Node 3: neighbors are {2, 6} → labels {1, 6} → mode = 1 → label = 1
Node 4: neighbors are {1, 5} → labels {2, 5} → mode = 2 → label = 2
Node 5: neighbors are {2, 4, 6} → labels {1, 2, 6} → mode = 1 → label = 1
Node 6: neighbors are {3, 5} → labels {1, 1} → mode = 1 → label = 1
```

**After iteration 1:**
```
Node: 1 2 3 4 5 6
Label: 2 1 1 2 1 1
```

Notice how nodes 2, 3, 5, 6 quickly converged to label 1! After a few more iterations, the network splits into communities.

## Handling Ties

What if two labels appear equally often?

```python
# Node has neighbors with labels: [5, 5, 7, 7]
# Both 5 and 7 appear twice - which to choose?
```

Groggy's `"mode"` aggregation handles ties **deterministically**:
- It picks the smallest value (in this case, 5)
- This ensures reproducible results

If you want randomness:
```python
# Future feature: random tie-breaking
new_labels = G.builder.graph.neighbor_agg(labels, "mode", tie_break="random")
```

## Synchronous vs Asynchronous: A Comparison

### Synchronous LPA
```python
@algorithm("community_sync")
def lpa_sync(G, max_iter=100):
    labels = G.nodes(unique=True)
    with G.builder.iter.loop(max_iter):
        new_labels = G.builder.graph.neighbor_agg(labels, "mode")
        labels = G.builder.var("labels", new_labels)
    return labels
```

**Characteristics:**
- All nodes update simultaneously based on previous iteration
- More iterations to converge
- Results independent of node ordering
- Good for theoretical analysis

### Asynchronous LPA
```python
@algorithm("community_async")
def lpa_async(G, max_iter=100):
    labels = G.nodes(unique=True)
    with G.builder.iter.loop(max_iter):
        labels = G.builder.graph.neighbor_mode_update(labels)
    return labels
```

**Characteristics:**
- Nodes update one at a time, seeing latest values
- Fewer iterations to converge
- Results depend on node order (but deterministic in Groggy)
- More realistic for distributed settings

**Performance:** Async typically converges in 5-10 iterations vs 20-50 for sync.

## Exercises

1. **Weighted LPA**: Give more weight to neighbors with stronger connections
   ```python
   # Load edge weights
   weights = G.builder.attr.load_edge("weight", default=1.0)
   
   # Use in mode aggregation (weights affect voting)
   new_labels = G.builder.graph.neighbor_agg(labels, "mode", weights=weights)
   ```

2. **Early stopping**: Stop when labels don't change
   - Count how many labels changed in each iteration
   - Stop if change_count < threshold
   
   ```python
   # Hint: compare old and new labels
   changed = (labels != new_labels)
   change_count = changed.reduce("sum")
   # (Need convergence detection - future feature)
   ```

3. **Multi-scale communities**: Run LPA on the output to find sub-communities
   ```python
   # First pass: find major communities
   communities_1 = lpa_async(max_iter=10)
   
   # Second pass: for each community, find sub-communities
   # (Requires subgraph operations)
   ```

4. **Compare community quality**: Compute modularity of detected communities
   ```python
   # Modularity measures how well-separated communities are
   # Q = (edges_within - expected_edges) / total_edges
   ```

## Key Takeaways

✅ Use `"mode"` aggregation to find the most common neighbor value  
✅ Use `neighbor_mode_update()` for efficient asynchronous updates  
✅ Async updates often converge faster than sync updates  
✅ `G.nodes(unique=True)` initializes nodes with unique IDs  
✅ LPA is simple but effective for community detection  
✅ Tie-breaking is deterministic for reproducibility  

## Common Mistakes

❌ **Using mean instead of mode**
```python
# Wrong for LPA - mean of labels doesn't make sense!
new_labels = G.builder.graph.neighbor_agg(labels, "mean")
```

✅ **Correct - use mode for label propagation**
```python
new_labels = G.builder.graph.neighbor_agg(labels, "mode")
```

❌ **Expecting fast convergence with sync updates**
```python
# Sync may need 50+ iterations
lpa_sync(max_iter=10)  # Probably not enough!
```

✅ **Use async for faster convergence**
```python
# Async typically converges in 10-20 iterations
lpa_async(max_iter=20)
```

❌ **Forgetting that labels are arbitrary**
```python
# Wrong - label numbers don't have meaning!
if communities[node] > 5:
    print("Node is in a high community")
```

✅ **Treat labels as categories**
```python
# Correct - compare equality, not magnitude
if communities[node] == communities[other_node]:
    print("Nodes are in the same community")
```

## Performance Tips

1. **Fewer iterations needed**: LPA converges quickly (10-20 iterations usually enough)
2. **Node order matters**: Async updates are faster but order-dependent
3. **Large networks**: LPA scales well to millions of nodes
4. **Quality vs speed**: More iterations → more stable communities

## Next Steps

- [Tutorial 4: Custom Metrics](04_custom_metrics.md) - Build complex node/edge metrics
- [API Reference: GraphOps](../api/graph.md) - Learn about all aggregation options
- [Migration Guide](../guides/migration.md) - Convert old builder code to new DSL

---

**Ready to build custom metrics?** Continue to [Tutorial 4: Custom Metrics](04_custom_metrics.md)!
