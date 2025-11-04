# Algorithm Implementation Analysis: Core vs Algorithm Module

## Problem Statement

The benchmark shows that `g.view().connected_components()` (core method) is 2.5x faster than `g.apply(community.connected_components())` (algorithm wrapper). We need to make them use the same implementation so performance is identical.

## Current Implementation Comparison

### Core Method: `TraversalEngine::connected_components_for_nodes()`
**Location**: `src/query/traversal.rs`
**Algorithm**: **BFS (Breadth-First Search)**
**Approach**:
```rust
// 1. Get adjacency snapshot (zero-copy Arc reference)
let (_,_, _, _neighbors) = space.snapshot(pool);

// 2. Convert nodes to HashSet for O(1) lookups
let nodes_set: HashSet<NodeId> = nodes.iter().copied().collect();

// 3. BFS for each unvisited node
for &start_node in &nodes {
    if !visited.contains(&start_node) {
        let mut queue = VecDeque::new();
        queue.push_back(start_node);
        
        while let Some(current) = queue.pop_front() {
            // Get neighbors directly from adjacency map
            if let Some(current_neighbors) = _neighbors.get(&current) {
                for &(neighbor, edge_id) in current_neighbors {
                    // O(1) HashSet lookup
                    if nodes_set.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
}
```

**Key Optimizations**:
- ✅ Zero-copy adjacency snapshot (Arc reference)
- ✅ O(1) node membership checks (HashSet)
- ✅ Single-pass BFS
- ✅ Direct adjacency map access

### Algorithm Module: `ConnectedComponents::compute_undirected_or_weak()`
**Location**: `src/algorithms/community/components.rs`
**Algorithm**: **Union-Find**
**Approach**:
```rust
// 1. Initialize Union-Find
let mut uf = UnionFind::new(nodes);

// 2. Union all edges (❌ PROBLEM: calling subgraph.neighbors() in loop!)
for node in nodes {
    if let Ok(neighbors) = subgraph.neighbors(*node) {  // ← EXPENSIVE!
        for neighbor in neighbors {
            uf.union(*node, neighbor);
        }
    }
}

// 3. Get components
let components = uf.get_components();
```

**Problems**:
- ❌ Calling `subgraph.neighbors(*node)` for EVERY node (overhead per call)
- ❌ Each `neighbors()` call may create temporary structures
- ❌ Not using adjacency snapshot efficiently
- ⚠️ Union-Find should theoretically be faster (O(m α(n))), but overhead kills it

## Performance Analysis

### Why Core is Faster
1. **Zero-copy adjacency access**: Uses Arc reference to shared adjacency map
2. **Batch setup**: Creates node HashSet once, reuses it
3. **Direct iteration**: No function call overhead per node
4. **Optimized BFS**: Well-tuned queue-based traversal

### Why Algorithm is Slower
1. **Per-node function calls**: `subgraph.neighbors()` called N times
2. **Possible allocations**: Each neighbors() call may allocate
3. **Indirection**: Going through SubgraphOperations trait
4. **Union-Find overhead**: While asymptotically better, constant factors matter

## Solution Options

### Option 1: Make Algorithm Use Core Implementation (RECOMMENDED)
Make `ConnectedComponents::compute_undirected_or_weak()` call the TraversalEngine directly:

```rust
fn compute_undirected_or_weak(
    &self,
    subgraph: &Subgraph,
    nodes: &[NodeId],
) -> Result<HashMap<NodeId, i64>> {
    // Use the fast TraversalEngine directly!
    let graph = subgraph.graph();
    let graph_ref = graph.borrow();
    let pool = graph_ref.pool();
    let space = graph_ref.space();
    
    let mut traversal_engine = TraversalEngine::new();
    let options = TraversalOptions::default();
    
    let result = traversal_engine.connected_components_for_nodes(
        pool,
        space,
        nodes.to_vec(),
        options,
    )?;
    
    // Convert result to component ID mapping
    let mut node_to_component = HashMap::new();
    for (component_id, component) in result.components.into_iter().enumerate() {
        for node in component.nodes {
            node_to_component.insert(node, component_id as i64);
        }
    }
    
    Ok(node_to_component)
}
```

**Pros**:
- ✅ Same performance as core method
- ✅ Reuses optimized code
- ✅ Single source of truth
- ✅ Easy to maintain

**Cons**:
- Changes algorithm implementation
- Loses Union-Find approach (but BFS is equally correct and faster in practice)

### Option 2: Optimize Union-Find Implementation
Get adjacency snapshot once and iterate efficiently:

```rust
fn compute_undirected_or_weak(
    &self,
    subgraph: &Subgraph,
    nodes: &[NodeId],
) -> Result<HashMap<NodeId, i64>> {
    let mut uf = UnionFind::new(nodes);
    
    // Get adjacency snapshot ONCE (like core method)
    let graph = subgraph.graph();
    let graph_ref = graph.borrow();
    let pool = graph_ref.pool();
    let space = graph_ref.space();
    let (_, _, _, neighbors_map) = space.snapshot(pool);
    
    // Build node set for O(1) lookups
    let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
    
    // Union all edges efficiently
    for &node in nodes {
        if let Some(neighbors) = neighbors_map.get(&node) {
            for &(neighbor, _edge_id) in neighbors {
                if node_set.contains(&neighbor) {
                    uf.union(node, neighbor);
                }
            }
        }
    }
    
    // Get components
    let components = uf.get_components();
    let mut node_to_component = HashMap::new();
    for (component_id, (_, members)) in components.iter().enumerate() {
        for &node in members {
            node_to_component.insert(node, component_id as i64);
        }
    }
    
    Ok(node_to_component)
}
```

**Pros**:
- ✅ Keeps Union-Find approach
- ✅ Should be very fast (O(m α(n)) ≈ O(m))
- ✅ Removes per-node function call overhead

**Cons**:
- Requires understanding of graph internals
- Two implementations to maintain

### Option 3: Expose Fast Path in SubgraphOperations
Add a `neighbors_bulk()` or `adjacency_snapshot()` method to avoid per-call overhead.

**Pros**:
- ✅ Improves API for all algorithms
- ✅ Maintains abstraction

**Cons**:
- More complex API change
- Still two implementations

## Recommendation

**Use Option 1: Delegate to Core Implementation**

This is the best solution because:
1. Eliminates code duplication
2. Ensures performance parity
3. Single source of truth for the algorithm
4. Easy to understand and maintain

The Union-Find approach is theoretically elegant, but in practice the BFS implementation in TraversalEngine is already highly optimized and just as fast (or faster due to lower constant factors).

## Implementation Steps

1. **Immediate**: Update `ConnectedComponents::compute_undirected_or_weak()` to use TraversalEngine
2. **Test**: Verify performance matches core method
3. **Repeat**: Apply same pattern to other algorithms (betweenness, pagerank, etc.)
4. **Document**: Make it clear that algorithm wrappers should delegate to optimized core implementations

## Other Algorithms to Check

Once connected_components is fixed, check if other algorithms have the same issue:

- **Betweenness**: Is it using optimized shortest paths or reimplementing?
- **PageRank**: Already fast - probably using core correctly
- **LPA**: Check if it's using efficient neighbor iteration
- **Louvain/Leiden**: Check modularity computation efficiency

The pattern should be: **Algorithm wrappers should orchestrate and configure, not reimplement.**
