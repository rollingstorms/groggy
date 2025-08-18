# Graph Cache Architecture Plan

## Problem Analysis

### Current Cache Systems
We currently have **TWO separate cache systems** that can become inconsistent:

1. **Topology Cache** (in `GraphSpace`)
   - Format: Columnar vectors `(edge_ids[], sources[], targets[])`
   - Used by: Graph API methods (`neighbors()`, `degree()`), subgraph operations
   - Rebuilt: When `get_columnar_topology_with_rebuild()` called
   - Invalidated: When edges added/removed via `activate_edge()`, `deactivate_edge()`

2. **Adjacency Cache** (in `TraversalEngine`) 
   - Format: Adjacency list `HashMap<NodeId, Vec<(NodeId, EdgeId)>>`
   - Used by: BFS, DFS, Connected Components algorithms
   - Rebuilt: When topology generation changes, reads from topology cache
   - Invalidated: When topology generation increments

### Root Cause: Cache Coherency Problem

**The Issue**: Adjacency cache rebuilds FROM the topology cache, but the topology cache might be dirty when read:

```
1. Edge added → topology_cache_dirty = true, generation++
2. BFS called → checks adjacency cache generation  
3. Adjacency cache rebuilds → reads DIRTY topology cache
4. Adjacency cache gets STALE data
5. BFS returns wrong results
```

Meanwhile:
```
1. Graph API called → checks topology cache, rebuilds if needed
2. Returns FRESH data (different from what BFS saw)
```

### Performance Issues

1. **Connected Components**: Calls `neighbors()` for every node → each call might trigger cache rebuilding → O(n²) behavior
2. **Double Cache Overhead**: Memory and rebuild costs for two separate cache systems
3. **Inconsistent Fast/Slow Paths**: Some operations get fresh data, others get stale data

## Architecture Options

### Option 1: Unified Cache with Rebuild Coordination

**Strategy**: Make topology cache the single source of truth, coordinate rebuilds

**Implementation**:
```rust
// In GraphSpace - central cache manager
impl GraphSpace {
    /// Ensure topology cache is fresh and return immutable reference
    pub fn get_fresh_topology(&mut self, pool: &GraphPool) -> (&[EdgeId], &[NodeId], &[NodeId]) {
        if self.topology_cache_dirty {
            self.rebuild_topology_cache(pool);
        }
        self.get_columnar_topology()
    }
    
    /// Get topology without rebuilding (for immutable contexts)
    pub fn get_topology_if_clean(&self) -> Option<(&[EdgeId], &[NodeId], &[NodeId])> {
        if !self.topology_cache_dirty {
            Some(self.get_columnar_topology())
        } else {
            None
        }
    }
}

// In Graph API - coordinate cache access
impl Graph {
    pub fn neighbors(&self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
        // Try clean cache first
        if let Some((_, sources, targets)) = self.space.get_topology_if_clean() {
            // Fast path - use clean cache
            return self.neighbors_from_topology(node, sources, targets);
        }
        // Must use mutable access to rebuild
        // This means we need to change the API or use interior mutability
    }
}

// TraversalEngine gets fresh data
impl TraversalEngine {
    fn ensure_adjacency_cache(&mut self, space: &mut GraphSpace, pool: &GraphPool) {
        let topology_generation = space.get_topology_generation();
        if !self.adjacency_cache.is_up_to_date(topology_generation) {
            let (edge_ids, sources, targets) = space.get_fresh_topology(pool);
            self.adjacency_cache.rebuild(edge_ids, sources, targets, topology_generation);
        }
    }
}
```

**Pros**: 
- Single source of truth
- Consistent data across all operations
- Can optimize rebuild timing

**Cons**:
- Requires API changes (mutable access for cache rebuilding)
- More complex coordination logic

### Option 2: Always-Fresh Strategy (Simplified)

**Strategy**: Remove lazy caching, always rebuild when needed

**Implementation**:
```rust
// In GraphSpace - always rebuild if dirty
impl GraphSpace {
    pub fn get_columnar_topology(&mut self, pool: &GraphPool) -> (&[EdgeId], &[NodeId], &[NodeId]) {
        if self.topology_cache_dirty {
            self.rebuild_topology_cache(pool);
        }
        (&self.active_edge_ids, &self.edge_sources, &self.edge_targets)
    }
    
    // Remove the immutable version that can return stale data
}

// In Graph API - always get fresh data
impl Graph {
    pub fn neighbors(&mut self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
        let (_, sources, targets) = self.space.get_columnar_topology(&self.pool);
        self.neighbors_from_topology(node, sources, targets)
    }
    
    // Create immutable versions that delegate to mutable ones
    pub fn neighbors_immutable(&self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
        // Use less efficient edge-by-edge lookup for immutable contexts
        self.neighbors_slow_path(node)
    }
}

// TraversalEngine always gets fresh data
impl TraversalEngine {
    pub fn bfs(&mut self, space: &mut GraphSpace, pool: &GraphPool, ...) -> Result<...> {
        // Always get fresh topology
        let (edge_ids, sources, targets) = space.get_columnar_topology(pool);
        // Rebuild adjacency cache if needed
        self.adjacency_cache.rebuild(edge_ids, sources, targets, space.get_topology_generation());
        // ... rest of BFS
    }
}
```

**Pros**:
- Simple and correct
- No cache coherency issues
- Easy to reason about

**Cons**:
- May rebuild cache more often than needed
- Requires mutable access for many operations

### Option 3: Smart Interior Mutability

**Strategy**: Use `RefCell`/`Mutex` for cache management within immutable APIs

**Implementation**:
```rust
// In GraphSpace - use interior mutability for cache
pub struct GraphSpace {
    // ... other fields
    topology_cache: RefCell<TopologyCache>,
}

struct TopologyCache {
    edge_sources: Vec<NodeId>,
    edge_targets: Vec<NodeId>, 
    active_edge_ids: Vec<EdgeId>,
    is_dirty: bool,
    generation: usize,
}

impl GraphSpace {
    pub fn get_fresh_topology(&self, pool: &GraphPool) -> (&[EdgeId], &[NodeId], &[NodeId]) {
        let mut cache = self.topology_cache.borrow_mut();
        if cache.is_dirty {
            cache.rebuild(pool, &self.active_edges);
        }
        // Return references... (lifetime issue to solve)
    }
}
```

**Pros**:
- Keep immutable APIs
- Automatic cache management
- Thread-safe with `Mutex`

**Cons**:
- Runtime borrow checking overhead
- Complex lifetime management
- Potential for runtime panics

### Option 4: Remove Adjacency Cache (Simplify)

**Strategy**: Remove the adjacency cache, make everything use topology cache

**Implementation**:
```rust
// Remove AdjacencyCache entirely
impl TraversalEngine {
    pub fn bfs(&mut self, space: &mut GraphSpace, pool: &GraphPool, ...) -> Result<...> {
        let (_, sources, targets) = space.get_columnar_topology(pool);
        
        // Build neighbors on-demand from columnar topology
        let mut queue = VecDeque::new();
        // ... BFS logic using columnar topology directly
        
        while let Some(current) = queue.pop_front() {
            // Find neighbors by scanning sources/targets arrays
            for i in 0..sources.len() {
                if sources[i] == current {
                    queue.push_back(targets[i]);
                } else if targets[i] == current {
                    queue.push_back(sources[i]);
                }
            }
        }
    }
}
```

**Pros**:
- Single cache system
- No cache coherency issues
- Simpler architecture

**Cons**:
- O(E) neighbor lookup instead of O(degree)
- Slower for high-degree nodes
- Re-scans topology for each neighbor lookup

## Recommended Architecture: Option 2 (Always-Fresh Strategy)

### Why Option 2?

1. **Correctness First**: Eliminates cache coherency bugs entirely
2. **Performance Acceptable**: Cache rebuilds are fast (O(E) where E = active edges)
3. **Simple Implementation**: Easy to understand and maintain
4. **Clear APIs**: Mutable operations rebuild caches, immutable operations use slow paths

### Implementation Plan

#### Phase 1: Fix Cache Rebuilding (Immediate)

1. **Update GraphSpace**:
   ```rust
   // Make get_columnar_topology always rebuild if dirty
   pub fn get_columnar_topology(&mut self, pool: &GraphPool) -> (&[EdgeId], &[NodeId], &[NodeId]) {
       if self.topology_cache_dirty {
           self.rebuild_topology_cache(pool);
       }
       (&self.active_edge_ids, &self.edge_sources, &self.edge_targets)
   }
   ```

2. **Update Graph API**:
   ```rust
   // Make neighbors/degree mutable to ensure fresh cache
   pub fn neighbors(&mut self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
       let (_, sources, targets) = self.space.get_columnar_topology(&self.pool);
       // ... use fresh topology
   }
   ```

3. **Update TraversalEngine**:
   ```rust
   // Always get fresh topology before algorithms
   pub fn bfs(&mut self, space: &mut GraphSpace, pool: &GraphPool, ...) -> Result<...> {
       let topology_generation = space.get_topology_generation();
       if !self.adjacency_cache.is_up_to_date(topology_generation) {
           let (edge_ids, sources, targets) = space.get_columnar_topology(pool);
           self.adjacency_cache.rebuild(edge_ids, sources, targets, topology_generation);
       }
       // ... rest of BFS with fresh adjacency cache
   }
   ```

#### Phase 2: Update FFI Layer

1. **Make FFI operations mutable where needed**:
   ```rust
   // Operations that need topology should use mutable borrow
   fn degree(&self, py: Python, node: NodeId) -> PyResult<usize> {
       let mut graph = self.graph.borrow_mut(py);  // Mutable for cache rebuild
       graph.inner.degree(node).map_err(graph_error_to_py_err)
   }
   ```

2. **Batch operations to minimize cache rebuilds**:
   ```rust
   // For connected components, get fresh topology once
   fn connected_components(&self, py: Python) -> PyResult<Vec<Vec<NodeId>>> {
       let mut graph = self.graph.borrow_mut(py);
       // Ensure fresh cache once
       let _ = graph.inner.space.get_columnar_topology(&graph.inner.pool);
       // Now run algorithm
       graph.inner.traversal.connected_components(...)
   }
   ```

#### Phase 3: Optimize Performance

1. **Add bulk operations to minimize rebuilds**
2. **Consider adjacency cache optimizations for repeated traversals**
3. **Profile and optimize hot paths**

### Expected Performance Impact

- **Connected Components**: From O(n²) to O(n + E) - massive improvement
- **BFS/DFS**: From wrong results to correct results with minimal overhead
- **Cache Rebuilds**: Predictable O(E) cost, only when topology changes
- **Memory**: Slight reduction (fewer stale cache copies)

### Migration Strategy

1. **Immediate**: Implement Phase 1 for correctness
2. **Test**: Run benchmarks to verify correctness and performance
3. **Optimize**: Implement Phases 2-3 for performance
4. **Validate**: Compare against NetworkX for correctness verification

This approach prioritizes **correctness over performance** initially, then optimizes performance while maintaining correctness.