# Subgraph Architecture & Ownership Analysis

## Current Subgraph Structure

### Core Components
```rust
#[derive(Debug, Clone)]
pub struct Subgraph {
    /// Reference to the parent graph (shared across all subgraphs)
    /// Uses RefCell for interior mutability to allow batch operations
    graph: Rc<RefCell<Graph>>,

    /// Set of node IDs that are included in this subgraph
    nodes: HashSet<NodeId>,

    /// Set of edge IDs that are included in this subgraph  
    edges: HashSet<EdgeId>,

    /// Type identifier for this subgraph (e.g. "filtered_nodes", "neighborhood")
    subgraph_type: String,
}
```

### Python FFI Layer
```rust
#[pyclass(name = "Subgraph", unsendable)]
pub struct PySubgraph {
    /// Core Rust subgraph (when created properly)
    inner: Option<RustSubgraph>,
    
    /// Direct storage (when created via FFI accessors) 
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    subgraph_type: String,
    
    /// Reference to Python graph object
    graph: Option<Py<PyGraph>>,
}
```

## The Dual-Mode Problem

### Mode 1: Proper Subgraphs (`inner: Some(RustSubgraph)`)
- **Created by**: Core graph operations, proper subgraph factories
- **Has access to**: Full graph reference `Rc<RefCell<Graph>>`
- **Connected components**: ✅ **Works correctly** - uses proper BFS algorithm
- **Data access**: Can call `graph.borrow()` to access full Graph API

### Mode 2: Accessor Subgraphs (`inner: None`)
- **Created by**: FFI accessor methods (`g.nodes[...]`, `g.all()`)
- **Has access to**: Python graph reference `Py<PyGraph>` 
- **Connected components**: ❌ **Broken** - hits fallback that returns single component
- **Data access**: Must call `graph_py.borrow(py)` with Python GIL

## Ownership & Access Patterns

### What Subgraph Needs for Connected Components:

1. **Node set**: ✅ Available in both modes (`self.nodes`)
2. **Edge set**: ✅ Available in both modes (`self.edges`) 
3. **Graph reference**: ⚠️ **Different types in each mode**
4. **Edge endpoint lookup**: ⚠️ **Different access patterns**

### Current Graph Access Patterns:

**Mode 1 (RustSubgraph):**
```rust
let graph = self.graph.borrow();  // Returns Graph
graph.edge_endpoints(edge_id)     // Direct method call
```

**Mode 2 (PySubgraph fallback):**
```rust
let py = Python::acquire_gil().python();
let graph = graph_py.borrow(py);  // Returns PyGraph  
graph.inner.edge_endpoints(edge_id)  // Access through .inner field
```

## Why TraversalEngine Integration Is Complex

### Interface Mismatch:
```rust
// TraversalEngine expects:
fn connected_components(
    &mut self,
    pool: &GraphPool,        // ← Need this
    space: &GraphSpace,      // ← Need this  
    options: TraversalOptions,
) -> GraphResult<ConnectedComponentsResult>

// Subgraph has:
graph: Rc<RefCell<Graph>>    // Contains pool & space, but not directly accessible
```

### Borrow Checker Issues:
```rust
let mut graph = self.graph.borrow_mut();
let pool = &graph.pool.borrow();     // ← Immutable borrow
let space = &graph.space;            // ← Immutable borrow  
let traversal = &mut graph.traversal_engine;  // ← Mutable borrow!
// ❌ Cannot have both mutable and immutable borrows
```

### Graph Field Accessibility:
```rust
pub struct Graph {
    pool: Rc<RefCell<GraphPool>>,     // ← Private field
    space: GraphSpace,                // ← Private field
    traversal_engine: TraversalEngine, // ← Private field
    // ...
}
```
- **Problem**: Fields are private, no direct access from Subgraph
- **Need**: Public accessor methods or friend-like access

## Potential Solutions

### Option 1: Fix the FFI Fallback (Simple)
- **Approach**: Implement proper connected components in the fallback case
- **Pros**: Minimal changes, preserves architecture
- **Cons**: Duplicates algorithm (but simpler version is fine)

### Option 2: Unify on RustSubgraph (Medium)  
- **Approach**: Make all subgraph creation go through RustSubgraph constructor
- **Pros**: Single code path, uses optimized algorithm
- **Cons**: Need to solve Graph → Rc<RefCell<Graph>> conversion

### Option 3: Add Graph Accessors (Medium)
- **Approach**: Add public methods to Graph for accessing internal components
- **Pros**: Enables TraversalEngine usage from Subgraph
- **Cons**: Exposes internal architecture, complex borrow patterns

### Option 4: Composition Over Delegation (Complex)
- **Approach**: Give Subgraph its own TraversalEngine instance  
- **Pros**: No borrowing issues, clean separation
- **Cons**: Memory overhead, duplication of engines

## Recommended Approach: Fix the FFI Fallback

### Why This Is The Right Solution:

1. **Preserves Architecture**: Doesn't break existing patterns
2. **Simple Implementation**: Just implement BFS in the fallback
3. **Performance**: Fallback only used for accessor-created subgraphs (typically small)
4. **Maintainable**: Clear separation of concerns

### Implementation Plan:

```rust
// In PySubgraph::connected_components fallback case:
} else {
    // Implement proper BFS algorithm here
    // Use self.graph_py.borrow(py).inner for edge lookups
    // Same algorithm as RustSubgraph, different data access pattern
}
```

## Root Cause of Current Problem

The issue isn't that "subgraph can't implement connected components" - **it already does correctly!** 

The problem is the **FFI layer's dual-mode architecture** where:
- ✅ `inner: Some(RustSubgraph)` → Works correctly
- ❌ `inner: None` → Hits broken fallback that always returns 1 component

The core RustSubgraph connected_components algorithm is **perfect** - it's just the Python wrapper's fallback that's broken.

## Conclusion

The ownership patterns are actually **well-designed**. The issue is simply that the FFI fallback case returns a fake single component instead of running the same BFS algorithm with different data access patterns.

**Fix**: Implement the same BFS algorithm in the FFI fallback, just with `graph_py.borrow(py).inner` access pattern instead of `graph.borrow()` access pattern.