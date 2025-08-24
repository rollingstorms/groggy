# Connected Components Issue Investigation

## Problem Statement
When creating subgraphs using `g.nodes[g.degree() > 4]` on a Watts-Strogatz graph with parameters (100, 4, 0.1), the connected components analysis returns only 1 component when we expect multiple components.

## Test Cases
```python
# Test 1: Watts-Strogatz graph with high-degree node selection
g = gr.generators.watts_strogatz(100, 4, 0.1)
high_degree_nodes = g.nodes[g.degree() > 4]
print(len(high_degree_nodes.connected_components()))  # Expected: >1, Actual: 1

# Test 2: Simple disconnected graph
g = gr.Graph()
for i in range(6):
    g.add_node()
# Triangle 1: 0-1-2-0
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
# Triangle 2: 3-4-5-3  
g.add_edge(3, 4)
g.add_edge(4, 5)
g.add_edge(5, 3)
subgraph = g.nodes[[0, 1, 2, 3, 4, 5]]
print(len(subgraph.connected_components()))  # Expected: 2, Actual: 1

# Test 3: Isolated nodes
g = gr.Graph()
for i in range(2):
    g.add_node()
subgraph = g.nodes[[0, 1]]
print(len(subgraph.connected_components()))  # Expected: 2, Actual: 1
```

## Root Cause Analysis

### Architecture Overview
- **PySubgraph** (FFI layer): Python wrapper with dual-mode support
  - `inner: Option<RustSubgraph>` - Core Rust subgraph implementation
  - Fallback mode when `inner = None`
  
- **RustSubgraph** (Core): Contains proper BFS connected components algorithm

### Issue Identified
The problem is in the FFI layer fallback path in `PySubgraph::connected_components()`:

```rust
// In src/ffi/core/subgraph.rs line 512-519
} else {
    // Fallback - return single component for now
    Ok(vec![PySubgraph::new(
        self.nodes.clone(),
        self.edges.clone(),
        "component".to_string(),
        self.graph.clone(),
    )])
}
```

**This fallback always returns exactly 1 component regardless of connectivity!**

### When Does Fallback Trigger?
- **PySubgraph::from_core_subgraph()**: Sets `inner: Some(subgraph)` ✅ Works correctly
- **PySubgraph::new()**: Sets `inner: None` ❌ Hits broken fallback

### Subgraph Creation Paths
1. **Via accessors** (`g.nodes[...]`): Uses `PySubgraph::new()` → `inner = None` → broken fallback
2. **Via core operations**: Uses `from_core_subgraph()` → `inner = Some()` → works correctly

## Manual BFS Verification
Manual BFS algorithm on the test case correctly identifies 2 components:
```python
# Adjacency list for two triangles
adj = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
# BFS finds: Component 0: [0, 1, 2], Component 1: [3, 4, 5] ✅
```

## Core Algorithm Verification
The RustSubgraph connected_components algorithm (lines 270-352 in `/src/core/subgraph.rs`) is correct:
- Proper adjacency list building
- Correct BFS traversal  
- Accurate component edge collection
- **Algorithm is not the problem**

## Watts-Strogatz Behavior Analysis
The original user observation about high-degree nodes is actually **correct behavior**:
- High-degree nodes selected based on **full graph** degree
- Subgraph contains only **induced edges** between selected nodes
- Many connections that made nodes "high-degree" are lost (connected to unselected nodes)
- Result: Selected high-degree nodes may have few connections to each other
- **This should create multiple disconnected components**

## Solution Options

### Option 1: Fix Fallback Path (Quick Fix)
Implement proper BFS algorithm in the fallback case.
- **Pros**: Simple, targeted fix
- **Cons**: Duplicates algorithm logic

### Option 2: Always Create RustSubgraph (Architectural Fix)  
Modify accessor methods to always create RustSubgraph with proper `Rc<RefCell<Graph>>` reference.
- **Pros**: Uses single algorithm implementation, cleaner architecture
- **Cons**: More complex, requires understanding Rust ownership patterns

### Option 3: Eliminate Dual-Mode Architecture
Remove the fallback entirely - all PySubgraphs must have `inner`.
- **Pros**: Simplest long-term solution
- **Cons**: Requires larger refactoring

## Current Status
- Connected components algorithm is correct
- Issue is in FFI fallback implementation returning wrong result
- Problem affects all subgraphs created via accessor syntax (`g.nodes[...]`)
- User expectation and behavior analysis are both correct

## Recommendation
**Option 2** is preferred: Modify accessors to create proper RustSubgraph instances, eliminating the need for fallback logic and ensuring consistent behavior across all subgraph creation methods.