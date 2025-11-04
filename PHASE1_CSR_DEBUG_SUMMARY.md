# Phase 1 CSR Optimization - Debug Summary

## What We're Trying To Do

Optimize step primitives to use CSR (Compressed Sparse Row) for neighbor access, following the STYLE_ALGO pattern that made our algorithms 10-50x faster.

**Goal**: Make `subgraph.neighbors()` and `subgraph.degree()` use cached CSR instead of the old `neighbors_filtered()` snapshot approach.

---

## What We Implemented

### Files Changed

1. **`src/subgraphs/subgraph.rs`**

Added 3 methods:

```rust
// Line ~426: Helper to get or build CSR
fn get_or_build_csr_internal(&self, add_reverse: bool) -> GraphResult<Arc<Csr>> {
    // Check cache first
    if let Some(cached) = self.csr_cache_get(add_reverse) {
        return Ok(cached);
    }

    // Cache miss - build CSR
    let nodes = self.ordered_nodes();
    let edges = self.ordered_edges();
    
    // Build indexer: node_id -> CSR index
    let mut node_to_idx: HashMap<NodeId, usize> = HashMap::new();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }
    
    // Build CSR
    let mut csr = Csr::default();
    {
        let graph = self.graph.borrow();
        let pool = graph.pool();
        
        build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |nid| node_to_idx.get(&nid).copied(),
            |eid| pool.get_edge_endpoints(eid),
            CsrOptions {
                add_reverse_edges: add_reverse,
                sort_neighbors: false,
            },
        );
    }
    
    let csr_arc = Arc::new(csr);
    self.csr_cache_store(add_reverse, csr_arc.clone());
    Ok(csr_arc)
}

// Line ~1678: Override neighbors()
fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
    let add_reverse = false;  // ← KEY PARAMETER
    let csr = self.get_or_build_csr_internal(add_reverse)?;
    let ordered_nodes = self.ordered_nodes();
    
    let node_idx = ordered_nodes.binary_search(&node_id)
        .map_err(|_| GraphError::NodeNotFound { ... })?;
    
    // Get neighbors from CSR
    let neighbor_indices = csr.neighbors(node_idx);
    
    // Map back to NodeIds
    Ok(neighbor_indices
        .iter()
        .map(|&idx| ordered_nodes[idx])
        .collect())
}

// Line ~1709: Override degree()
fn degree(&self, node_id: NodeId) -> GraphResult<usize> {
    let add_reverse = false;  // ← KEY PARAMETER
    let csr = self.get_or_build_csr_internal(add_reverse)?;
    let ordered_nodes = self.ordered_nodes();
    
    let node_idx = ordered_nodes.binary_search(&node_id)
        .map_err(|_| GraphError::NodeNotFound { ... })?;
    
    Ok(csr.neighbors(node_idx).len())
}
```

---

## The Problem

**3 tests are failing** because they expect different neighbor/degree counts than what our CSR implementation returns:

### Failing Test 1: `node_degree_step_computes_degrees`

**Location**: `src/algorithms/steps/mod.rs:243`

**Test Code**:
```rust
let mut graph = Graph::new();  // Creates UNDIRECTED graph (default)
let a = graph.add_node();
let b = graph.add_node();
let c = graph.add_node();
graph.add_edge(a, b).unwrap();  // Edge a→b
graph.add_edge(b, c).unwrap();  // Edge b→c

// Expected degrees:
assert_eq!(degree[a], 1);  // a connected to b
assert_eq!(degree[b], 2);  // b connected to a and c  ← FAILS
assert_eq!(degree[c], 1);  // c connected to b
```

**What Happens**:
- Expected: `degree[b] = 2`
- Actual: `degree[b] = 1`

**Why**: Node `b` should see both neighbors `a` and `c`, but our CSR only shows one.

---

### Failing Test 2: `k_core_mark_step_identifies_cores`

**Location**: `src/algorithms/steps/mod.rs:419`

Similar issue - degree counts are wrong.

---

### Failing Test 3: `triangle_count_step_counts_triangles`

**Location**: `src/algorithms/steps/mod.rs:449`

Triangle counting depends on correct neighbor lists.

---

## Root Cause Analysis

### Key Facts:

1. **Graph Type**: `Graph::new()` creates **UNDIRECTED** graph by default
   - `src/types.rs`: `impl Default for GraphType` returns `Self::Undirected`

2. **Old Behavior**: `subgraph.neighbors()` called `graph.neighbors_filtered()`
   - Which called `graph.neighbors()`
   - Which used `space.snapshot()` to get neighbors
   - **This returned bidirectional neighbors for undirected graphs**

3. **New Behavior**: We build CSR with `add_reverse=false`
   - For undirected graphs, this might not be correct

4. **The Question**: When you add edge (a,b) to an undirected graph:
   - Does the storage automatically create both (a,b) and (b,a)?
   - Or does CSR need `add_reverse=true` to make it bidirectional?

---

## CSR Build Parameters

In our algorithms (PageRank, LPA, etc.), we use this pattern:

```rust
let is_directed = subgraph.graph().borrow().is_directed();
let add_reverse = !is_directed;  // TRUE for undirected, FALSE for directed
```

But in our new implementation, we're using:

```rust
let add_reverse = false;  // Always false
```

**Hypothesis**: We should match the algorithm pattern:
- **Undirected graph**: `add_reverse=false` (edges already bidirectional in storage)
- **Directed graph**: `add_reverse=false` (respect direction)

OR:
- **Undirected graph**: `add_reverse=true` (CSR needs to duplicate edges)
- **Directed graph**: `add_reverse=false` (respect direction)

---

## How To Debug

### Option 1: Check Edge Storage

Add debug print to see what edges exist:

```rust
// In get_or_build_csr_internal, before building CSR:
let edges = self.ordered_edges();
eprintln!("Building CSR with {} edges:", edges.len());
for &eid in edges.iter().take(10) {
    if let Some((src, tgt)) = pool.get_edge_endpoints(eid) {
        eprintln!("  Edge {}: {} -> {}", eid, src, tgt);
    }
}
```

**Question**: For undirected graph with `add_edge(a,b)`, do we see:
- Only `a→b`? (need `add_reverse=true`)
- Both `a→b` and `b→a`? (use `add_reverse=false`)

### Option 2: Check CSR Output

Add debug after CSR build:

```rust
// After build_csr_from_edges_with_scratch:
eprintln!("CSR built: {} nodes, {} neighbors total", 
    csr.offsets.len() - 1, csr.neighbors.len());
for i in 0..csr.offsets.len().saturating_sub(1).min(5) {
    let nbrs = csr.neighbors(i);
    eprintln!("  Node idx {}: {} neighbors: {:?}", i, nbrs.len(), nbrs);
}
```

### Option 3: Compare With Old Implementation

Temporarily add old implementation side-by-side:

```rust
fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
    // New CSR way
    let csr_result = self.neighbors_csr(node_id)?;
    
    // Old way
    let old_result = {
        let graph = self.graph.borrow();
        graph.neighbors_filtered(node_id, self.node_set())?
    };
    
    if csr_result != old_result {
        eprintln!("MISMATCH for node {}:", node_id);
        eprintln!("  CSR: {:?}", csr_result);
        eprintln!("  Old: {:?}", old_result);
    }
    
    Ok(csr_result)
}
```

---

## Test To Run

```bash
# Run the failing test with output
cargo test --lib algorithms::steps::tests::node_degree_step_computes_degrees -- --nocapture

# Or all 3 failing tests
cargo test --lib algorithms::steps::tests 2>&1 | grep -A 5 "FAILED\|panicked"
```

---

## Expected Fix

Once we know how edges are stored, the fix is simple:

**If edges are NOT duplicated in storage**:
```rust
fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
    let is_directed = self.graph.borrow().is_directed();
    let add_reverse = !is_directed;  // TRUE for undirected
    let csr = self.get_or_build_csr_internal(add_reverse)?;
    // ...
}
```

**If edges ARE duplicated in storage**:
```rust
// Keep add_reverse=false (current implementation)
// But investigate why CSR isn't seeing both directions
```

---

## Files Modified

- `src/subgraphs/subgraph.rs` - Added CSR-based neighbors/degree overrides
- `notes/planning/BUILDER_CHECKLIST.md` - Updated progress

## Files to Check for Understanding

- `src/api/graph.rs:640` - Old `neighbors_filtered()` implementation
- `src/api/graph.rs:370` - How `add_edge()` works for undirected graphs
- `src/state/topology/csr.rs:75` - How `build_csr_from_edges_with_scratch` works
- `src/algorithms/centrality/pagerank.rs:195` - How PageRank decides `add_reverse`

---

## Summary

**Status**: Implementation complete, but 3 tests failing due to incorrect neighbor counts

**Root Cause**: `add_reverse` parameter is likely wrong for undirected graphs

**Next Step**: Debug to see how edges are stored, then adjust `add_reverse` logic

**Expected**: Once fixed, all step primitives automatically get 10-50x faster neighbor access!
