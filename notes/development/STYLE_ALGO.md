# STYLE_ALGO – Canonical Algorithm Pattern

**Purpose**: Define the standard high-performance pattern for all Groggy algorithms  
**Audience**: Algorithm implementers  
**Status**: Active (applies to all new algorithms and refactoring work)

---

## Overview

STYLE_ALGO establishes a consistent, cache-friendly, instrumented pattern for graph algorithms that scales efficiently to 200K+ node graphs. This pattern was developed through iterative optimization of core algorithms (Connected Components, PageRank, LPA, Louvain, Betweenness) and distills the key practices that deliver sub-100ms execution times.

---

## Pseudo-Code Template

```
ALGO <ALGO_ID>:

PARAMS:
  - output_attr: Text (default "<algo>_score")
  - ... (algo-specific: tol, max_iters, damping, seed, etc.)

PHASE PREFIX:
  pfx = "<algo>"

EXECUTE(ctx, subgraph):
  T0 = now()
  NODES = timed(pfx+".collect_nodes"): subgraph.ordered_nodes()
  STATS(pfx+".count.input_nodes", len(NODES))

  // Indexer
  INDEXER = timed(pfx+".build_indexer"): NodeIndexer::new(NODES)

  // Edges + directedness
  EDGES   = timed(pfx+".collect_edges"): subgraph.ordered_edges()
  IS_DIR  = subgraph.graph().borrow().is_directed()
  ADD_REV = NEED_REVERSE_EDGES(IS_DIR, params)  // per-algo rule

  // CSR (cache-aware)
  if subgraph.csr_cache_get(ADD_REV) -> CSR:
      CALL(pfx+".csr_cache_hit")
  else:
      CALL(pfx+".csr_cache_miss")
      (CSR, endpoint_time, core_time) =
        build_csr_from_edges_with_scratch(
          nodes_count = len(NODES),
          edges_iter  = EDGES,
          node_map    = (nid) -> INDEXER.get(nid),       // Option<usize>
          endpoints   = (eid) -> pool.get_edge_endpoints(eid),
          options     = { add_reverse_edges: ADD_REV, sort_neighbors: false }
        )
      CALL_TIME(pfx+".collect_edge_endpoints", endpoint_time)
      CALL_TIME(pfx+".build_csr", core_time)
      subgraph.csr_cache_store(ADD_REV, CSR)

  STATS(pfx+".count.csr_nodes",  CSR.node_count())
  STATS(pfx+".count.csr_edges",  CSR.neighbors_len())

  // CORE KERNEL
  RESULTS = COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params)  // no inner allocs

  // EMIT
  if ctx.persist_results():
      timed(pfx+".write_attributes"): subgraph.set_node_attr_column(params.output_attr, RESULTS)
  else:
      timed(pfx+".store_output"): ctx.add_output(ALGO_ID+".result", NodeAttributes(RESULTS))

  CALL_TIME(pfx+".total_execution", now()-T0)
  if env("GROGGY_PROFILE_*"): ctx.print_profiling_report("<Algo Display Name>")

  return subgraph
```

---

## Decision Rules

### NEED_REVERSE_EDGES

Determines whether CSR should include reverse edges for bidirectional access:

```
NEED_REVERSE_EDGES(is_directed, params):
  // Undirected graph → edges already bidirectional
  if !is_directed: return false
  
  // Algorithm-specific rules for directed graphs:
  
  // Treat as undirected (e.g., Connected Components, LPA)
  if ALGO_TREATS_AS_UNDIRECTED: return true
  
  // Directional algorithms (PageRank, SCC, SSSP)
  if ALGO_IS_DIRECTIONAL: return false
  
  // Default: consult algorithm documentation
  return false
```

**Examples**:
- **Connected Components** on directed graph → `true` (treats as undirected)
- **PageRank** on directed graph → `false` (uses directional flow)
- **LPA** on directed graph → `true` (propagates in both directions)
- **Betweenness** on directed graph → `false` (respects edge direction)

---

## COMPUTE_KERNEL Variants

### Variant 1: One-Pass (Linear, No Iterations)

Used for: Connected Components, Degree Centrality, Triangle Counting

```
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  T = now()
  OUT = Vec<(NodeId, AttrValue)>(capacity = CSR.node_count())
  
  for u_idx in 0..CSR.node_count():
      nbrs = CSR.neighbors(u_idx)         // slice; no alloc
      
      // --- per-algo math here (use nbrs, maybe node attributes) ---
      score = F(nbrs, u_idx, params)      // keep tight; avoid branching
      
      OUT.push( (NODES[u_idx], AttrValue::Float(score)) )
  
  CALL_TIME(pfx+".compute", now()-T)
  STATS(pfx+".count.nodes", CSR.node_count())
  return OUT
```

**Key Pattern**: Direct pass over nodes, compute score from neighbors, emit.

### Variant 2: Iterative Solver

Used for: PageRank, Eigenvector Centrality, Closeness (iterative)

```
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  // Pre-allocate buffers (OUTSIDE iteration loop)
  alloc x, x_new, scratch with capacity CSR.node_count()
  init x[:] = init_value

  for it in 0..max_iters:
      IT_T = now()
      converged = true
      
      for u in 0..CSR.node_count():
          nbrs = CSR.neighbors(u)
          
          // --- per-algo relax/update ---
          x_new[u] = G(u, nbrs, x, params)

          if |x_new[u]-x[u]| > tol: converged = false

      swap(x, x_new)  // ✅ O(1) pointer swap, no allocation
      
      CALL_TIME(pfx+".compute.iter", now()-IT_T)
      STATS(pfx+".count.iteration", it+1)
      
      if converged: break

  // Pack results
  OUT = Vec<(NodeId, AttrValue)>(CSR.node_count())
  for u in 0..CSR.node_count():
      OUT.push( (NODES[u], AttrValue::Float(x[u])) )
  
  return OUT
```

**Key Pattern**: Pre-allocate two buffers, swap each iteration, check convergence.

### Variant 3: Multi-Output (Components, Paths)

Used for: Community detection with per-community metadata, shortest path trees

```
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  // Produce custom structs; still no inner-loop allocs
  
  // Example: community detection
  labels = compute_labels(CSR, NODES)
  sizes = compute_community_sizes(labels)
  modularity = compute_modularity(CSR, labels)
  
  // Emit multiple outputs
  ctx.add_output(ALGO_ID+".labels", NodeAttributes(labels))
  ctx.add_output(ALGO_ID+".sizes", CommunityData(sizes))
  ctx.add_output(ALGO_ID+".modularity", Scalar(modularity))
  
  return labels  // or primary output
```

**Key Pattern**: Multiple related outputs, structured data, still cache-friendly.

### Variant 4: Edge Attributes (Betweenness on Edges)

Used for: Edge betweenness, edge weights, flow algorithms

```
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  // Similar to Variant 1 but produces edge results
  
  RESULTS = Vec<(EdgeId, AttrValue)>()
  
  // ... compute edge scores using CSR ...
  
  return RESULTS
```

**Emit**:
```
timed(pfx+".write_attributes"): 
    subgraph.set_edge_attr_column(params.output_attr, RESULTS)
```

---

## Profiling Keys (Consistent Naming)

All algorithms use these standard keys:

### Timers (record_call_time)
- `{pfx}.collect_nodes` – Collecting node list
- `{pfx}.build_indexer` – Building NodeId → usize map
- `{pfx}.collect_edges` – Collecting edge list
- `{pfx}.collect_edge_endpoints` – Looking up edge endpoints during CSR build
- `{pfx}.build_csr` – Core CSR construction
- `{pfx}.compute` – Main algorithm computation (one-pass)
- `{pfx}.compute.iter` – Single iteration (iterative algorithms)
- `{pfx}.write_attributes` – Writing results to subgraph
- `{pfx}.store_output` – Storing results in context
- `{pfx}.total_execution` – End-to-end time

### Calls (record_call)
- `{pfx}.csr_cache_hit` – CSR found in cache
- `{pfx}.csr_cache_miss` – CSR not found, building

### Stats (record_stat)
- `{pfx}.count.input_nodes` – Number of nodes in input
- `{pfx}.count.input_edges` – Number of edges in input
- `{pfx}.count.csr_nodes` – Nodes in CSR (should match input)
- `{pfx}.count.csr_edges` – Edges in CSR (may differ if reverse edges added)
- `{pfx}.count.iteration` – Iteration count (iterative algorithms)
- `{pfx}.count.components` – Component count (community detection)
- Algorithm-specific: `{pfx}.count.triangles`, `{pfx}.count.relaxations`, etc.

**Prefix Conventions**:
- `cc` – Connected Components
- `pr` – PageRank
- `lpa` – Label Propagation
- `louvain` – Louvain
- `bc` – Betweenness Centrality
- `dijkstra`, `bfs`, `dfs`, `astar`, etc.

---

## Minimal Agent Checklist (Apply to Each Algo)

When implementing a new algorithm or refactoring an existing one:

1. ✅ Replace ad-hoc neighbor access with `CSR.neighbors(u)` slices
2. ✅ Use `ordered_nodes()` and `ordered_edges()` for determinism
3. ✅ Add `NodeIndexer::new(nodes)` and pass `INDEXER.get(nid)` into CSR builder
4. ✅ Decide `ADD_REV` per algorithm (see NEED_REVERSE_EDGES rules)
5. ✅ Check CSR cache before building; store after building
6. ✅ Move all allocations outside inner loops; reuse buffers
7. ✅ Add consistent profiling keys (timers, calls, stats) as listed above
8. ✅ Emit results via `set_*_attr_column` or `ctx.add_output`
9. ✅ Wrap entire execute in `total_execution` timer
10. ✅ Validate: run tests, benchmark, check profiling report coverage

---

## Performance Budget Guidelines (200K Nodes, 600K Edges)

| Algorithm Category | Target | Rationale |
|--------------------|--------|-----------|
| Simple traversal (BFS, DFS, Connected Components) | <50ms | O(n+m) single pass |
| Degree/structural (Degree, Triangle Count) | <100ms | O(m) or O(m·deg) with small constants |
| Iterative solvers (PageRank, LPA) | <300ms | O(k·m) where k ≈ 10 iterations |
| All-pairs (Betweenness, Closeness) | <1s | O(n·m) or O(n²·log n) inherently expensive |
| Multi-phase (Louvain, Leiden) | <200ms | O(m) per phase, ~3-5 phases |
| Expensive (Girvan-Newman, Infomap) | <5s | O(m²·n) or complex computation, best-effort |

If your algorithm exceeds budget by >20%, investigate before merging.

---

## Anti-Patterns to Avoid

### ❌ Ad-Hoc Neighbor Access
```rust
for node in nodes {
    for edge_id in subgraph.incident_edges(node) {
        let (src, tgt) = pool.get_edge_endpoints(edge_id);
        // ... process neighbor ...
    }
}
```

### ❌ Inner-Loop Allocations
```rust
for iter in 0..max_iters {
    let mut new_labels = vec![0; n];  // ← heap allocation every iteration
    // ...
}
```

### ❌ Multiple Graph Traversals
```rust
let edges = subgraph.ordered_edges();  // traversal 1
// ... later ...
let edges2 = subgraph.ordered_edges();  // traversal 2 (redundant)
```

### ❌ Missing Profiling
```rust
fn execute(...) {
    let result = do_computation(subgraph);
    subgraph.set_node_attr_column("result", result);
    // No timers, no visibility
}
```

### ✅ Optimal Patterns

See **PERFORMANCE_TUNING_GUIDE.md** for detailed examples of optimal patterns.

---

## Reference Implementations

Study these for canonical examples:

- **`src/algorithms/community/components.rs`** – Simplest: union-find with CSR (Variant 1)
- **`src/algorithms/centrality/pagerank.rs`** – Iterative solver with buffer swap (Variant 2)
- **`src/algorithms/community/lpa.rs`** – HashMap reuse, convergence check (Variant 2)
- **`src/algorithms/community/louvain.rs`** – Multi-phase, modularity caching (Variant 3)
- **`src/algorithms/centrality/betweenness.rs`** – All-pairs, traversal state reuse (Variant 2)

---

## Testing & Validation

After implementing:

1. **Unit Tests**: Edge cases (empty graph, single node, disconnected)
2. **Integration Tests**: Realistic scenarios on small graphs
3. **Benchmark**: Run `benchmark_algorithms_comparison.py` @ 200K nodes
4. **Profiling**: Check report with `GROGGY_PROFILE_{ALGO}=1`
5. **Determinism**: Multiple runs produce identical results
6. **Performance**: Meets or beats budget (see table above)

---

## Related Documentation

- **PERFORMANCE_TUNING_GUIDE.md** – Detailed optimization patterns and debugging
- **REFACTOR_PLAN_PERFORMANCE_STYLE.md** – Refactoring roadmap and case studies
- **ALGORITHM_REFACTORING_SUMMARY.md** – Current state and priorities
- **notes/planning/advanced-algorithms/** – Phase-by-phase algorithm plans

---

**Last Updated**: 2024  
**Maintainer**: Performance & Architecture Team  
**Status**: Active (canonical pattern for all algorithms)
