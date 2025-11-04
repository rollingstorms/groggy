# Algorithm Implementation Style Guide

## Purpose

This guide provides a **mechanical, drop-in template** for implementing graph algorithms in groggy with optimal performance and consistent profiling. Follow this pattern to ensure:
- Zero inner-loop allocations
- CSR-based neighbor access
- Consistent profiling instrumentation
- Cache-aware topology handling

## Template Pattern (Pseudo-Code)

```rust
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

## Reverse Edge Rules

```rust
NEED_REVERSE_EDGES(is_directed, params):
  // Undirected behavior on directed graph → true
  // Directional algos (PR, SCC, SSSP) → false
  // Your call per-algo
```

**Common Cases:**
- **PageRank**: `false` (uses incoming edges naturally)
- **Label Propagation**: `!is_directed` (needs bidirectional for undirected)
- **Connected Components**: `!is_directed` (symmetric reachability)
- **BFS/DFS**: `false` (directional by nature)
- **Betweenness**: `!is_directed` (symmetric paths)

## Compute Kernel Patterns

### Pattern 1: One-Pass Linear (No Iterations)

**Use for**: Degree centrality, local clustering, triangle counting

```rust
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

### Pattern 2: Iterative Solver (Convergence-Based)

**Use for**: PageRank, eigenvector centrality, belief propagation

```rust
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  // Pre-allocate all buffers
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

      swap(x, x_new)
      CALL_TIME(pfx+".compute.iter", now()-IT_T)
      STATS(pfx+".count.iteration", it+1)
      
      if converged: break

  // Pack results
  OUT = Vec<(NodeId, AttrValue)>(CSR.node_count())
  for u in 0..CSR.node_count():
      OUT.push( (NODES[u], AttrValue::Float(x[u])) )
  
  return OUT
```

### Pattern 3: Multi-Output (Components/Paths)

**Use for**: Connected components, community detection, shortest paths

```rust
COMPUTE_KERNEL(ctx, CSR, NODES, INDEXER, params):
  // Produce custom structs; still no inner-loop allocs
  // BFS/DFS/Union-Find with pre-allocated workspace
  
  components = detect_components(CSR, NODES, params)
  
  // At emit: use ctx.add_output for structured data
  ctx.add_output(
    ALGO_ID+".components", 
    AlgorithmOutput::Components(components)
  )
```

## Edge Attribute Variant

For algorithms that compute edge attributes (e.g., edge betweenness):

```rust
RESULTS: Vec<(EdgeId, AttrValue)>

// Emit phase:
if ctx.persist_results():
    timed(pfx+".write_attributes"): 
        subgraph.set_edge_attr_column(params.output_attr, RESULTS)
else:
    timed(pfx+".store_output"): 
        ctx.add_output(ALGO_ID+".result", EdgeAttributes(RESULTS))
```

## Profiling Keys (Keep Consistent)

### Call Timers
- `pfx.collect_nodes` - Gathering ordered node list
- `pfx.build_indexer` - Building NodeId→index map
- `pfx.collect_edges` - Gathering ordered edge list
- `pfx.csr_cache_hit` - CSR found in cache
- `pfx.csr_cache_miss` - CSR needs building
- `pfx.collect_edge_endpoints` - Getting edge source/target pairs
- `pfx.build_csr` - Building CSR structure
- `pfx.compute` - Core algorithm computation
- `pfx.compute.iter` - Per-iteration time (for iterative algos)
- `pfx.write_attributes` - Writing results to graph
- `pfx.store_output` - Storing results in context
- `pfx.total_execution` - Full algorithm execution

### Counter Stats
- `pfx.count.input_nodes` - Input node count
- `pfx.count.input_edges` - Input edge count
- `pfx.count.csr_nodes` - CSR node count
- `pfx.count.csr_edges` - CSR edge count (includes reverse if added)
- `pfx.count.iteration` - Iteration number (iterative algos)
- `pfx.count.converged_nodes` - Nodes that converged (if tracked)

## Minimal Agent Checklist

Apply these transformations to each algorithm:

1. ✅ **Replace ad-hoc neighbor access** with `CSR.neighbors(u)`
2. ✅ **Use ordered collections**: `ordered_nodes()` / `ordered_edges()`
3. ✅ **Add NodeIndexer**: `NodeIndexer::new(nodes)` and pass `INDEXER.get(nid)` to CSR builder
4. ✅ **Decide ADD_REV** per algo and use CSR cache keyed by it
5. ✅ **Move allocations out of inner loops**; reuse buffers
6. ✅ **Emit via set_*_attr_column** or `ctx.add_output`
7. ✅ **Add consistent profiling** keys per above naming scheme

## Example: Before/After

### Before (Old Style)
```rust
pub fn run(&mut self, ctx: &mut AlgorithmContext, subgraph: &Subgraph) -> GraphResult<Subgraph> {
    let graph = subgraph.graph();
    let pool = graph.borrow().pool();
    
    // Collect nodes (no profiling)
    let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
    let n = nodes.len();
    
    // Build adjacency (no caching)
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for node in &nodes {
        let neighbors: Vec<usize> = graph.borrow()
            .neighbors(*node)  // SLOW: repeated graph borrows
            .iter()
            .filter_map(|&n| nodes.iter().position(|&id| id == n))
            .collect();  // ALLOCATION IN LOOP
        adj.insert(nodes.iter().position(|&id| id == *node).unwrap(), neighbors);
    }
    
    // Compute (with inner allocations)
    let mut scores = vec![0.0; n];
    for i in 0..n {
        if let Some(nbrs) = adj.get(&i) {  // Hash lookup every iteration
            scores[i] = nbrs.len() as f64;  // Inner vec allocation
        }
    }
    
    // Emit (no profiling)
    let results: Vec<_> = nodes.iter()
        .enumerate()
        .map(|(i, &nid)| (nid, AttrValue::Float(scores[i])))
        .collect();
    
    subgraph.set_node_attr_column(self.params.output_attr.clone(), results)?;
    Ok(subgraph.clone())
}
```

### After (New Style)
```rust
pub fn run(&mut self, ctx: &mut AlgorithmContext, subgraph: &Subgraph) -> GraphResult<Subgraph> {
    let start_time = Instant::now();
    
    // Collect nodes with profiling
    let collect_start = Instant::now();
    let nodes = subgraph.ordered_nodes();
    ctx.record_call_time("degree.collect_nodes", collect_start.elapsed());
    ctx.record_stat("degree.count.input_nodes", nodes.len() as f64);
    
    // Build indexer
    let idx_start = Instant::now();
    let indexer = NodeIndexer::new(&nodes);
    ctx.record_call_time("degree.build_indexer", idx_start.elapsed());
    
    // Collect edges
    let edges_start = Instant::now();
    let edges = subgraph.ordered_edges();
    ctx.record_call_time("degree.collect_edges", edges_start.elapsed());
    
    // Get or build CSR (with caching)
    let add_reverse = !subgraph.graph().borrow().is_directed();
    let csr = match subgraph.csr_cache_get(add_reverse) {
        Some(cached) => {
            ctx.record_call("degree.csr_cache_hit");
            cached
        },
        None => {
            ctx.record_call("degree.csr_cache_miss");
            let (csr, endpoint_time, build_time) = build_csr_from_edges_with_scratch(
                nodes.len(),
                &edges,
                |nid| indexer.get(nid),
                |eid| pool.borrow().get_edge_endpoints(eid).ok(),
                CsrBuildOptions {
                    add_reverse_edges: add_reverse,
                    sort_neighbors: false,
                }
            );
            ctx.record_call_time("degree.collect_edge_endpoints", endpoint_time);
            ctx.record_call_time("degree.build_csr", build_time);
            subgraph.csr_cache_store(add_reverse, csr.clone());
            csr
        }
    };
    
    ctx.record_stat("degree.count.csr_nodes", csr.node_count() as f64);
    ctx.record_stat("degree.count.csr_edges", csr.neighbors_len() as f64);
    
    // Compute (zero allocations in loop)
    let compute_start = Instant::now();
    let mut results = Vec::with_capacity(nodes.len());
    for u_idx in 0..csr.node_count() {
        let degree = csr.neighbors(u_idx).len() as f64;  // Slice; no alloc
        results.push((nodes[u_idx], AttrValue::Float(degree)));
    }
    ctx.record_call_time("degree.compute", compute_start.elapsed());
    
    // Emit
    if ctx.persist_results() {
        let write_start = Instant::now();
        subgraph.set_node_attr_column(self.params.output_attr.clone(), results)?;
        ctx.record_call_time("degree.write_attributes", write_start.elapsed());
    } else {
        let store_start = Instant::now();
        ctx.add_output("degree.result", AlgorithmOutput::NodeAttributes(results));
        ctx.record_call_time("degree.store_output", store_start.elapsed());
    }
    
    ctx.record_call_time("degree.total_execution", start_time.elapsed());
    Ok(subgraph.clone())
}
```

## Performance Benefits

Following this pattern ensures:
- **CSR caching**: Topology built once, reused across algorithms
- **Zero inner-loop allocations**: All buffers pre-allocated
- **Cache-friendly access**: CSR uses contiguous memory for neighbors
- **Consistent profiling**: Easy to identify bottlenecks
- **Optimal memory layout**: Columnar results for bulk operations

## Common Pitfalls to Avoid

❌ **Don't**: Repeatedly borrow graph in loops
```rust
for node in nodes {
    let nbrs = graph.borrow().neighbors(node);  // SLOW
}
```

✅ **Do**: Build CSR once, use indexed access
```rust
let csr = build_csr(...);
for u_idx in 0..csr.node_count() {
    let nbrs = csr.neighbors(u_idx);  // FAST
}
```

❌ **Don't**: Allocate inside loops
```rust
for u in 0..n {
    let mut vec = Vec::new();  // SLOW
    for v in neighbors { vec.push(v); }
}
```

✅ **Do**: Pre-allocate and reuse buffers
```rust
let mut buffer = Vec::with_capacity(max_degree);
for u in 0..n {
    buffer.clear();
    for v in neighbors { buffer.push(v); }
}
```

❌ **Don't**: Use HashMap for dense integer indices
```rust
let mut map: HashMap<usize, f64> = HashMap::new();
```

✅ **Do**: Use Vec for dense indices
```rust
let mut vec: Vec<f64> = vec![0.0; n];
```

## Testing Checklist

After implementing an algorithm with this pattern:

1. ✅ Run on small graph (10 nodes) - verify correctness
2. ✅ Run on medium graph (10k nodes) - check CSR cache hit on second run
3. ✅ Run on large graph (200k nodes) - verify no performance regression
4. ✅ Check profiling output - all expected timers present
5. ✅ Verify memory usage - no unexpected allocations
6. ✅ Test both directed and undirected graphs
7. ✅ Test with `persist=True` and `persist=False`

## Quick Reference: Common Algorithms

| Algorithm | Pattern | ADD_REV | Special Notes |
|-----------|---------|---------|---------------|
| PageRank | Iterative | `false` | Needs incoming edges |
| Label Propagation | Iterative | `!is_directed` | Symmetric updates |
| Connected Components | Multi-output | `!is_directed` | Union-Find or BFS |
| Degree Centrality | One-pass | `!is_directed` | Just count neighbors |
| Betweenness | Multi-pass | `!is_directed` | Multiple BFS passes |
| Closeness | Iterative/BFS | `!is_directed` | Shortest paths |
| Eigenvector | Iterative | `false` | Power iteration |

## Maintenance

When this pattern evolves:
1. Update this document first
2. Update 2-3 reference algorithms
3. Create migration checklist for remaining algorithms
4. Update tests to verify pattern compliance
