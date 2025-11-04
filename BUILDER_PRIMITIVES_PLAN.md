# Builder Primitives Plan

**Status**: Planning  
**Created**: 2025-11-02  
**Goal**: Add missing step primitives to support composable PageRank and LPA implementations

## Problem Statement

The current builder relies heavily on `map_nodes` with expression strings for complex operations. This approach has several issues:

1. **Expression complexity** - The expression parser has to understand every pattern (division, masks, conditionals)
2. **Debugging difficulty** - No visibility into intermediate values
3. **Performance** - Expression evaluation is not optimized like dedicated primitives
4. **Limited coverage** - Missing primitives for common operations (reciprocal, comparisons, reductions)

Currently:
- **PageRank** in builder differs significantly from native because it can't divide by out-degree or handle sinks properly
- **LPA** works but only because `neighbor_values` includes self-value (which breaks PageRank)

## Solution Approach

Add focused, well-scoped primitives for:
- Element-wise arithmetic and logic
- Scalar operations (reduce & broadcast)
- Weighted neighbor aggregation
- Specialized community detection

This keeps `map_nodes` available for ad-hoc logic (prototypes, simple aggregations) while giving real algorithms explicit, testable, performant steps.

---

## Phase 1: Core Arithmetic & Logic Primitives

These are essential for implementing PageRank correctly.

### 1.1 Element-wise Reciprocal

**Rust Primitive**: `core.recip`

```rust
pub struct RecipStep {
    source: String,
    target: String,
    epsilon: f64,
}

// Computes: target[i] = 1.0 / (source[i] + epsilon)
// Safe handling of zero/near-zero values
```

**Python API**:
```python
inv_degrees = builder.core.recip(degrees, epsilon=1e-10)
```

**Use case**: Computing `1/out_degree` for PageRank contributions

---

### 1.2 Element-wise Comparison

**Rust Primitive**: `core.compare`

```rust
pub enum CompareOp {
    Eq, Ne, Lt, Le, Gt, Ge
}

pub struct CompareStep {
    left: String,
    op: CompareOp,
    right: Either<String, f64>,  // variable name or scalar
    target: String,
}

// Produces 0.0/1.0 mask for boolean results
```

**Python API**:
```python
is_sink = builder.core.compare(degrees, "eq", 0.0)
is_high = builder.core.compare(ranks, "gt", threshold_var)
```

**Use case**: Identifying sinks (degree==0), creating masks for conditional operations

---

### 1.3 Element-wise Conditional Selection

**Rust Primitive**: `core.where`

```rust
pub struct WhereStep {
    condition: String,         // 0/1 mask
    if_true: Either<String, f64>,
    if_false: Either<String, f64>,
    target: String,
}

// Computes: target[i] = condition[i] != 0 ? if_true[i] : if_false[i]
```

**Python API**:
```python
sink_ranks = builder.core.where(is_sink, ranks, 0.0)
clamped = builder.core.where(mask, new_values, old_values)
```

**Use case**: Selecting values based on masks (e.g., extracting sink ranks)

---

### 1.4 Scalar Reduction

**Rust Primitive**: `core.reduce_scalar`

```rust
pub enum ReductionOp {
    Sum, Mean, Min, Max
}

pub struct ReduceScalarStep {
    source: String,
    op: ReductionOp,
    target: String,  // Creates a "scalar variable"
}

// Reduces entire node map to single value
// Stores in variables as single-element map or special scalar type
```

**Python API**:
```python
sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
total_rank = builder.core.reduce_scalar(ranks, op="sum")
max_rank = builder.core.reduce_scalar(ranks, op="max")
```

**Use case**: Computing sink mass, checking normalization, convergence checks

---

### 1.5 Scalar Broadcast

**Rust Primitive**: `core.broadcast_scalar`

```rust
pub struct BroadcastScalarStep {
    source: String,  // Scalar variable from reduce_scalar
    target: String,  // Node map with scalar value at every node
}

// Expands scalar to all nodes: target[i] = scalar_value for all i
```

**Python API**:
```python
sink_per_node = builder.core.broadcast_scalar(sink_mass)
uniform_value = builder.core.broadcast_scalar(total)
```

**Use case**: Distributing sink mass across all nodes, adding constants

---

### 1.6 Weighted Neighbor Aggregation

**Rust Primitive**: `core.neighbor_agg`

```rust
pub enum AggOp {
    Sum, Mean, Min, Max, Count
}

pub struct NeighborAggStep {
    source: String,              // Values to aggregate
    target: String,
    agg: AggOp,
    weights: Option<String>,     // Optional per-node weights
    normalize: bool,             // Divide by sum of weights
}

// For each node i: target[i] = agg(source[j] * weight[j] for j in neighbors(i))
// Without weights: target[i] = agg(source[j] for j in neighbors(i))
```

**Python API**:
```python
# Unweighted sum (current behavior)
neighbor_sum = builder.neighbor_agg(ranks, agg="sum")

# Weighted sum (PageRank contribution)
contrib = builder.core.mul(ranks, inv_degrees)
neighbor_sum = builder.neighbor_agg(contrib, agg="sum")

# Alternative: pass weights directly
neighbor_sum = builder.neighbor_agg(ranks, agg="sum", weights=inv_degrees)

# Normalized weighted sum
avg = builder.neighbor_agg(values, agg="sum", weights=edge_weights, normalize=True)
```

**Use case**: PageRank neighbor contributions, weighted graph operations

---

## Phase 2: Decomposed LPA Primitives

Instead of a monolithic `majority_label_update`, we can break it into composable pieces:

### 2.1 Neighbor Value Collection

**Rust Primitive**: `core.collect_neighbor_values`

```rust
pub struct CollectNeighborValuesStep {
    source: String,
    target: String,
    include_self: bool,      // Prepend own value to list
    aggregation: Option<String>,  // If None, return list; else aggregate
}

// For each node: target[i] = [source[j] for j in neighbors(i)]
// Produces a map of NodeId -> Vec<Value>
```

**Python API**:
```python
# Collect neighbor labels into lists
neighbor_labels = builder.core.collect_neighbor_values(
    labels, 
    include_self=True
)
```

**Use case**: Collecting values before aggregation (mode, histogram, custom logic)

---

### 2.2 Mode/Majority Vote

**Rust Primitive**: `core.mode`

```rust
pub struct ModeStep {
    source: String,      // Map of lists (from collect_neighbor_values)
    target: String,
    tie_break: TieBreak, // Lowest, Highest, Random, Keep
}

// For each node: find most frequent value in the list
// Returns single value per node
```

**Python API**:
```python
new_labels = builder.core.mode(
    neighbor_labels,
    tie_break="lowest"
)
```

**Use case**: Computing majority vote, finding most common value

---

### 2.3 In-place Update (Async)

**Rust Primitive**: `core.update_in_place`

```rust
pub struct UpdateInPlaceStep {
    source: String,      // New values
    target: String,      // Map to update (will be mutated)
    ordered: bool,       // Process nodes in deterministic order
}

// Updates target map in place from source values
// If ordered=true, processes nodes in order so later nodes see updates
```

**Python API**:
```python
labels = builder.core.update_in_place(
    new_labels,
    target=labels,
    ordered=True
)
```

**Use case**: Async updates where order matters (LPA, iterative refinement)

---

### 2.4 Alternative: Histogram-based Approach

**Rust Primitive**: `core.histogram`

```rust
pub struct HistogramStep {
    source: String,      // Map of lists
    target: String,      // Map of histograms (value -> count)
    top_k: usize,        // Only keep top K most frequent
}

// For each node: compute histogram of values in list
```

**Python API**:
```python
label_counts = builder.core.histogram(neighbor_labels, top_k=5)
argmax_labels = builder.core.argmax(label_counts)  # Get key with max count
```

**Use case**: When you need the full distribution, not just the mode

---

### Composed LPA Example

```python
def build_lpa_composed():
    """LPA using composable primitives."""
    builder = AlgorithmBuilder("lpa_composed")
    
    labels = builder.init_nodes(unique=True)
    
    with builder.iterate(10):
        # Collect neighbor labels (including own)
        neighbor_labels = builder.core.collect_neighbor_values(
            labels,
            include_self=True
        )
        
        # Find most frequent label
        new_labels = builder.core.mode(
            neighbor_labels,
            tie_break="lowest"
        )
        
        # Update in-place (async) for LPA semantics
        labels = builder.core.update_in_place(
            new_labels,
            target=labels,
            ordered=True
        )
    
    builder.attach_as("community", labels)
    return builder.build()
```

---

### Trade-offs: Monolithic vs Decomposed

**Decomposed (collect + mode + update):**
- ✅ More composable - each primitive reusable
- ✅ Better for debugging - see intermediate neighbor lists
- ✅ Flexible - can swap mode for median, custom aggregation
- ✅ Testable - each piece tested independently
- ❌ More steps = more overhead
- ❌ Neighbor list materialization (memory)

**Monolithic (majority_label_update):**
- ✅ Single pass - no intermediate lists
- ✅ Optimized - can fuse operations
- ✅ Less verbose for common case
- ❌ Less flexible - hardcoded to mode
- ❌ Can't reuse pieces for other algorithms
- ❌ Harder to debug

---

### Recommendation: **Hybrid Approach**

1. **Implement decomposed primitives first** (Phase 2a):
   - `core.collect_neighbor_values` 
   - `core.mode`
   - `core.update_in_place` (or integrate into existing steps)

2. **Add optimized monolithic version later** (Phase 2b - optional):
   - `core.majority_label_update` as a single fused step
   - Only if profiling shows significant benefit
   - Both APIs available - users choose based on needs

This gives us the best of both worlds:
- Composable building blocks for flexibility
- Optimized path for production if needed

---

### 2.2 Value Clamping (optional)

**Rust Primitive**: `core.clip`

```rust
pub struct ClipStep {
    source: String,
    target: String,
    min_value: Option<f64>,
    max_value: Option<f64>,
}

// Clamps values: target[i] = clamp(source[i], min, max)
```

**Python API**:
```python
safe_degrees = builder.core.clip(degrees, min_value=1.0)
normalized = builder.core.clip(values, min_value=0.0, max_value=1.0)
```

**Use case**: Preventing division by zero, ensuring value ranges

---

## Phase 3: Sugar & Extensions

### 3.1 Auto-detect Scalar vs Map

Extend existing primitives (`core.add`, `core.mul`, etc.) to automatically handle:
- `map + scalar` → broadcast scalar then add
- `map * scalar` → element-wise multiply

This eliminates the need for explicit `broadcast_scalar` in simple cases.

### 3.2 Graph-level Constants

Add primitives to access graph properties:
```python
n = builder.graph_node_count()
m = builder.graph_edge_count()
```

These return scalar variables that can be used in computations.

---

## Example: PageRank with New Primitives

```python
def build_pagerank_proper():
    """PageRank using explicit primitives - matches native implementation."""
    builder = AlgorithmBuilder("pagerank_proper")
    
    # Constants
    damping = 0.85
    teleport = 1.0 - damping
    
    # Initialize
    ranks = builder.init_nodes(default=1.0)
    degrees = builder.node_degrees()
    
    # Prepare for weighted aggregation
    inv_degrees = builder.core.recip(degrees, epsilon=1e-10)
    
    # Identify sinks (degree == 0)
    is_sink = builder.core.compare(degrees, "eq", 0.0)
    
    with builder.iterate(20):
        # Compute contributions: rank / out_degree
        contrib = builder.core.mul(ranks, inv_degrees)
        
        # Sum neighbor contributions (incoming edge weights)
        neighbor_sum = builder.neighbor_agg(contrib, agg="sum")
        
        # Handle sinks: collect mass from nodes with no outgoing edges
        sink_ranks = builder.core.where(is_sink, ranks, 0.0)
        sink_mass = builder.core.reduce_scalar(sink_ranks, op="sum")
        
        # Distribute sink mass uniformly
        sink_per_node = builder.core.broadcast_scalar(sink_mass)
        # Note: Need to divide by N and multiply by damping
        # This requires either graph_node_count() or accepting N as parameter
        
        # Apply PageRank formula: damping * (neighbor_sum + sink/N) + teleport/N
        damped_neighbors = builder.core.mul(neighbor_sum, damping)
        damped_sinks = builder.core.mul(sink_per_node, damping)
        # Add sink contribution (sink_per_node already divided by N)
        with_sinks = builder.core.add(damped_neighbors, damped_sinks)
        
        # Add teleport term (requires N)
        # teleport_per_node = teleport / N
        ranks = builder.core.add(with_sinks, teleport)  # Assuming broadcast
        
        # Normalize to sum=1
        ranks = builder.core.normalize_sum(ranks)
    
    builder.attach_as("pagerank", ranks)
    return builder.build()
```

**Note**: This example shows the pattern. Final implementation needs:
- Way to get/use graph node count (N)
- Scalar arithmetic (divide sink_mass by N)

---

## Example: LPA with Composed Primitives

```python
def build_lpa_proper():
    """Label Propagation using composable primitives."""
    builder = AlgorithmBuilder("lpa_proper")
    
    labels = builder.init_nodes(unique=True)
    
    with builder.iterate(10):
        # Step 1: Collect neighbor labels (including own)
        neighbor_labels = builder.core.collect_neighbor_values(
            labels,
            include_self=True
        )
        
        # Step 2: Find most frequent label
        new_labels = builder.core.mode(
            neighbor_labels,
            tie_break="lowest"
        )
        
        # Step 3: Update in-place for async semantics
        labels = builder.core.update_in_place(
            new_labels,
            target=labels,
            ordered=True
        )
    
    builder.attach_as("community", labels)
    return builder.build()
```

**Benefits of composition**:
- Each step is testable in isolation
- Can substitute `mode` with `median`, `mean`, or custom aggregation
- Can debug by inspecting `neighbor_labels` intermediate values
- Primitives reusable for other algorithms (voting, consensus, etc.)

---

## Implementation Checklist

### Phase 1: Essential (for PageRank)
- [x] `core.recip` - reciprocal with epsilon
  - [x] Rust step implementation
  - [x] Registry registration
  - [x] Python API in `CoreOps`
  - [x] Unit tests
  
- [x] `core.compare` - comparison operators
  - [x] Rust step implementation with `CompareOp` enum
  - [x] Support both scalar and map on right-hand side
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests (all ops: eq, ne, lt, le, gt, ge)
  
- [x] `core.where` - conditional selection
  - [x] Rust step implementation
  - [x] Support scalar and map for both branches
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests
  
- [x] `core.reduce_scalar` - scalar reduction
  - [x] Rust step implementation with `ReductionOp` enum
  - [x] Uses scalar storage in StepScope
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests (sum, mean, min, max)
  
- [x] `core.broadcast_scalar` - scalar expansion
  - [x] Rust step implementation
  - [x] Handle scalar variable from reduce_scalar
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests
  
- [x] `core.neighbor_agg` - weighted neighbor aggregation
  - [x] Extended existing `NeighborAggregationStep` with weights parameter
  - [x] Registry registration update
  - [x] Python API in `CoreOps`
  - [x] Unit tests (with/without weights)

### Phase 2a: Decomposed LPA Primitives
- [x] `core.collect_neighbor_values` - gather neighbor values into lists
  - [x] Rust step implementation using CSR
  - [x] Support include_self option
  - [x] Stores values as JSON arrays
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests
  
- [x] `core.mode` - find most frequent value
  - [x] Rust step implementation as `ModeListStep` (renamed from `ModeStep` to avoid conflict)
  - [x] Tie-breaking strategies (lowest, highest, keep)
  - [x] Handle JSON arrays (from collect_neighbor_values)
  - [x] Registry registration as `core.mode_list`
  - [x] Python API as `mode()` method
  - [x] Unit tests
  
- [x] `core.update_in_place` - async in-place updates
  - [x] Rust step implementation as `UpdateInPlaceStep`
  - [x] Ordered traversal for determinism (ordered=true sorts nodes)
  - [x] Update target map with source values in place
  - [x] Registry registration as `core.update_in_place`
  - [x] Python API as `update_in_place()` method
  - [x] Builder encoding support
  - [x] Unit tests
  
- [x] `core.histogram` (optional) - compute value frequencies
  - [x] Rust step implementation
  - [x] Top-K filtering
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests

### Phase 2b: Optimized Monolithic (optional)
- [ ] `core.majority_label_update` - fused LPA step
  - [ ] Only implement if profiling shows benefit
  - [ ] Single-pass neighbor collection + mode + update
  - [ ] Should match composed version exactly
  - [ ] Benchmark against decomposed version
  
- [x] `core.clip` - value clamping (nice-to-have)
  - [x] Rust step implementation
  - [x] Registry registration
  - [x] Python API
  - [x] Unit tests

### Phase 3: Polish ✅ COMPLETE
- [x] Auto-detect scalar in existing ops
  - [x] Update `core.add`, `core.mul`, etc. to handle scalars
  - [x] Tests for mixed scalar/map operations
  - [x] See `PHASE3_SCALAR_AUTO_DETECT_COMPLETE.md`
  
- [x] Graph constants
  - [x] `graph_node_count()` primitive - Returns node count as O(1) scalar
  - [x] `graph_edge_count()` primitive - Returns edge count as O(1) scalar
  - [x] Scalar variable type using `StepValue::Scalar`
  - [x] Python API: `builder.graph_node_count()` and `builder.graph_edge_count()`
  - [x] Step encoding support in `_encode_step`
  - [x] Comprehensive tests in `test_graph_constants.py`
  - [x] See `PHASE3_GRAPH_CONSTANTS_COMPLETE.md`

### Testing & Validation
- [ ] Update `benchmark_builder_vs_native.py` with proper PageRank implementation and LPA
- [ ] PageRank values should match native within tolerance
- [ ] Add test for each new primitive in isolation
- [ ] Integration test: full PageRank algorithm
- [ ] Integration test: full LPA algorithm
- [ ] Performance comparison: builder vs native

---

## Open Questions

1. **Scalar variable representation**: Should scalars be stored as single-element maps, or introduce a separate scalar variable type in the step scope?

2. **Scalar arithmetic**: Do we need `scalar + scalar` operations, or is everything always broadcast to maps?

3. **Graph constants**: How should `N` (node count) be accessed in the builder? Options:
   - Pre-compute and pass as parameter to algorithm
   - Add `graph_node_count()` primitive that returns scalar
   - Use subgraph size (available at runtime)

4. **Weighted neighbor_agg**: Should weights be:
   - Per-node weights (weight[source] applied to all edges from source)
   - Per-edge weights (require loading edge attributes)
   - Current design assumes per-node

5. **Expression system future**: Once primitives exist, should we:
   - Keep `map_nodes` for simple cases (mode, basic math)
   - Deprecate in favor of explicit primitives
   - Extend for convenience but discourage for production

---

## Success Criteria

1. **PageRank correctness**: Builder-based PageRank produces results within 1e-6 of native PageRank on test graphs
2. **LPA correctness**: Builder LPA finds same number of communities (±10%) as native LPA
3. **Performance**: Builder algorithms run within 50x of native (acceptable for composable DSL)
4. **Composability**: Primitives are reusable across multiple algorithms
5. **Testability**: Each primitive has unit tests and works in isolation
6. **Documentation**: Each primitive has clear docstrings and examples

---

## Reusability: Other Algorithms Using These Primitives

The decomposed primitives unlock many algorithms beyond PageRank and LPA:

### Using `collect_neighbor_values` + `mode`/`median`/`mean`

1. **Belief Propagation** - aggregate neighbor beliefs
2. **Consensus Algorithms** - nodes vote on values
3. **Anomaly Detection** - find outliers vs neighbor median
4. **Smoothing/Denoising** - median filter on graph-structured data

### Using `compare` + `where` + `reduce_scalar`

1. **Thresholding** - classify nodes above/below threshold
2. **Outlier Detection** - identify values beyond N standard deviations
3. **Budget Allocation** - distribute resources based on conditions
4. **Convergence Checking** - count nodes that changed significantly

### Using `recip` + `neighbor_agg` (weighted)

1. **Personalized PageRank** - with personalization weights
2. **HITS Algorithm** - authority and hub scores
3. **SimRank** - similarity propagation
4. **Influence Propagation** - with edge weights

### Using `histogram` + `argmax`

1. **K-Means on Graphs** - cluster assignment based on neighbor clusters
2. **Color Assignment** - graph coloring with conflict resolution
3. **Feature Aggregation** - summarize categorical neighbor features

### Using `update_in_place` (ordered async)

1. **Gauss-Seidel Iteration** - any iterative solver needing async updates
2. **Loopy Belief Propagation** - message passing with order dependencies
3. **Online/Streaming Algorithms** - incremental updates as data arrives

---

## Future Extensions

Once core primitives are stable, consider:

### Graph Structure Primitives
- **Triangle counting** - local clustering coefficient
- **K-core decomposition** - recursive degree filtering
- **Motif counting** - subgraph pattern matching

### Path-based Primitives
- **Distance maps** - BFS/Dijkstra distance tracking
- **Predecessor tracking** - shortest path trees
- **Path accumulation** - betweenness-style path counting

### Flow & Capacity
- **Residual capacity** - flow network augmentation
- **Cut computation** - min-cut/max-flow
- **Matching** - bipartite matching primitives

### Temporal & Dynamic
- **Windowed aggregation** - time-window based computation
- **Change detection** - delta between snapshots
- **Temporal walks** - time-respecting path traversal

The goal is to build a library of composable, performant primitives that cover common graph algorithm patterns while keeping `map_nodes` available for quick prototyping.
