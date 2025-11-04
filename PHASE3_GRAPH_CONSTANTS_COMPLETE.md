# Phase 3: Graph Constants Implementation - Complete

## Summary

Successfully implemented graph constant primitives (`graph_node_count` and `graph_edge_count`) to enable algorithms like PageRank to access graph properties at runtime. These primitives create O(1) scalar variables that can be used in arithmetic operations with node maps.

## Changes Made

### 1. Rust Core Changes

#### Added Graph Constant Steps (`src/algorithms/steps/init.rs`)
- **`GraphNodeCountStep`**: Returns the number of nodes in the subgraph as a scalar
- **`GraphEdgeCountStep`**: Returns the number of edges in the subgraph as a scalar
- Both steps have O(1) time complexity (just accessing `subgraph.node_count()` / `subgraph.edge_count()`)
- Store results as `Scalar` variables using `scope.variables_mut().set_scalar()`

```rust
pub struct GraphNodeCountStep {
    target: String,
}

impl Step for GraphNodeCountStep {
    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let count = scope.subgraph().node_count();
        scope.variables_mut().set_scalar(
            self.target.clone(),
            AlgorithmParamValue::Int(count as i64),
        );
        Ok(())
    }
}
```

#### Registry Registration (`src/algorithms/steps/registry.rs`)
- Registered `core.graph_node_count` with factory function
- Registered `core.graph_edge_count` with factory function
- Both parse `target` parameter for the output scalar variable name

#### Module Exports (`src/algorithms/steps/mod.rs`)
- Exported `GraphNodeCountStep` and `GraphEdgeCountStep` from init module

### 2. Python Builder Changes (`python-groggy/python/groggy/builder.py`)

#### Added Builder API Methods
```python
def graph_node_count(self) -> VarHandle:
    """Get the number of nodes in the current subgraph as a scalar variable."""
    var = self._new_var("n")
    self.steps.append({
        "type": "graph_node_count",
        "output": var.name
    })
    return var

def graph_edge_count(self) -> VarHandle:
    """Get the number of edges in the current subgraph as a scalar variable."""
    var = self._new_var("m")
    self.steps.append({
        "type": "graph_edge_count",
        "output": var.name
    })
    return var
```

#### Added Step Encoding Support (`_encode_step` method)
```python
if step_type == "graph_node_count":
    return {
        "id": "core.graph_node_count",
        "params": {"target": step["output"]}
    }

if step_type == "graph_edge_count":
    return {
        "id": "core.graph_edge_count",
        "params": {"target": step["output"]}
    }
```

### 3. Testing

#### Created Comprehensive Test Suite (`test_graph_constants.py`)
Five test cases covering:

1. **`test_graph_node_count()`** - Basic node count retrieval
   - Creates graph with 10 nodes
   - Retrieves count as scalar, broadcasts to node map, attaches as attribute
   - Verifies count equals 10

2. **`test_graph_edge_count()`** - Basic edge count retrieval
   - Creates graph with 5 nodes and 4 edges
   - Retrieves count as scalar, broadcasts to node map
   - Verifies count equals 4

3. **`test_graph_constants_in_arithmetic()`** - Using graph constants in computations
   - Computes uniform distribution: 1/N for each node
   - Tests scalar-map division (1.0 / N)
   - Verifies each node has value 0.1 (±1e-6 tolerance)

4. **`test_pagerank_with_node_count()`** - Real algorithm usage
   - Implements simplified PageRank iteration with teleport term
   - Computes teleport probability: (1-damping) / N
   - Verifies correct update: rank + teleport_term

5. **`test_spec_encoding()`** - Pipeline specification validation
   - Builds algorithm using both graph constants
   - Verifies `core.graph_node_count` and `core.graph_edge_count` appear in spec
   - Ensures proper serialization for Rust execution

All tests pass successfully.

## API Usage Examples

### Basic Usage
```python
builder = AlgorithmBuilder("example")
n = builder.graph_node_count()
m = builder.graph_edge_count()

# Scalars must be broadcast to use with attach_as
ref = builder.init_nodes(default=0.0)
n_map = builder.core.broadcast_scalar(n, ref)
builder.attach_as("node_count", n_map)
```

### PageRank Teleport Probability
```python
builder = AlgorithmBuilder("pagerank")
n = builder.graph_node_count()
ranks = builder.init_nodes(default=1.0)

# Compute (1 - damping) / N
damping = 0.85
teleport = 1.0 - damping
teleport_const = builder.init_nodes(default=teleport)
teleport_prob = builder.core.div(teleport_const, n)

with builder.iterate(20):
    # ... PageRank computation using teleport_prob
    ranks = builder.core.add(ranks, teleport_prob)

builder.attach_as("pagerank", ranks)
```

### Uniform Distribution
```python
builder = AlgorithmBuilder("uniform")
ones = builder.init_nodes(default=1.0)
n = builder.graph_node_count()
uniform = builder.core.div(ones, n)  # Each node gets 1/N
builder.attach_as("probability", uniform)
```

## Design Notes

### Scalar vs Map Distinction

Graph constants return **scalar variables**, not node maps:
- **Scalar**: Single value stored as `StepValue::Scalar(AlgorithmParamValue)`
- **Node Map**: HashMap with one entry per node

This is efficient because:
- O(1) storage instead of O(n)
- O(1) creation time instead of O(n)
- Natural representation of graph-level constants

### Broadcasting Required for attach_as

The `attach_as` step expects node maps, so scalars must be broadcast first:

```python
# ✗ This fails - attach_as needs a node map
n = builder.graph_node_count()
builder.attach_as("count", n)  # RuntimeError: variable 'n_0' is not a node map

# ✓ This works - broadcast scalar to all nodes
n = builder.graph_node_count()
ref = builder.init_nodes(default=0.0)
n_map = builder.core.broadcast_scalar(n, ref)
builder.attach_as("count", n_map)
```

### Arithmetic Operations with Scalars

The auto-detection system (Phase 3) handles scalar-map operations:
- `map + scalar` → broadcasts scalar then adds
- `map * scalar` → broadcasts scalar then multiplies
- `scalar + map` → reversed order also works
- `map / scalar` → divides each map value by scalar

However, `scalar + scalar` operations require at least one operand to be a map:
```python
# ✗ This fails - no map operand
uniform = builder.core.div(1.0, n)  # RuntimeError: requires at least one map operand

# ✓ This works - ones is a map
ones = builder.init_nodes(default=1.0)
uniform = builder.core.div(ones, n)
```

## Performance

### Time Complexity
- **Creation**: O(1) - just reads subgraph metadata
- **Storage**: O(1) - single scalar value
- **Arithmetic**: Same as before - the scalar is broadcast on-demand during operations

### Comparison with Old Approach

If we created full node maps for graph constants:
```python
# Hypothetical "bad" approach
n_map = builder.init_nodes(default=graph.node_count())  # O(n) storage
m_map = builder.init_nodes(default=graph.edge_count())  # O(n) storage
```

New scalar approach:
```python
# Actual implementation
n = builder.graph_node_count()  # O(1) storage
m = builder.graph_edge_count()  # O(1) storage
```

**Savings**: For algorithms that use multiple graph constants across iterations, this saves O(k × n) storage where k is the number of constants.

## Integration with Existing Primitives

Graph constants work seamlessly with Phase 1 and Phase 2 primitives:

### With Arithmetic Operations (Phase 1)
```python
n = builder.graph_node_count()
degrees = builder.node_degrees()
avg_degree = builder.core.div(degrees, n)  # Per-node average
```

### With Scalar Operations (Phase 3)
```python
n = builder.graph_node_count()
m = builder.graph_edge_count()

# Compute graph density (would need scalar arithmetic)
# For now, need to broadcast first:
n_map = builder.core.broadcast_scalar(n, ref)
m_map = builder.core.broadcast_scalar(m, ref)
n_squared = builder.core.mul(n_map, n_map)
density = builder.core.div(m_map, n_squared)
```

### With Reductions (Phase 1)
```python
n = builder.graph_node_count()
ranks = builder.init_nodes(default=1.0)
total_rank = builder.core.reduce_scalar(ranks, op="sum")

# Verify normalization: total_rank should equal n
```

## Validation Results

### Existing Tests
- ✅ All 29 tests in `tests/test_builder_core.py` pass
- ✅ No regressions in existing functionality

### New Tests
- ✅ `test_graph_node_count()` - Basic retrieval
- ✅ `test_graph_edge_count()` - Basic retrieval
- ✅ `test_graph_constants_in_arithmetic()` - Computation usage
- ✅ `test_pagerank_with_node_count()` - Real algorithm
- ✅ `test_spec_encoding()` - Serialization

### Build Status
- ✅ Rust code compiles without errors
- ✅ Python extension builds successfully with `maturin develop --release`
- ⚠️ 8 unused variable warnings in unrelated code (pre-existing)

## What's Next

With Phase 3 complete, all planned primitives are now implemented:
- ✅ Phase 1: Core arithmetic and logic primitives
- ✅ Phase 2: Decomposed LPA primitives
- ✅ Phase 3: Scalar auto-detection and graph constants

Next steps:
1. **Build proper PageRank** using all primitives (recip, neighbor_agg, graph_node_count)
2. **Build proper LPA** using decomposed primitives (collect_neighbor_values, mode, update_in_place)
3. **Update benchmark script** to test builder vs native implementations
4. **Validate correctness** - ensure builder algorithms match native results
5. **Performance testing** - measure overhead of composable primitives

## Files Changed

### Rust Core
- `src/algorithms/steps/init.rs` - Added `GraphNodeCountStep` and `GraphEdgeCountStep`
- `src/algorithms/steps/mod.rs` - Exported new steps
- `src/algorithms/steps/registry.rs` - Registered `core.graph_node_count` and `core.graph_edge_count`

### Python Builder
- `python-groggy/python/groggy/builder.py` - Added `graph_node_count()`, `graph_edge_count()`, and encoding support

### Tests
- `test_graph_constants.py` - Comprehensive test suite for graph constants (5 tests)

---

**Phase 3 Graph Constants Status**: ✅ Complete  
**All Builder Primitives Status**: ✅ Complete (Phases 1-3 done)  
**Next Phase**: Build and validate PageRank & LPA algorithms
