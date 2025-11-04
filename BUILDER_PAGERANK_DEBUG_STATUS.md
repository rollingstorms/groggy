# Builder PageRank Debugging Status

## Current Issue

The builder-based PageRank implementation produces values that differ from the native implementation on larger graphs. The maximum differences observed are:
- 100-node graph: max diff = 0.0094 (target: < 0.0000005)  
- 50k-node graph: max diff ≈ 0.00004

## Root Cause Identified

The `node_degrees()` step in the builder is returning **incorrect degree values**:

### Test Case: A→B→C (nodes 0→1→2)
**Expected degrees:**
- Node 0: out_degree = 1 (has edge 0→1)
- Node 1: out_degree = 1 (has edge 1→2)  
- Node 2: out_degree = 0 (sink, no outgoing edges)

**Actual degrees from builder:**
- Node 0: degree = 0 ❌
- Node 1: degree = 1 ✓
- Node 2: degree = 1 ❌

The degrees are completely wrong for nodes 0 and 2.

## Why This Breaks PageRank

PageRank computes `contrib[i] = rank[i] / out_degree[i]`, then sums contributions from neighbors. With wrong degrees:
- Node 0 has degree=0 → `contrib[0] = rank[0] / 0` → division by zero (handled by recip epsilon, but gives huge value)
- Node 2 has degree=1 instead of 0 → treated as non-sink → doesn't contribute to sink mass redistribution

## Investigation Details

### Python Builder Code  
```python
def node_degrees(self, nodes: VarHandle) -> VarHandle:
    var = self._new_var("degrees")
    self.steps.append({
        "type": "node_degree",
        "input": nodes.name,    # ← Passed but not used by Rust!
        "output": var.name
    })
    return var
```

### Rust Step Registration (registry.rs)
```rust
registry.register(
    "core.node_degree",
    ...,
    |spec| {
        let target = spec.params.expect_text("target")?;  // Only reads "target"
        Ok(Box::new(NodeDegreeStep::new(target)))
    },
)?;
```

The step registration only extracts `target`, not `input`.

### Rust Step Implementation (structural.rs:39-50)
```rust
fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
    let subgraph = scope.subgraph();
    let graph = subgraph.graph();
    let graph_ref = graph.borrow();
    
    let mut map = HashMap::with_capacity(subgraph.node_set().len());
    for &node in scope.node_ids() {
        let degree = graph_ref.out_degree(node).unwrap_or(0);  // Should be correct
        map.insert(node, AlgorithmParamValue::Int(degree as i64));
    }
    ...
}
```

The implementation calls `graph_ref.out_degree(node)` which should return correct values.

## Hypothesis

There are two possibilities:

1. **Mismatch between step spec and execution**: The Python builder is using `"type": "node_degree"` but maybe it should be using `"id": "core.node_degree"` and passing params differently?

2. **Node ordering issue**: The degrees are being computed correctly but stored/retrieved with wrong node IDs somehow?

## Next Steps

1. ✅ Verify native PageRank computes correct degrees (confirmed working)
2. ✅ Identify that builder node_degrees returns wrong values  
3. ⏭️ Check how other steps (like init_nodes) handle node mapping - do they have the same issue?
4. ⏭️ Compare the step spec format between working steps and node_degrees
5. ⏭️ Add logging to NodeDegreeStep::apply to see what it actually computes vs what gets returned

## Workaround Possibility

Since we know native PageRank works correctly, we could:
- Use the native PageRank directly (defeats purpose of builder)
- OR fix the node_degrees step to actually work correctly

## Files to Fix

- `python-groggy/python/groggy/builder.py` - node_degrees() method
- `src/algorithms/steps/structural.rs` - NodeDegreeStep implementation  
- `src/algorithms/steps/registry.rs` - step registration

