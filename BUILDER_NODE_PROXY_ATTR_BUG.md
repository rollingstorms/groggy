# Builder Node Proxy Attribute Access Bug

**Date:** 2025-11-03  
**Status:** Root cause identified, needs FFI fix

## Summary

PageRank and LPA builder algorithms are producing incorrect results due to a bug in how `NodeProxy.__getattr__` accesses node attributes. The underlying Rust code computes and stores values correctly, but Python attribute access returns wrong values.

## Root Cause

When accessing attributes via `node.attribute_name` on a NodeProxy object returned from `result.nodes` iteration:
- The attribute lookup is returning values from the **wrong position** in the attribute column
- Direct calls to `result.get_node_attribute(node_id, attr_name)` return **correct** values
- The bug is in the Python FFI `NodeProxy.__getattr__` implementation

## Evidence

Test case: 3-node directed graph (0→1→2), computing out-degrees:

```python
g = Graph(directed=True)
n0, n1, n2 = g.add_node(), g.add_node(), g.add_node()
g.add_edge(n0, n1)  # n0 has degree 1
g.add_edge(n1, n2)  # n1 has degree 1, n2 has degree 0
```

### Rust Layer (Correct)
```
[DEBUG NodeDegreeStep] Node 0: degree = 1  ✅
[DEBUG NodeDegreeStep] Node 1: degree = 1  ✅
[DEBUG NodeDegreeStep] Node 2: degree = 0  ✅

[DEBUG AttachNodeAttrStep] Node 0: Int(1)  ✅
[DEBUG AttachNodeAttrStep] Node 2: Int(0)  ✅
[DEBUG AttachNodeAttrStep] Node 1: Int(1)  ✅

[DEBUG set_node_attrs] Node 0 -> index 0  ✅
[DEBUG set_node_attrs] Node 2 -> index 1  ✅
[DEBUG set_node_attrs] Node 1 -> index 2  ✅
```

Column storage: `[1, 0, 1]` at indices `[0, 1, 2]`
- Node 0 → index 0 → value 1
- Node 2 → index 1 → value 0
- Node 1 → index 2 → value 1

### Python Layer (Broken)

**Via NodeProxy attribute access (WRONG):**
```python
for node in result.nodes:
    print(f"Node {node.id}: {node.degree}")
# Output:
#   Node 0: 1  ✅
#   Node 1: 2  ❌ (should be 1)
#   Node 2: 1  ❌ (should be 0)
```

**Via get_node_attribute (CORRECT):**
```python
for node_id in [0, 1, 2]:
    val = result.get_node_attribute(node_id, 'degree')
    print(f"Node {node_id}: {val}")
# Output:
#   Node 0: 1  ✅
#   Node 1: 1  ✅
#   Node 2: 0  ✅
```

## Impact

- **PageRank**: With 20 iterations, nodes end up with swapped values (diff ~0.13-0.29)
- **LPA**: Community counts differ by ~32 communities (35179 vs 35147)
- **All builder algorithms**: Any algorithm using `node_degrees` or similar primitives will have incorrect results when accessed via `node.attr`

## Secondary Issue Fixed

Also fixed loop aliasing bug in `python-groggy/python/groggy/builder.py:998-1005`:
- Changed `var_mapping[target] = source` to `var_mapping[target] = remapped_source`
- This ensures loop iterations properly chain: iteration N+1 reads the output of iteration N
- Without this fix, all iterations were reading from the initial variable (e.g., `nodes_0`)

## Next Steps

### P0: Fix NodeProxy Attribute Access

**Location:** `python-groggy/src/ffi/entities/node.rs:__getattr__`

The `__getattr__` implementation calls:
```rust
self.inner.get_attribute(&name.into())
```

Where `self.inner` is a `groggy::entities::Node` (from `groggy::traits::GraphEntity`).

**The bug is likely in one of these paths:**

1. **GraphEntity::get_attribute** implementation (`src/entities/node.rs` or `src/traits/`)
   - This method may be using an incorrect index when looking up columnar attributes
   - Check if it's using an iteration position instead of the stored attribute index

2. **Index lookup in Graph::get_node_attr**
   - The path from GraphEntity → Graph storage may be passing wrong indices
   - Compare with how `Graph::get_node_attribute` (the working path) does it

3. **Columnar attribute storage/retrieval**
   - When attributes are stored in columns, the mapping `node_id → column_index` may be corrupted
   - Or the retrieval is using `node_position` instead of `node_id → stored_index`

### Debug Strategy

1. **Add logging to GraphEntity::get_attribute:**
   ```rust
   // In src/entities/node.rs or relevant trait impl
   fn get_attribute(&self, attr_name: &AttrName) -> Result<Option<AttrValue>> {
       eprintln!("[DEBUG GraphEntity::get_attribute] Node {}, attr '{}'", 
                 self.id(), attr_name);
       // ... existing code ...
       eprintln!("[DEBUG GraphEntity::get_attribute]   Using index: {:?}, got value: {:?}", 
                 index, value);
       // ... return value ...
   }
   ```

2. **Compare with Subgraph::get_node_attribute:**
   - This method works correctly, so trace its path
   - Look for differences in how it retrieves columnar attributes
   - Likely in `src/traits/subgraph_operations.rs`

3. **Check Node entity initialization:**
   - When `Node::new(node_id, graph)` is called during iteration
   - Does it store any cached indices that become stale?
   - Look at `src/entities/node.rs::new()`

### Most Likely Fix

Based on the symptoms, the issue is probably in how `GraphEntity::get_attribute` looks up the column index for a node. It may be:

```rust
// WRONG: using some iteration position or cached value
let index = self.position_in_iteration;  // ❌

// CORRECT: should look up the stored index for this specific node
let index = graph.space().get_node_attr_index(self.id(), attr_name);  // ✅
```

### Workaround (Temporary)
Until fixed, builder algorithm users should:
```python
# DON'T do this:
for node in result.nodes:
    value = node.my_attr  # ❌ WRONG

# DO this instead:
for node in result.nodes:
    value = result.get_node_attribute(node.id, 'my_attr')  # ✅ CORRECT
```

## Test Cases to Add

Once fixed, add these regression tests:

### test_node_proxy_attr_access.py
```python
def test_node_proxy_vs_direct_access():
    """NodeProxy.attr should match get_node_attribute."""
    g = Graph(directed=True)
    nodes = [g.add_node() for _ in range(10)]
    
    # Create non-uniform graph
    for i in range(9):
        g.add_edge(nodes[i], nodes[i+1])
    
    builder = AlgorithmBuilder('test')
    vals = builder.init_nodes(default=1.0)
    degrees = builder.node_degrees(vals)
    builder.attach_as('degree', degrees)
    
    result = g.view().apply(builder.build())
    
    for node in result.nodes:
        proxy_val = node.degree
        direct_val = result.get_node_attribute(node.id, 'degree')
        assert proxy_val == direct_val, \
            f"Node {node.id}: proxy={proxy_val}, direct={direct_val}"
```

### test_builder_pagerank_correctness.py
```python
def test_pagerank_matches_native():
    """Builder PageRank should match native within tight tolerance."""
    g = create_test_graph(100, avg_degree=5)
    
    native = g.view().apply(
        centrality.pagerank(max_iter=20, damping=0.85, output_attr='pr')
    )
    
    builder_algo = build_pagerank_algorithm(100, damping=0.85, max_iter=20)
    builder = g.view().apply(builder_algo)
    
    for node in native.nodes:
        native_val = node.pr
        builder_val = builder.get_node_attribute(node.id, 'pagerank')
        assert abs(native_val - builder_val) < 1e-6, \
            f"Node {node.id}: native={native_val}, builder={builder_val}"
```

## Files Modified

1. `python-groggy/python/groggy/builder.py:1000` - Fixed loop alias chaining
2. `src/algorithms/steps/structural.rs:64` - Added debug logging (remove after fix)
3. `src/algorithms/steps/attributes.rs:121` - Added debug logging (remove after fix)
4. `src/traits/subgraph_operations.rs:1210` - Added debug logging (remove after fix)

## Rollback Instructions

If the FFI fix causes issues:
```bash
git diff python-groggy/src/ffi/ > ffi_changes.patch
git checkout HEAD -- python-groggy/src/ffi/
# Review and selectively reapply from patch
```

## References

- Original issue notes: `BUILDER_DEBUG_NODE_DEGREES_ISSUE.md`
- Loop debugging: `BUILDER_PAGERANK_DEBUG_STATUS.md`
- Benchmark script: `benchmark_builder_vs_native.py`
