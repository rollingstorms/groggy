# Query Performance Optimization Plan

## Current Performance Issues (From Test Results)

### Scaling Problem: 116x time increase for 50x nodes (O(n²) behavior)
- **Target**: Linear O(n) scaling  
- **Current**: Worse than O(n²) scaling
- **Root cause**: Pseudo-bulk processing with individual lookups

### AND Filter Overhead: 1.6x time per additional filter
- **Target**: Constant or logarithmic overhead per filter
- **Current**: Linear overhead multiplying total time
- **Root cause**: Recursive calls with vector copying

## Critical Code Issues

### Issue 1: Fake "Bulk" Processing
```rust
// CURRENT: Pretends to be bulk but does N individual lookups
let entity_indices: Vec<(NodeId, Option<usize>)> = nodes
    .iter()
    .map(|&node_id| (node_id, space.get_node_attr_index(node_id, name))) // ❌ N lookups!
    .collect();
```

### Issue 2: Recursive Vector Copying  
```rust
// CURRENT: Creates new vectors for each sub-filter
for sub_filter in filters {
    current_nodes = self.filter_nodes_columnar(&current_nodes, pool, space, sub_filter)?; // ❌ Vector copy
}
```

### Issue 3: Missing True Bulk Methods
The code calls `pool.get_attribute_column()` but this method doesn't exist with proper bulk optimization.

## Solution: Real Columnar Bulk Processing

### Phase 1: True Bulk Attribute Retrieval
```rust
impl GraphPool {
    /// Get attributes for ALL nodes in a single operation
    pub fn get_node_attributes_bulk(
        &self,
        space: &GraphSpace,
        attr_name: &AttrName,
        node_ids: &[NodeId]
    ) -> Vec<(NodeId, Option<AttrValue>)> {
        // Single hash map lookup + single vector traversal
        // Instead of N individual lookups
    }
}
```

### Phase 2: Bitset-Based Filtering
```rust
// Use bitsets instead of vector copying for AND operations
struct NodeFilterSet {
    bitset: BitVec,
    node_mapping: Vec<NodeId>,
}

impl NodeFilterSet {
    fn and_filter(&mut self, other: &Self) {
        self.bitset &= &other.bitset; // Vectorized AND operation
    }
    
    fn to_node_vec(&self) -> Vec<NodeId> {
        // Single pass to collect matching nodes
    }
}
```

### Phase 3: SIMD Vectorization
```rust
// For numeric comparisons
#[cfg(target_arch = "x86_64")]
fn filter_numeric_avx2(values: &[f64], threshold: f64, op: CompareOp) -> BitVec {
    // Process 4-8 values simultaneously with SIMD
}
```

## Implementation Steps

### Step 1: Add Real Bulk Methods to GraphPool ✅
- `get_node_attributes_bulk()`  
- `get_edge_attributes_bulk()`

### Step 2: Replace Fake Bulk with Real Bulk ✅
- Remove individual `get_node_attr_index()` calls
- Use single bulk retrieval per attribute

### Step 3: Add Bitset-Based AND/OR Processing ✅
- Replace vector copying with bitset operations
- Enable early termination without vector allocation

### Step 4: Add SIMD Optimizations (Phase 2)
- AVX2 for numeric comparisons
- String SIMD for text matching

## Expected Performance Gains

### From Bulk Processing:
- **Simple filters**: 3x-5x improvement
- **Attribute lookups**: 5x-10x improvement  

### From Bitset AND/OR:
- **Complex filters**: 4x-8x improvement
- **Memory usage**: 50-90% reduction

### From SIMD:
- **Numeric filters**: 4x-8x improvement
- **Text matching**: 2x-4x improvement

### Overall Target:
- **Linear O(n) scaling** instead of O(n²)
- **10x-20x overall improvement** for complex queries
- **Competitive with NetworkX** performance
