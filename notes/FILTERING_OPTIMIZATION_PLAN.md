# Filtering Performance Optimization Plan

## Current State

**Performance**: 75.9x scaling (vs 50x linear ideal)  
**Bottleneck**: Nested HashMap lookups in `node_attribute_indices`  
**Root Cause**: `HashMap<NodeId, HashMap<AttrName, usize>>` structure

## Architectural Changes Required for Linear Scaling

### 1. **Flatten Node Attribute Indices** (High Impact, Medium Effort)

**Current Structure:**
```rust
node_attribute_indices: HashMap<NodeId, HashMap<AttrName, usize>>
```

**Proposed Structure:**
```rust
node_attribute_indices: HashMap<(NodeId, AttrName), usize>
```

**Implementation Requirements:**
- Change all `get_node_attr_index` calls from 2-level to 1-level lookup
- Update `set_attr_index` to use composite keys
- Modify node removal to scan and remove all `(node_id, *)` entries
- **Estimated Impact**: 40-60% improvement in lookup performance

**Files to Change:**
- `src/core/space.rs` - Change HashMap structure and all accessor methods
- `src/core/query.rs` - Update all attribute lookup calls
- `src/api/graph.rs` - Update any direct attribute access
- All test files that use attribute operations

### 2. **Columnar Node Storage** (High Impact, High Effort)

**Current Structure:**
```rust
active_nodes: HashSet<NodeId>
node_attribute_indices: HashMap<(NodeId, AttrName), usize>
```

**Proposed Structure:**
```rust
// Store nodes in dense vectors for cache locality
node_ids: Vec<NodeId>,                              // [1, 2, 3, 4, ...]
node_active: Vec<bool>,                             // [true, false, true, true, ...]
node_attr_indices: HashMap<AttrName, Vec<Option<usize>>>,  // "dept" -> [Some(5), None, Some(7), ...]
```

**Benefits:**
- **Cache locality**: Sequential memory access instead of hash table jumps
- **Vectorization**: Can use SIMD operations for filtering
- **Memory efficiency**: Dense storage vs sparse hash tables

**Implementation Requirements:**
- Redesign `GraphSpace` entirely
- Add node ID → vector index mapping
- Implement active node compaction/defragmentation
- Update all node iteration to use vector indices
- **Estimated Impact**: 2-3x improvement in large-scale operations

### 3. **SIMD-Optimized Filtering** (Medium Impact, Medium Effort)

**Current Approach:**
```rust
for (node_id, attr_opt) in node_attr_pairs {
    if let Some(attr_value) = attr_opt {
        if filter.matches(attr_value) {
            matching_nodes.push(node_id);
        }
    }
}
```

**Proposed Approach:**
```rust
// For numeric comparisons, use vectorized operations
let indices: Vec<usize> = // ... collect all indices
let values: Vec<f64> = pool.get_numeric_values_simd(&indices);
let mask: Vec<bool> = simd_compare_greater_than(&values, threshold);
let results: Vec<NodeId> = apply_mask(&node_ids, &mask);
```

**Implementation Requirements:**
- Add SIMD-specific attribute access methods to `GraphPool`
- Implement vectorized comparison functions for common types
- Add feature flags for different SIMD instruction sets (AVX2, AVX512, NEON)
- **Estimated Impact**: 2-4x improvement for numeric filtering

### 4. **Attribute Indexing** (Medium Impact, High Effort)

**Current Approach:**
- Linear scan through all nodes for each filter operation

**Proposed Approach:**
```rust
// Pre-built indices for frequently filtered attributes
attribute_indices: HashMap<AttrName, HashMap<AttrValue, Vec<NodeId>>>
// Example: "department" -> { "Engineering" -> [1, 5, 9], "Marketing" -> [2, 6, 8] }
```

**Benefits:**
- **O(1) lookup** for exact-match filters instead of O(n) scan
- **Range queries** possible with sorted indices
- **Composite indices** for multi-attribute filters

**Implementation Requirements:**
- Add index management system to `GraphSpace`
- Implement index update triggers on attribute changes
- Add query optimizer to choose between scan vs index lookup
- Memory management for large indices
- **Estimated Impact**: 10-100x improvement for exact-match queries

### 5. **Memory Layout Optimization** (Low Impact, Low Effort)

**Current Issues:**
- Fragmented allocations from nested HashMaps
- Poor cache utilization due to random access patterns

**Proposed Solutions:**
```rust
// Use arena allocators for better memory locality
use bumpalo::Bump;
attribute_arena: Bump,

// Pool allocator for HashMap entries
use slotmap::SlotMap;
attribute_slots: SlotMap<AttrKey, AttrValue>,
```

**Implementation Requirements:**
- Replace standard HashMap with arena-backed versions
- Implement custom allocators for attribute storage
- Add memory pool management for frequently allocated/deallocated objects
- **Estimated Impact**: 10-20% improvement

## Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
1. **Flatten Node Attribute Indices** (#1)
2. **Memory Layout Optimization** (#5)
3. **Expected Result**: 50-70% better scaling

### Phase 2: Major Restructuring (1-2 months)
1. **Columnar Node Storage** (#2)
2. **SIMD-Optimized Filtering** (#3)
3. **Expected Result**: Near-linear scaling for most operations

### Phase 3: Advanced Features (2-3 months)
1. **Attribute Indexing** (#4)
2. **Query Optimization**
3. **Expected Result**: Better than linear for many query patterns

## Risk Assessment

### Low Risk
- **Flatten Node Attribute Indices**: Straightforward refactoring
- **Memory Layout Optimization**: Additive changes

### Medium Risk
- **SIMD Optimization**: Platform-specific code, testing complexity
- **Attribute Indexing**: Index consistency and memory management

### High Risk
- **Columnar Node Storage**: Fundamental architecture change, affects all components

## Alternative Approaches

### Option A: Hybrid Architecture
Keep current HashMap for flexibility, add columnar storage for performance-critical paths:
```rust
// Fast path for bulk operations
columnar_storage: Option<ColumnarNodeStorage>,
// Fallback for complex operations  
hash_storage: HashMap<NodeId, HashMap<AttrName, usize>>,
```

### Option B: Pluggable Storage Backends
Design abstraction layer allowing different storage strategies:
```rust
trait NodeStorage {
    fn get_attr_index(&self, node_id: NodeId, attr: &AttrName) -> Option<usize>;
    fn filter_nodes(&self, filter: &NodeFilter) -> Vec<NodeId>;
}

// Implementations: HashMapStorage, ColumnarStorage, IndexedStorage
```

### Option C: Incremental Migration
Implement optimizations one attribute type at a time:
```rust
enum AttributeStorage {
    HashMap(HashMap<NodeId, usize>),        // Legacy
    Vector(Vec<Option<usize>>),             // Dense numeric attrs
    Index(BTreeMap<AttrValue, Vec<NodeId>>), // Categorical attrs
}
```

## Performance Targets

**Current Performance** (50K nodes):
- Single attribute filter: ~10ms (4.7M nodes/sec)
- Complex AND filter: ~9ms  
- Scaling: 75.9x (vs 50x linear)

**Target Performance** (50K nodes):
- Single attribute filter: ~1ms (50M nodes/sec)
- Complex AND filter: ~2ms
- Scaling: 55x (90% linear efficiency)

**Stretch Goals** (50K nodes):
- With indexing: ~0.1ms (500M nodes/sec)
- Perfect linear scaling: 50x

## Testing Strategy

1. **Benchmark Suite**: Comprehensive performance tests across scales
2. **Regression Testing**: Ensure correctness through architectural changes  
3. **Memory Profiling**: Track allocation patterns and cache behavior
4. **Platform Testing**: Validate SIMD optimizations across architectures
5. **Integration Testing**: End-to-end workflow validation

## Conclusion

The current 75.9x scaling can be improved to near-linear with targeted architectural changes. The highest-impact, lowest-risk approach is to start with flattening the node attribute indices, then gradually adopt more advanced optimizations based on real-world usage patterns.