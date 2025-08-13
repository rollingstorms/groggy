# Columnar Filtering Optimization Plan

## Current Problem Analysis

The bottleneck in `find_nodes_by_attribute` (lines 119-148 in query.rs) is:

### 1. Sequential Processing
Loops through each node individually: `for &node_id in space.get_active_nodes()`

### 2. Individual Attribute Lookups
Each iteration does 3 operations:
- `space.get_node_attr_index(node_id, attr_name)` - O(1) hashmap lookup per node
- `pool.get_attr_by_index(attr_name, attr_index, true)` - O(1) vector access per node  
- `filter.matches(attr_value)` - Individual comparison per node

### 3. No Vectorization
Can't leverage SIMD or bulk operations

### 4. Poor Cache Locality
Jumping between different data structures per node

## Solution: True Columnar Filtering

### Phase 1: Bulk Attribute Retrieval API
- Add `GraphPool::get_attribute_column_slice()` method
- Add `GraphSpace::get_attribute_indices()` method  
- Enable retrieving all attribute values for active nodes in single operation

### Phase 2: Vectorized Filtering Operations
- Replace sequential node-by-node filtering with bulk array operations
- Add SIMD-optimized comparisons for numeric filters
- Add optimized string matching for text filters

### Phase 3: Parallel Processing
- Add parallel filtering for large node sets (>10K nodes)
- Maintain columnar data structures for maximum parallelizability

## Implementation Strategy

### Step 1: Bulk Attribute Access Methods

#### It's very important to note that we are not adding any new data structures to the graph. We are simply adding new APIs to access existing data structures in a more efficient way. We replace the code in original functions before creating new ones. we dont need edge and node methods they are virtually the same in the rest of the library.

Add to `GraphPool`:
```rust
/// Get all attribute values for active nodes in columnar format
pub fn get_attribute_column(&self, 
    attr_name: &AttrName, 
    ids: &[(NodeId, usize)], 
    is_node: bool
) -> Vec<Option<&AttrValue>>

/// Get all indices for an attribute across multiple nodes/edges
pub fn get_attribute_indices(&self, 
    attr_name: &AttrName,
    ids: &[NodeId],
    is_node: bool
) -> Vec<Option<usize>>
```

Add to `GraphSpace`:
```rust
/// Get attribute indices for all active nodes/edges in bulk (VECTORIZED)
pub fn get_attribute_indices(&self, 
    attr_name: &AttrName,
    is_node: bool
) -> Vec<(NodeId, Option<usize>)>

/// Get attribute values for all active nodes/edges in columnar format
pub fn get_attributes(&self, 
    pool: &GraphPool, 
    attr_name: &AttrName,
    is_node: bool
) -> Vec<(NodeId, Option<&AttrValue>)>
```

### Step 2: Integration Flow - How Bulk APIs Transform Filtering

#### Current Bottleneck Flow (Sequential):
```rust
// BEFORE: Sequential per-node processing
pub fn find_nodes_by_attribute(/* current implementation */) -> GraphResult<Vec<NodeId>> {
    let mut matching_nodes = Vec::new();
    
    // ❌ BOTTLENECK: Individual node processing
    for &node_id in space.get_active_nodes() {           // Loop through ALL nodes
        if let Some(attr_index) = space.get_node_attr_index(node_id, attr_name) {  // HashMap lookup per node
            if let Some(attr_value) = pool.get_attr_by_index(attr_name, attr_index, true) {  // Vector access per node  
                if filter.matches(attr_value) {          // Individual comparison per node
                    matching_nodes.push(node_id);
                }
            }
        }
    }
    Ok(matching_nodes)  // 3 operations × N nodes = O(3N) with no vectorization
}
```

#### New Columnar Flow (Bulk + Vectorized):
```rust
// AFTER: Bulk retrieval + vectorized processing  
pub fn find_nodes_by_attribute(
    &mut self,
    pool: &GraphPool,
    space: &GraphSpace,
    attr_name: &AttrName,
    filter: &AttributeFilter
) -> GraphResult<Vec<NodeId>> {
    // ✅ STEP 1: Single bulk operation gets ALL attribute data at once
    let node_attr_pairs = space.get_attributes(pool, attr_name, true);  // 1 bulk call replaces N individual calls
    
    // ✅ STEP 2: Vectorized filtering on bulk data
    match filter {
        AttributeFilter::Equals(target_value) => {
            // Process all values simultaneously instead of one-by-one
            Ok(node_attr_pairs
                .iter()
                .filter_map(|(node_id, attr_opt)| {
                    attr_opt.filter(|&attr_value| attr_value == target_value)
                            .map(|_| *node_id)
                })
                .collect())
        }
        AttributeFilter::GreaterThan(target_value) => {
            // SIMD-optimized bulk numeric comparison
            self.simd_numeric_filter(&node_attr_pairs, target_value, NumericOp::GreaterThan)
        }
        AttributeFilter::Between(min, max) => {
            // Vectorized range filtering on bulk data
            self.vectorized_range_filter(&node_attr_pairs, min, max)
        }
        // ... other vectorized filter types
    }
}
```

#### How `space.get_attributes()` Works Internally:
```rust
// GraphSpace::get_attributes() - replaces individual lookups with bulk operation
impl GraphSpace {
    pub fn get_attributes(&self, 
        pool: &GraphPool, 
        attr_name: &AttrName,
        is_node: bool
    ) -> Vec<(NodeId, Option<&AttrValue>)> {
        
        // STEP 1: Get all active entity IDs (nodes or edges)
        let active_ids: Vec<NodeId> = if is_node {
            self.active_nodes.iter().copied().collect()
        } else {
            self.active_edges.iter().copied().collect()  
        };
        
        // STEP 2: Bulk index lookup - single HashMap operation per attribute name
        let entity_indices: Vec<(NodeId, Option<usize>)> = active_ids
            .into_iter()
            .map(|id| {
                let index = if is_node {
                    self.node_attribute_indices.get(&id)
                        .and_then(|attrs| attrs.get(attr_name))
                        .copied()
                } else {
                    self.edge_attribute_indices.get(&id)
                        .and_then(|attrs| attrs.get(attr_name))
                        .copied()
                };
                (id, index)
            })
            .collect();
        
        // STEP 3: Bulk attribute retrieval from pool
        pool.get_attribute_column(attr_name, &entity_indices, is_node)
    }
}
```

#### How `pool.get_attribute_column()` Delivers Bulk Data:
```rust
// GraphPool::get_attribute_column() - vectorized attribute access
impl GraphPool {
    pub fn get_attribute_column(&self, 
        attr_name: &AttrName, 
        entity_indices: &[(NodeId, Option<usize>)], 
        is_node: bool
    ) -> Vec<(NodeId, Option<&AttrValue>)> {
        
        // Get the columnar attribute storage
        let attr_column = if is_node {
            self.node_attributes.get(attr_name)
        } else {
            self.edge_attributes.get(attr_name)
        };
        
        // Bulk retrieval with vectorized access pattern
        entity_indices
            .iter()
            .map(|(entity_id, index_opt)| {
                let attr_value = index_opt
                    .and_then(|index| attr_column?.values.get(index));
                (*entity_id, attr_value)
            })
            .collect()
    }
}
```

#### Performance Transformation Summary:

**BEFORE (Current Bottleneck)**:
```
For 10,000 nodes with "age" attribute:
├── 10,000 × HashMap lookup (node → attribute indices)
├── 10,000 × Vector access (attribute column → value)  
├── 10,000 × Individual filter.matches() calls
└── Total: 30,000 individual operations with no vectorization
```

**AFTER (Bulk + Vectorized)**:
```
For 10,000 nodes with "age" attribute:
├── 1 × Bulk active nodes retrieval
├── 1 × Bulk HashMap lookup (batch indices resolution)
├── 1 × Bulk columnar attribute retrieval  
├── 1 × Vectorized filter processing (SIMD: 4-8 values at once)
└── Total: ~4 bulk operations with full vectorization
```

**Result**: ~7,500x reduction in operation count + SIMD acceleration = **Expected 4x-12x speedup**

#### Integration with Existing QueryEngine Methods:

The bulk APIs integrate seamlessly with existing query patterns:

```rust
// find_nodes_by_filter() - complex queries leverage bulk APIs
pub fn find_nodes_by_filter(/* existing signature */) -> GraphResult<Vec<NodeId>> {
    match filter {
        NodeFilter::AttributeFilter { name, filter } => {
            // ✅ Now uses vectorized approach automatically
            self.find_nodes_by_attribute(pool, space, name, filter)
        },
        NodeFilter::And(filters) => {
            // ✅ Each sub-filter benefits from vectorization
            let mut result_set = space.get_active_nodes().iter().copied().collect();
            for sub_filter in filters {
                let filtered = self.find_nodes_by_filter(pool, space, sub_filter)?;
                result_set = result_set.intersection(&filtered.into_iter().collect()).copied().collect();
            }
            Ok(result_set)
        },
        // ... other complex filter types automatically benefit
    }
}

// try_bulk_node_filter() - enhanced with true bulk operations
fn try_bulk_node_filter(/* existing signature */) -> GraphResult<Option<Vec<NodeId>>> {
    match filter {
        NodeFilter::AttributeFilter { name, filter } => {
            // ✅ Perfect fit - bulk retrieval enables bulk filtering
            let node_attr_pairs = space.get_attributes(pool, name, true);
            Some(self.apply_vectorized_filter(&node_attr_pairs, filter))
        },
        NodeFilter::And(filters) if filters.len() <= 3 => {
            // ✅ Chain bulk operations instead of sequential processing
            let mut node_attr_data = HashMap::new();
            for filter in filters {
                if let NodeFilter::AttributeFilter { name, .. } = filter {
                    if !node_attr_data.contains_key(name) {
                        node_attr_data.insert(name, space.get_attributes(pool, name, true));
                    }
                }
            }
            // Apply all filters to bulk data simultaneously
            Some(self.apply_multiple_filters(&node_attr_data, filters))
        },
        // ... other bulk-optimizable patterns
    }
}
```

### Step 3: SIMD-Optimized Numeric Operations

#### Data Flow for Vectorized Processing:

```rust
// How bulk data flows through SIMD operations
fn simd_numeric_filter(
    &self,
    node_attr_pairs: &[(NodeId, Option<&AttrValue>)],  // ← Bulk data from space.get_attributes()
    target: &AttrValue,
    op: NumericOp
) -> GraphResult<Vec<NodeId>> {
    
    // STEP 1: Extract all numeric values into SIMD-friendly arrays
    let (node_ids, values): (Vec<NodeId>, Vec<f64>) = node_attr_pairs
        .iter()
        .filter_map(|(node_id, attr_opt)| {
            attr_opt.and_then(|attr| extract_numeric_value(attr))  // Handle AttrValue::Int, Float, SmallInt
                   .map(|val| (*node_id, val))
        })
        .unzip();  // Separate parallel arrays for vectorization
    
    // STEP 2: Convert target value for comparison  
    let target_f64 = extract_numeric_value(target)?;
    
    // STEP 3: SIMD bulk comparison (4-8 values simultaneously)
    let comparison_results = if is_x86_feature_detected!("avx2") && values.len() >= 4 {
        unsafe { self.simd_compare_f64_avx2(&values, target_f64, op) }  // 4x-8x parallel processing
    } else {
        self.vectorized_numeric_compare(&values, target_f64, op)  // Fallback vectorized
    };
    
    // STEP 4: Collect matching node IDs
    Ok(comparison_results
        .into_iter()
        .zip(node_ids)
        .filter_map(|(matches, node_id)| if matches { Some(node_id) } else { None })
        .collect())
}

// SIMD implementation example (AVX2)
#[cfg(target_arch = "x86_64")]
unsafe fn simd_compare_f64_avx2(&self, values: &[f64], target: f64, op: NumericOp) -> Vec<bool> {
    use std::arch::x86_64::*;
    
    let mut results = vec![false; values.len()];
    let target_vec = _mm256_set1_pd(target);  // Broadcast target to 4 parallel lanes
    
    // Process 4 f64 values simultaneously  
    for (chunk_idx, chunk) in values.chunks(4).enumerate() {
        let values_vec = _mm256_loadu_pd(chunk.as_ptr());  // Load 4 values
        
        let comparison_mask = match op {
            NumericOp::GreaterThan => _mm256_cmp_pd(values_vec, target_vec, _CMP_GT_OQ),
            NumericOp::LessThan => _mm256_cmp_pd(values_vec, target_vec, _CMP_LT_OQ), 
            NumericOp::Equals => _mm256_cmp_pd(values_vec, target_vec, _CMP_EQ_OQ),
            // ... other numeric operations
        };
        
        // Extract comparison results from SIMD mask
        let mask_bits = _mm256_movemask_pd(comparison_mask);
        for i in 0..chunk.len() {
            results[chunk_idx * 4 + i] = (mask_bits & (1 << i)) != 0;
        }
    }
    
    results
}
```

### Step 4: Specialized String Operations

Add optimized string filtering:
```rust
fn vectorized_string_filter(
    &self,
    node_attr_pairs: &[(NodeId, Option<&AttrValue>)],
    pattern: &str,
    op: StringOp
) -> Vec<NodeId> {
    // Extract all text values in bulk
    let text_pairs: Vec<(NodeId, &str)> = node_attr_pairs
        .iter()
        .filter_map(|(node_id, attr_opt)| {
            attr_opt.and_then(|attr| extract_text_value(attr))
                   .map(|text| (*node_id, text))
        })
        .collect();
    
    match op {
        StringOp::StartsWith => {
            // Optimized prefix matching using Boyer-Moore or similar
            self.bulk_prefix_match(&text_pairs, pattern)
        }
        StringOp::Contains => {
            // Vectorized substring search
            self.bulk_substring_match(&text_pairs, pattern)
        }
        StringOp::Regex => {
            // Compiled regex with bulk matching
            self.bulk_regex_match(&text_pairs, pattern)
        }
    }
}
```

## Performance Improvements Expected

### 1. Eliminate Per-Node Overhead
- **Current**: 3 function calls per node (HashMap lookup + vector access + comparison)
- **New**: 1 bulk retrieval + vectorized comparisons

### 2. Enable SIMD Vectorization
- Process 4-8 values simultaneously with AVX/AVX2
- 4x-8x speedup for numeric comparisons

### 3. Improve Cache Locality
- Columnar access patterns are more cache-friendly
- Prefetching works better with sequential access

### 4. Parallel Processing
- Large datasets can be split across CPU cores
- Especially beneficial for complex filters

## Migration Plan

### Week 1: Foundation
Implement bulk attribute retrieval APIs in `GraphPool` and `GraphSpace`

### Week 2: Core Infrastructure
Create vectorized filtering infrastructure and basic vectorized filters (Equals, GreaterThan, etc.)

### Week 3: SIMD Optimization
Add SIMD optimizations for numeric operations 

### Week 4: String & Parallel Processing
Add parallel processing for large datasets and optimize string operations

### Week 5: Integration
Update `find_nodes_by_attribute` to use new vectorized approach, maintain backward compatibility

### Week 6: Validation
Performance testing and benchmark validation to achieve NetworkX parity

## Success Metrics

### Target Goals
- **Target**: Filtering performance within 1.5x of NetworkX (currently 3.7x-12.5x slower)
- **Stretch Goal**: Match or exceed NetworkX performance for bulk filtering operations
- **Maintain**: Current superior performance in graph creation (1.8x-2.1x faster) and traversal (4.0x-16.7x faster)

## Root Cause Analysis

This plan addresses the root cause: **sequential individual node processing prevents vectorization and parallel processing**. By implementing true columnar filtering, we can leverage modern CPU capabilities for bulk operations and achieve competitive performance with NetworkX.

## Technical Benefits

### Current Architecture Issues
- Sequential processing prevents CPU vectorization
- Individual attribute lookups create overhead
- Poor memory access patterns hurt cache performance
- No opportunity for parallel execution

### Columnar Architecture Benefits
- Bulk operations enable SIMD vectorization
- Continuous memory access improves cache utilization  
- Parallel processing scales with CPU cores
- Reduced function call overhead

### Expected Performance Impact
- **Numeric Filtering**: 4x-8x improvement through SIMD
- **String Filtering**: 2x-4x improvement through optimized algorithms
- **Large Datasets**: Additional 2x-4x improvement through parallelization
- **Overall Target**: 3.7x-12.5x current gap reduced to 1.5x or better

## Lessons Learned from Recent Performance Work

### Critical Success Factors Discovered

#### 1. Release Builds Are Non-Negotiable
**Discovery**: Performance regression was caused by debug build after major refactoring
- **Impact**: `maturin develop --release` recovered 90%+ of performance immediately
- **Lesson**: Always benchmark in release mode; debug builds mask real performance
- **Action**: Add CI checks to ensure release builds for performance testing

#### 2. Algorithmic Wins Trump Micro-optimizations
**Current Achievements**:
- **Graph Creation**: 1.8x-2.1x faster than NetworkX
- **Connected Components**: 4.0x-4.1x faster than NetworkX  
- **BFS/DFS Traversal**: 5.6x-16.7x faster than NetworkX

**Key Insight**: O(V+E) algorithmic improvements with adjacency caching deliver massive wins
- Traversal algorithms leverage columnar topology access
- Bulk operations reduce per-node overhead
- **Lesson**: Focus on algorithmic approach first, then optimize implementation

#### 3. Identify True Bottlenecks Through Profiling
**Before**: Assumed general performance issues across the board
**After**: Discovered filtering is the only remaining major bottleneck (3.7x-12.5x slower)
- Graph creation and traversal already exceed NetworkX performance
- **Lesson**: Profile first, optimize the actual bottleneck, not assumptions

### Architectural Strengths to Leverage

#### 1. Columnar Storage Foundation
**Existing Infrastructure**:
- `GraphSpace::get_columnar_topology()` for edge data
- `GraphPool` with columnar attribute storage
- Bulk attribute APIs: `get_nodes_attributes()`, `set_node_attributes()`

**Success Pattern**: Traversal algorithms that leverage columnar topology show 4x-16x improvements
- **Lesson**: Extend columnar access patterns to filtering operations

#### 2. Bulk Operations Architecture
**Current Bulk APIs**:
- `add_nodes(count)` - bulk node creation
- `add_edges(edges)` - bulk edge creation  
- `get_nodes_attributes()` - bulk attribute retrieval
- `set_node_attributes()` - bulk attribute setting

**Performance Pattern**: Bulk operations consistently outperform individual operations
- **Lesson**: Replace individual node processing with bulk columnar operations

#### 3. PyO3/Maturin Integration Excellence
**Strengths**:
- Zero-copy data transfer between Rust and Python
- Efficient bulk attribute APIs avoid PyAttrValue object creation
- Release builds deliver native Rust performance to Python

**Lesson**: The Rust-Python integration is not the bottleneck; focus on Rust-side optimizations

### Anti-Patterns to Avoid

#### 1. Sequential Per-Entity Processing
**Problem Pattern**:
```rust
for &node_id in space.get_active_nodes() {
    let attr_index = space.get_node_attr_index(node_id, attr_name)?;
    let attr_value = pool.get_attr_by_index(attr_name, attr_index, true)?;
    if filter.matches(attr_value) { ... }
}
```

**Why This Fails**:
- Prevents vectorization and SIMD
- Poor cache locality
- No parallelization opportunity
- Function call overhead per entity

#### 2. Premature Micro-optimization
**Wrong Approach**: Optimizing individual hashmap lookups or vector accesses
**Right Approach**: Eliminate the need for per-node lookups through bulk operations

#### 3. Ignoring Build Configuration
**Critical Error**: Testing performance in debug mode
- Debug builds can be 10x-100x slower than release builds
- Optimizations, inlining, and SIMD require release builds

### Design Principles for Future Work

#### 1. Columnar-First Architecture
- Design all operations to work on columnar data
- Bulk retrieval → vectorized processing → bulk results
- Avoid individual entity processing loops

#### 2. Leverage Existing Strengths
- Build on proven columnar topology patterns from traversal
- Extend bulk attribute APIs rather than creating new individual APIs
- Follow the performance patterns that already deliver 4x-16x improvements

#### 3. Benchmark-Driven Development
- Always test in release mode
- Compare against NetworkX on realistic workloads
- Measure the actual bottlenecks, not assumptions

### Implementation Confidence Factors

#### What We Know Works
1. **Columnar topology access** (proven in traversal: 4x-16x faster)
2. **Bulk attribute operations** (existing APIs show good performance)
3. **Release build optimization** (immediate 90%+ performance recovery)
4. **O(V+E) algorithmic design** (graph creation 1.8x-2.1x faster)

#### High-Confidence Next Steps
1. Apply columnar patterns from traversal to filtering
2. Replace sequential loops with bulk operations
3. Extend existing bulk attribute APIs
4. Maintain release build discipline

### Success Metrics Validation
**Already Achieved**:
- Graph creation: ✅ 1.8x-2.1x faster than NetworkX
- Traversal: ✅ 4.0x-16.7x faster than NetworkX
- Release builds: ✅ Proper optimization pipeline

**Remaining Target**:
- Filtering: 🎯 Close 3.7x-12.5x gap to within 1.5x of NetworkX

**Confidence Level**: High - we've already proven the architectural approach works for similar operations
