# DataFrame/Batch Optimization Ideas - Extracted for Future Work

This document contains all the DataFrame and batch optimization ideas that were extracted from the recent commits to preserve them for future development in an isolated branch.

## Overview

These optimizations were removed from the main codebase because they caused 3x performance slowdowns in core graph operations. However, they contain valuable ideas for future DataFrame-style batch operations that could be implemented properly with careful benchmarking.

## Python Core Optimizations (from python/groggy/graph/core.py)

### Vectorized Filtering Methods
- **vectorized_filter_nodes()**: Efficient vectorized attribute-based filtering using numpy-style operations
- **vectorized_filter_edges()**: Batch edge filtering with optimized attribute lookups
- **multi_criteria_filter()**: Complex multi-attribute filtering with AND/OR logic

### Batch Operations
- **bulk_add_nodes()**: Chunked bulk node creation with optimized attribute handling
- **bulk_add_edges()**: Efficient batch edge creation with bulk attribute assignment
- **bulk_update_attributes()**: Mass attribute updates with minimal overhead

### DataFrame Export Methods
- **to_dataframe()**: Convert graph data to pandas DataFrame format
- **nodes_to_dataframe()**: Export node data as DataFrame with attribute columns
- **edges_to_dataframe()**: Export edge data as DataFrame with endpoint and attribute columns

## Rust Backend Optimizations (from src/graph/core.rs)

### Auto-Optimized Attribute Retrieval

#### Smart Bulk Operations
```rust
/// Get a specific attribute for multiple nodes efficiently  
/// Auto-optimizes based on dataset size and density
pub fn get_nodes_attribute(
    &self,
    attr_name: &str,
    node_ids: Vec<String>
) -> PyResult<HashMap<String, pyo3::PyObject>>
```

**Key Features:**
- Automatically switches to bulk retrieval for requests > 10 nodes
- Uses columnar store optimization for large datasets
- Falls back to individual lookups for sparse data

#### Vectorized Multi-Attribute Access
```rust
/// Get all attributes for multiple nodes efficiently
/// Auto-optimizes based on request pattern and data density
pub fn get_nodes_attributes(
    &self,
    node_ids: Vec<String>
) -> PyResult<HashMap<String, HashMap<String, pyo3::PyObject>>>
```

**Key Features:**
- Bulk multi-attribute retrieval for requests > 5 nodes
- Batch conversion of JSON to Python objects
- Optimized memory allocation with capacity pre-calculation

#### Single Attribute Statistics
```rust
/// Get a specific attribute for all nodes efficiently (useful for statistics)
/// Auto-optimizes based on data density and sparsity
pub fn get_all_nodes_attribute(
    &self,
    attr_name: &str
) -> PyResult<HashMap<String, pyo3::PyObject>>
```

**Key Features:**
- Vectorized retrieval for large graphs (> 500 nodes)
- Automatic density detection for optimization strategy selection
- Single-pass extraction with minimal lookups

### DataFrame-Optimized Export Methods

#### Ultra-Fast DataFrame Export
```rust
/// Ultra-fast DataFrame-style attribute retrieval - bypasses Python conversion overhead
/// Returns data in format optimized for pandas/polars DataFrame creation
pub fn get_dataframe_data(
    &self,
    attr_names: Option<Vec<String>>,
    node_ids: Option<Vec<String>>
) -> PyResult<HashMap<String, Vec<pyo3::PyObject>>>
```

**Key Features:**
- Columnar store's optimized DataFrame export method
- Parallel-friendly chunk processing
- Pre-allocated result structures

#### Vectorized Bulk Retrieval
```rust
/// Ultra-fast vectorized bulk attribute retrieval using columnar store optimization
pub fn get_bulk_node_attribute_vectors(
    &self,
    attr_names: Vec<String>,
    node_ids: Option<Vec<String>>
) -> PyResult<HashMap<String, (Vec<usize>, Vec<PyObject>)>>
```

**Key Features:**
- Highly optimized columnar store vectorized method
- Batch JSON to Python conversion in chunks
- Efficient index to node ID mapping

### Chunked Processing for Large Datasets

#### Chunked Attribute Updates
```rust
/// Chunked bulk attribute updates for maximum performance with large datasets
pub fn set_node_attributes_chunked(
    &mut self,
    updates: &PyDict,
    chunk_size: Option<usize>
) -> PyResult<()>
```

**Key Features:**
- Default 10k chunk size for memory optimization
- Reduces lock contention in concurrent scenarios
- Skips missing nodes gracefully

#### High-Performance Batch Creation
```rust
/// High-performance batch node creation with chunked processing
pub fn add_nodes_chunked(
    &mut self,
    nodes_data: Vec<(String, Option<&PyDict>)>,
    chunk_size: Option<usize>
) -> PyResult<()>
```

**Key Features:**
- Chunked processing to optimize memory usage
- Bulk operations with optimized attribute handling
- Configurable chunk sizes based on workload

## Columnar Store Optimizations (from src/storage/columnar.rs)

### DataFrame-Optimized Bulk Retrieval

#### Vectorized Attribute Extraction
```rust
/// DataFrame-optimized bulk attribute retrieval - returns raw vectors for fast conversion
/// This method is designed for maximum performance when extracting data for pandas/polars
pub fn bulk_get_node_attribute_vectors(
    &self,
    attr_names: &[String],
    node_indices: Option<&[usize]>
) -> HashMap<String, (Vec<usize>, Vec<JsonValue>)>
```

**Key Features:**
- Pre-allocated vector capacity based on data size
- Minimized DashMap lookups with upfront attr_uid retrieval
- Separate handling for filtered vs. full retrieval

#### Ultra-Fast Single Attribute Access
```rust
/// Ultra-fast single attribute retrieval for statistics - returns parallel vectors
/// Optimized for cases like "get all salaries" or "get all weights"
pub fn get_single_attribute_fast(
    &self,
    attr_name: &str,
    is_node: bool
) -> Option<(Vec<usize>, Vec<JsonValue>)>
```

**Key Features:**
- Single pass extraction with no additional lookups
- Pre-allocated vectors for known capacity
- Consistent index ordering for reliable results

### Advanced Bulk Operations

#### Multi-Attribute Bulk Retrieval
```rust
/// Bulk multi-attribute retrieval for multiple nodes
/// Returns HashMap<node_index, HashMap<attr_name, JsonValue>>
pub fn get_multiple_attributes_bulk(
    &self,
    node_indices: &[usize]
) -> Option<HashMap<usize, HashMap<String, JsonValue>>>
```

**Key Features:**
- Upfront attr_uid collection to minimize lookups
- Per-node attribute collection with efficient iteration
- Graceful handling of sparse attribute patterns

#### DataFrame Table Export
```rust
/// High-performance node attribute table export for DataFrame libraries
/// Returns (node_indices, attribute_names, value_matrix) for maximum efficiency
pub fn export_node_attribute_table(
    &self,
    attr_names: Option<&[String]>,
    node_indices: Option<&[usize]>
) -> (Vec<usize>, Vec<String>, Vec<Vec<Option<JsonValue>>>)
```

**Key Features:**
- Aligned value matrix for consistent DataFrame structure
- Efficient None handling for missing values
- Optimal data layout for pandas/polars conversion

### Pandas Integration Methods

#### Direct Pandas Dictionary Export
```rust
/// DataFrame conversion methods for high-performance batch export
/// Returns formatted data ready for pandas/polars
pub fn to_pandas_dict(
    &self,
    attr_names: Option<Vec<String>>,
    node_indices: Option<Vec<usize>>
) -> PyResult<HashMap<String, Vec<PyObject>>>
```

**Key Features:**
- Direct Python object conversion in batches
- Includes node_id column for DataFrame indexing
- Batch error handling with early termination

#### Ultra-Fast Column Extraction
```rust
/// Ultra-fast single column extraction for DataFrame operations
pub fn get_column_fast(
    &self,
    attr_name: &str,
    node_indices: Option<Vec<usize>>
) -> PyResult<(Vec<usize>, Vec<PyObject>)>
```

**Key Features:**
- Single column extraction optimized for analytics
- Supports both filtered and full retrieval modes
- Minimal Python GIL interaction

## Implementation Strategy for Future Work

### Phase 1: Isolated Branch Development
1. Create a dedicated `dataframe-optimizations` branch
2. Implement optimizations incrementally with benchmarks
3. Ensure core API performance is never compromised

### Phase 2: Benchmarking Framework
1. Create comprehensive benchmarks comparing:
   - Core operations (add_node, filter_nodes, etc.)
   - Bulk operations vs. individual operations
   - DataFrame export performance vs. individual attribute access
2. Set performance thresholds:
   - Core operations must maintain current speed
   - Bulk operations should be 2x+ faster than equivalent individual calls
   - DataFrame exports should be 5x+ faster than manual conversion

### Phase 3: Auto-Optimization Logic
1. Implement smart thresholds that automatically choose:
   - Individual vs. bulk operations based on request size
   - Dense vs. sparse retrieval strategies based on data patterns
   - Memory vs. speed tradeoffs based on available resources

### Phase 4: API Design Considerations
1. Keep DataFrame operations in separate methods to avoid breaking core API
2. Provide clear performance characteristics documentation
3. Add optional DataFrame dependency to avoid bloating core library

## Performance Optimization Principles

### Avoid These Patterns (Lessons Learned)
- **Don't vectorize small operations**: Core operations like single node addition should remain simple
- **Don't auto-optimize core API**: Keep optimization explicit and optional
- **Don't mix paradigms**: Separate graph operations from DataFrame operations

### Follow These Patterns
- **Chunked processing**: Break large operations into memory-efficient chunks
- **Lazy evaluation**: Only convert to Python objects when actually needed
- **Smart thresholds**: Use data size and density to choose optimal strategies
- **Separate APIs**: Keep DataFrame operations separate from core graph API

## Testing Strategy

### Performance Regression Tests
1. Benchmark core operations on every commit
2. Ensure DataFrame optimizations don't slow down core API
3. Test memory usage patterns with large datasets

### Correctness Tests
1. Verify DataFrame exports match individual attribute access
2. Test edge cases with sparse and dense data
3. Validate chunked operations produce identical results to batch operations

### Integration Tests
1. Test with real pandas/polars workflows
2. Validate performance improvements in realistic scenarios
3. Test memory efficiency with large graph datasets

## Future Research Areas

### Advanced Optimizations
1. **SIMD acceleration**: Use CPU vector instructions for numeric operations
2. **Parallel processing**: Multi-threaded attribute retrieval for large datasets
3. **Memory mapping**: Zero-copy data sharing with DataFrame libraries

### Smart Caching
1. **Attribute access patterns**: Cache frequently accessed combinations
2. **Columnar layout optimization**: Reorganize data based on access patterns
3. **Lazy materialization**: Only compute DataFrame columns when accessed

### Integration Opportunities
1. **Apache Arrow**: Native columnar format for zero-copy DataFrame integration
2. **Polars**: Direct integration with Rust-native DataFrame library
3. **DuckDB**: SQL query interface over graph data

---

**Note**: This document represents extracted optimization ideas that caused performance regressions when implemented in the main codebase. They should be reintroduced carefully in an isolated branch with proper benchmarking to ensure they enhance rather than degrade performance.
