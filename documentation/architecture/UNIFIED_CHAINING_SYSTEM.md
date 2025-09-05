# Unified Chaining System Architecture

## Overview

The Groggy graph library now features a unified chaining system that provides consistent, high-performance array operations across all data types. The system supports both eager and lazy evaluation, with automatic optimization and type-safe operations.

## Core Components

### 1. Foundation Layer - BaseArray

```rust
pub struct BaseArray {
    data: Vec<AttrValue>,
    dtype: AttrValueType,
    name: Option<String>,
}
```

- **Columnar storage**: Optimized for analytical operations
- **Type consistency**: Enforces single data type per array
- **Memory efficient**: Compact representation of graph attributes

### 2. Specialized Arrays - Type Safety

```rust
pub struct NodesArray { /* NodeId collections */ }
pub struct EdgesArray { /* EdgeId collections */ }  
pub struct MetaNodeArray { /* MetaNode collections */ }
```

**Benefits:**
- **Type safety**: Prevents mixing node IDs with edge IDs
- **Trait-based methods**: Automatic method availability based on element type
- **Graph integration**: Maintains optional graph references for advanced operations

### 3. Dual Evaluation System

#### Eager Evaluation (ArrayIterator)
```rust
let results = nodes
    .iter()                    // Create eager iterator
    .filter_by_degree(5)       // Execute immediately  
    .take(10)                  // Execute immediately
    .collect();                // Already materialized
```

#### Lazy Evaluation (LazyArrayIterator)
```rust
let results = nodes
    .lazy_iter()               // Create lazy iterator
    .filter_by_degree(5)       // Queue operation
    .take(10)                  // Queue operation  
    .collect()?;               // Execute all operations with optimization
```

## Trait-Based Method Injection

Methods become available automatically based on element type:

### NodeIdLike Operations
```rust
impl NodeIdLike for NodeId {}

// Available methods for ArrayIterator<NodeId>:
nodes.filter_by_degree(min_degree)
nodes.get_neighbors()  
nodes.to_subgraph()
```

### EdgeLike Operations  
```rust
impl EdgeLike for EdgeId {}

// Available methods for ArrayIterator<EdgeId>:
edges.filter_by_weight(min_weight)
edges.filter_by_endpoints(source_pred, target_pred)
edges.group_by_source()
```

### SubgraphLike Operations
```rust
impl SubgraphLike for Subgraph {}

// Available methods for ArrayIterator<Subgraph>:
subgraphs.filter_nodes("age > 25")
subgraphs.filter_edges("weight > 0.5")
subgraphs.collapse(aggregations)
```

### MetaNodeLike Operations
```rust
impl MetaNodeLike for MetaNode {}

// Available methods for ArrayIterator<MetaNode>:
meta_nodes.expand()
meta_nodes.re_aggregate(new_aggs)
```

## Lazy Evaluation Optimizations

The lazy evaluation system provides several performance optimizations:

### 1. Operation Fusion
```rust
// Multiple filters are combined into a single pass
data.lazy_iter()
    .filter("age > 25")
    .filter("active = true")  
    .filter("score > 0.8")
    // Becomes: filter("age > 25 AND active = true AND score > 0.8")
```

### 2. Operation Reordering
```rust
// Operations are reordered for optimal performance:
// 1. Skip (reduces working set early)
// 2. Filters (reduce data as early as possible)  
// 3. Sample (after filters but before expensive ops)
// 4. Take (late to allow other ops to reduce data first)
// 5. Map/Transform (work on final reduced set)
// 6. Collapse (final aggregation)
```

### 3. Early Termination
```rust
// When take(n) is present, processing stops at n elements
large_dataset.lazy_iter()
    .filter("expensive_predicate")
    .take(10)  // Only processes until 10 matches found
    .collect()
```

### 4. Memory Efficiency
```rust
// Avoids intermediate collections until final collect()
// Pre-allocates result vectors based on size hints
// Uses reservoir sampling for efficient random sampling
```

## Usage Patterns

### Basic Array Operations
```rust
use groggy::storage::array::{BaseArray, ArrayOps};

let data = vec![
    AttrValue::Int(1),
    AttrValue::Int(2), 
    AttrValue::Int(3)
];

let array = BaseArray::new(data, AttrValueType::Int);

// Eager evaluation
let doubled: Vec<_> = array.iter()
    .map(|x| x.as_int().unwrap() * 2)
    .collect();

// Lazy evaluation with optimization
let result = array.lazy_iter()
    .filter("value > 1")
    .take(5)
    .collect()?;
```

### Graph-Aware Operations
```rust
use groggy::storage::array::NodesArray;

let nodes = NodesArray::with_graph(node_ids, graph.clone());

// Node-specific operations become available
let high_degree_nodes = nodes.iter()
    .filter_by_degree(10)        // Only available for NodeIdLike
    .get_neighbors()             // Returns ArrayIterator<Vec<NodeId>>
    .collect();
```

### Subgraph Processing
```rust
use groggy::subgraphs::Subgraph;

let subgraphs = vec![subgraph1, subgraph2, subgraph3];

let meta_nodes = ArrayIterator::new(subgraphs)
    .filter_nodes("age > 30")         // Filter nodes within each subgraph
    .filter_edges("weight > 0.7")     // Filter edges within each subgraph  
    .collapse(aggregations)           // Convert to meta-nodes
    .collect();
```

## Performance Characteristics

### Lazy vs Eager Trade-offs

| Aspect | Eager | Lazy |
|--------|-------|------|
| **Memory** | Higher (intermediate collections) | Lower (deferred execution) |
| **CPU** | Multiple passes | Single optimized pass |
| **Latency** | Lower for simple operations | Higher setup cost |  
| **Throughput** | Lower for complex chains | Higher for complex chains |
| **Debugging** | Easier to debug intermediate results | Harder to debug deferred operations |

### Recommendation Guidelines

**Use Eager (.iter()) when:**
- Simple 1-2 operation chains
- Need to debug intermediate results
- Working with small datasets (< 1000 elements)
- Operations have side effects

**Use Lazy (.lazy_iter()) when:**
- Complex operation chains (3+ operations)
- Large datasets (> 1000 elements)  
- Performance is critical
- Memory usage is a concern

## Integration with BaseTable

The array system integrates seamlessly with the BaseTable system:

```rust
use groggy::storage::table::BaseTable;

let table = BaseTable::new();
table.add_column("node_id", node_array);
table.add_column("degree", degree_array);

// Query returns arrays that support chaining
let high_degree = table.query("degree > 10")?
    .get_column("node_id")?
    .lazy_iter()
    .take(100)
    .collect()?;
```

## Error Handling

All operations return `GraphResult<T>` for consistent error handling:

```rust
let result = nodes.lazy_iter()
    .filter_by_degree(5)
    .collect()
    .map_err(|e| {
        eprintln!("Operation failed: {}", e);
        e
    })?;
```

## Extension Points

The system is designed for extensibility:

### Adding New Array Types
```rust
pub struct CustomArray {
    data: Vec<CustomType>,
    // ... other fields
}

impl ArrayOps<CustomType> for CustomArray {
    // Implement required methods
}

// Automatic lazy_iter() support via trait default
```

### Adding New Marker Traits
```rust
pub trait CustomLike {
    // Marker trait for custom operations
}

impl<T: CustomLike> ArrayIterator<T> {
    pub fn custom_operation(self) -> Self {
        // Custom operation implementation
    }
}

impl<T: CustomLike> LazyArrayIterator<T> {
    pub fn custom_operation(self) -> Self {
        // Lazy version of custom operation
    }
}
```

## Benchmarking and Performance

The system includes comprehensive benchmarking tools:

```rust
use groggy::storage::array::{Benchmarker, BenchmarkConfig};

let config = BenchmarkConfig {
    data_sizes: vec![1000, 10000, 100000],
    operation_chains: vec![
        "filter -> take".to_string(),
        "filter -> filter -> sample".to_string(),
    ],
    iterations: 10,
};

let results = Benchmarker::run_comparison(config)?;
println!("Lazy is {}% faster", results.performance_improvement());
```

This unified system provides the foundation for high-performance graph operations while maintaining type safety and ease of use.