# Rust Core Architecture

This document provides a comprehensive overview of Groggy's Rust core architecture, explaining how the various subsystems work together to provide high-performance graph processing with unified storage views.

## Table of Contents

1. [Overview](#overview)
2. [Core Subsystems](#core-subsystems)
3. [Pool Management](#pool-management)
4. [Space Management](#space-management)
5. [History Tracking](#history-tracking)
6. [Storage Views](#storage-views)
7. [Data Flow Architecture](#data-flow-architecture)
8. [Memory Management](#memory-management)
9. [Performance Optimizations](#performance-optimizations)

## Overview

Groggy's Rust core is designed around a layered architecture that separates concerns while enabling high-performance operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python FFI Layer                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 High-Level Graph API                       │
│                   (src/api/graph.rs)                       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Storage Views                           │
│       Array          Matrix           Table                │
│   (array.rs)       (matrix.rs)     (table.rs)             │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Core Subsystems                         │
│   Pool Management    Space Management    History Tracking   │
│    (pool.rs)          (space.rs)         (history.rs)     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Foundation Types                          │
│              (types.rs, errors.rs)                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Subsystems

### Pool Management (`src/core/pool.rs`)

The Pool is the central data store that manages all graph entities (nodes, edges) and their attributes in a columnar format.

#### Key Responsibilities

1. **Entity Storage**: Centralized storage for all nodes and edges
2. **Attribute Management**: Columnar storage of entity attributes with type safety
3. **Memory Pooling**: Efficient reuse of memory buffers to reduce allocation overhead
4. **Index Management**: Maintains various indices for fast lookups and queries

#### Core Data Structures

```rust
pub struct Pool {
    // Entity storage
    nodes: NodeStorage,
    edges: EdgeStorage,
    
    // Attribute storage (columnar)
    node_attributes: AttributeStorage,
    edge_attributes: AttributeStorage,
    
    // Memory pools for efficiency
    attribute_memory_pool: AttributeMemoryPool,
    
    // Indices and caches
    node_index: HashMap<NodeId, EntityIndex>,
    edge_index: HashMap<EdgeId, EntityIndex>,
    adjacency_cache: AdjacencyCache,
    
    // Statistics and metadata
    stats: PoolStatistics,
    config: PoolConfig,
}
```

#### Attribute Storage Architecture

Attributes are stored in a columnar format for optimal performance:

```rust
pub struct AttributeStorage {
    columns: HashMap<AttrName, AttributeColumn>,
    type_registry: TypeRegistry,
    memory_pool: AttributeMemoryPool,
}

pub struct AttributeColumn {
    name: AttrName,
    data_type: AttrValueType,
    values: Vec<AttrValue>,
    null_mask: BitVec,  // Tracks null/missing values
    stats: ColumnStats, // Cached statistics
}
```

#### Memory Pool System

The memory pool system reduces allocation overhead by reusing buffers:

```rust
pub struct AttributeMemoryPool {
    // Specialized pools for common types
    string_pool: Vec<String>,
    float_pool: Vec<f64>,
    byte_pool: Vec<Vec<u8>>,
    
    // Reuse statistics
    reuse_count: usize,
    allocation_count: usize,
}
```

#### Operations

**Adding Nodes:**
```rust
impl Pool {
    pub fn add_node(&mut self, node_id: NodeId, attributes: HashMap<AttrName, AttrValue>) -> GraphResult<()> {
        // 1. Allocate entity slot
        let entity_index = self.nodes.allocate_slot(node_id)?;
        
        // 2. Store attributes in columnar format
        for (attr_name, attr_value) in attributes {
            self.node_attributes.set_value(entity_index, attr_name, attr_value)?;
        }
        
        // 3. Update indices
        self.node_index.insert(node_id, entity_index);
        
        // 4. Update statistics
        self.stats.increment_node_count();
        
        Ok(())
    }
}
```

**Adding Edges:**
```rust
impl Pool {
    pub fn add_edge(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId, attributes: HashMap<AttrName, AttrValue>) -> GraphResult<()> {
        // 1. Validate source and target nodes exist
        self.validate_node_exists(source)?;
        self.validate_node_exists(target)?;
        
        // 2. Allocate edge slot
        let entity_index = self.edges.allocate_slot(edge_id, source, target)?;
        
        // 3. Store edge attributes
        for (attr_name, attr_value) in attributes {
            self.edge_attributes.set_value(entity_index, attr_name, attr_value)?;
        }
        
        // 4. Update adjacency structures
        self.adjacency_cache.add_edge(source, target, edge_id);
        
        // 5. Update indices and statistics
        self.edge_index.insert(edge_id, entity_index);
        self.stats.increment_edge_count();
        
        Ok(())
    }
}
```

### Space Management (`src/core/space.rs`)

The Space subsystem manages active entity sets and provides workspace isolation, allowing multiple views of the same underlying data.

#### Key Responsibilities

1. **Active Set Management**: Track which nodes and edges are currently active
2. **Workspace Isolation**: Multiple spaces can have different active sets
3. **View Filtering**: Filter operations based on active entities
4. **Change Tracking**: Track modifications within a workspace

#### Core Data Structures

```rust
pub struct Space {
    // Active entity sets
    active_nodes: ActiveSet<NodeId>,
    active_edges: ActiveSet<EdgeId>,
    
    // Reference to the pool
    pool: Arc<RwLock<Pool>>,
    
    // Change tracking
    changes: ChangeTracker,
    
    // Space metadata
    space_id: SpaceId,
    created_at: SystemTime,
    config: SpaceConfig,
}

pub struct ActiveSet<T> {
    entities: HashSet<T>,
    inclusion_mode: InclusionMode, // Include or Exclude
    version: u64, // For invalidation
}
```

#### Operations

**Activating Entities:**
```rust
impl Space {
    pub fn activate_nodes(&mut self, node_ids: &[NodeId]) -> GraphResult<()> {
        for &node_id in node_ids {
            // Validate node exists in pool
            self.pool.read().unwrap().validate_node_exists(node_id)?;
            
            // Add to active set
            self.active_nodes.entities.insert(node_id);
        }
        
        // Increment version for cache invalidation
        self.active_nodes.version += 1;
        
        Ok(())
    }
    
    pub fn filter_nodes<F>(&self, predicate: F) -> Vec<NodeId>
    where
        F: Fn(NodeId, &HashMap<AttrName, AttrValue>) -> bool,
    {
        let pool = self.pool.read().unwrap();
        
        self.active_nodes.entities
            .iter()
            .filter(|&&node_id| {
                let attributes = pool.get_node_attributes(node_id).unwrap_or_default();
                predicate(node_id, &attributes)
            })
            .copied()
            .collect()
    }
}
```

### History Tracking (`src/core/history.rs`)

The History subsystem provides version control and state management for graphs, allowing users to save, restore, and track changes over time.

#### Key Responsibilities

1. **State Management**: Save and restore graph states
2. **Change Tracking**: Track all modifications to the graph
3. **Branch Management**: Support for multiple development branches
4. **Delta Compression**: Efficient storage of changes

#### Core Data Structures

```rust
pub struct History {
    // State storage
    states: HashMap<StateId, GraphState>,
    current_state: StateId,
    
    // Change tracking
    change_log: Vec<Change>,
    pending_changes: Vec<Change>,
    
    // Branch management
    branches: HashMap<BranchName, Branch>,
    current_branch: BranchName,
    
    // Configuration
    config: HistoryConfig,
}

pub struct GraphState {
    state_id: StateId,
    timestamp: SystemTime,
    parent_state: Option<StateId>,
    
    // State data (delta-compressed)
    node_changes: HashMap<NodeId, NodeChange>,
    edge_changes: HashMap<EdgeId, EdgeChange>,
    
    // Metadata
    message: String,
    author: String,
}
```

#### Operations

**Saving State:**
```rust
impl History {
    pub fn save_state(&mut self, message: &str) -> GraphResult<StateId> {
        // 1. Create state ID
        let state_id = self.generate_state_id();
        
        // 2. Collect pending changes
        let changes = self.pending_changes.drain(..).collect();
        
        // 3. Create delta from current state
        let delta = self.create_delta_from_changes(&changes)?;
        
        // 4. Store state
        let state = GraphState {
            state_id: state_id.clone(),
            timestamp: SystemTime::now(),
            parent_state: Some(self.current_state.clone()),
            node_changes: delta.node_changes,
            edge_changes: delta.edge_changes,
            message: message.to_string(),
            author: self.config.default_author.clone(),
        };
        
        self.states.insert(state_id.clone(), state);
        self.current_state = state_id.clone();
        
        Ok(state_id)
    }
}
```

## Storage Views

The storage view subsystem provides three unified interfaces for accessing graph data: Arrays, Matrices, and Tables.

### GraphArray (`src/core/array.rs`)

Single-column typed data with statistical operations.

```rust
pub struct GraphArray {
    values: Vec<AttrValue>,
    name: Option<String>,
    data_type: AttrValueType,
    cached_stats: RefCell<Option<ArrayStats>>,
}

impl GraphArray {
    pub fn mean(&self) -> Option<f64> {
        self.ensure_stats_cached();
        self.cached_stats.borrow().as_ref().map(|s| s.mean)
    }
    
    pub fn filter<F>(&self, predicate: F) -> GraphArray
    where
        F: Fn(&AttrValue) -> bool,
    {
        let filtered_values: Vec<AttrValue> = self.values
            .iter()
            .filter(|&v| predicate(v))
            .cloned()
            .collect();
            
        GraphArray::from_vec(filtered_values)
            .with_name(self.name.clone().unwrap_or_default())
    }
}
```

### GraphMatrix (`src/core/matrix.rs`)

Collection of homogeneous arrays with linear algebra operations.

```rust
pub struct GraphMatrix {
    columns: Vec<GraphArray>,
    dtype: AttrValueType,
    properties: MatrixProperties,
    cached_stats: RefCell<Option<MatrixStats>>,
}

impl GraphMatrix {
    pub fn sum_axis(&self, axis: Axis) -> GraphArray {
        match axis {
            Axis::Row => {
                // Sum each row across columns
                let mut row_sums = Vec::new();
                for row_idx in 0..self.rows() {
                    let sum = self.columns.iter()
                        .filter_map(|col| col.get(row_idx))
                        .filter_map(|val| val.as_f64())
                        .sum::<f64>();
                    row_sums.push(AttrValue::Float(sum));
                }
                GraphArray::from_vec(row_sums)
            }
            Axis::Column => {
                // Sum each column
                self.columns.iter()
                    .map(|col| col.sum().unwrap_or(0.0))
                    .map(AttrValue::Float)
                    .collect::<Vec<_>>()
                    .into()
            }
        }
    }
}
```

### GraphTable (`src/core/table.rs`)

Collection of heterogeneous arrays with relational operations.

```rust
pub struct GraphTable {
    columns: Vec<GraphArray>,
    column_names: Vec<String>,
    index: Option<GraphArray>,
    metadata: TableMetadata,
}

impl GraphTable {
    pub fn group_by(&self, column: &str) -> GroupBy {
        let col_idx = self.column_index(column)
            .expect("Column not found");
            
        let grouping_column = &self.columns[col_idx];
        let groups = self.create_groups(grouping_column);
        
        GroupBy::new(self, groups, column.to_string())
    }
    
    pub fn join(&self, other: &GraphTable, on: &str, how: JoinType) -> GraphResult<GraphTable> {
        let join_engine = JoinEngine::new(how);
        join_engine.execute(self, other, on)
    }
}
```

## Data Flow Architecture

### Adding a Node

```
User Request
     ↓
Python FFI Layer
     ↓
Graph API (api/graph.rs)
     ↓
Space Validation
     ↓
Pool Storage (core/pool.rs)
     ↓
Attribute Columns
     ↓
History Tracking
     ↓
Index Updates
     ↓
Cache Invalidation
```

### Querying Data

```
Query Request
     ↓
Storage View (Array/Matrix/Table)
     ↓
Space Filtering
     ↓
Pool Data Access
     ↓
Columnar Retrieval
     ↓
Statistical Computation
     ↓
Result Caching
     ↓
Response
```

## Memory Management

### Allocation Strategy

1. **Memory Pools**: Reuse buffers for common operations
2. **Columnar Layout**: Optimize cache locality
3. **Lazy Evaluation**: Compute results on demand
4. **Smart Caching**: Cache expensive computations with invalidation

### Reference Management

```rust
// Example of safe reference management
pub struct GraphHandle {
    pool: Arc<RwLock<Pool>>,
    space: Arc<RwLock<Space>>,
    history: Arc<RwLock<History>>,
}

impl GraphHandle {
    pub fn with_pool<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Pool) -> R,
    {
        let pool = self.pool.read().unwrap();
        f(&*pool)
    }
    
    pub fn with_pool_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Pool) -> R,
    {
        let mut pool = self.pool.write().unwrap();
        f(&mut *pool)
    }
}
```

## Performance Optimizations

### Columnar Storage Benefits

1. **Cache Efficiency**: Better memory locality for column operations
2. **Vectorization**: SIMD operations on homogeneous data
3. **Compression**: Better compression ratios for typed columns
4. **Statistics**: Fast computation of column-level statistics

### Lazy Evaluation

```rust
pub struct LazyStats {
    computed: RefCell<bool>,
    mean: RefCell<Option<f64>>,
    std: RefCell<Option<f64>>,
    min: RefCell<Option<AttrValue>>,
    max: RefCell<Option<AttrValue>>,
}

impl LazyStats {
    pub fn ensure_computed(&self, data: &[AttrValue]) {
        if !*self.computed.borrow() {
            self.compute_all(data);
            *self.computed.borrow_mut() = true;
        }
    }
}
```

### Index Optimization

```rust
pub struct ColumnIndex {
    // Bitmap index for exact matches
    exact_match_index: HashMap<AttrValue, BitVec>,
    
    // B-tree index for range queries
    range_index: BTreeMap<AttrValue, Vec<usize>>,
    
    // Bloom filter for existence checks
    bloom_filter: BloomFilter,
}
```

## Error Handling

Groggy uses a comprehensive error handling system:

```rust
#[derive(Debug, Clone)]
pub enum GraphError {
    // Pool errors
    NodeNotFound(NodeId),
    EdgeNotFound(EdgeId),
    AttributeNotFound(AttrName),
    
    // Type errors
    TypeMismatch { expected: AttrValueType, found: AttrValueType },
    InvalidConversion { from: AttrValueType, to: AttrValueType },
    
    // Space errors
    SpaceNotFound(SpaceId),
    InvalidSpace(String),
    
    // History errors
    StateNotFound(StateId),
    BranchNotFound(BranchName),
    
    // I/O errors
    IoError(String),
    SerializationError(String),
}
```

This architecture provides a solid foundation for high-performance graph processing while maintaining type safety, memory efficiency, and extensibility.