# Groggy Library Primer: Rust Graph Engine Fundamentals

## Table of Contents
1. [Overview & Philosophy](#overview--philosophy)
2. [Core Architecture](#core-architecture)
3. [Key Design Principles](#key-design-principles)
4. [Component Deep Dive](#component-deep-dive)
5. [Temporal Storage Strategy](#temporal-storage-strategy)
6. [Usage Patterns](#usage-patterns)
7. [Performance Characteristics](#performance-characteristics)
8. [Getting Started Examples](#getting-started-examples)

---

## Overview & Philosophy

Groggy is a **comprehensive graph language and processing engine** designed to handle any kind of graph workload - from static analysis to dynamic, evolving networks. It combines high-performance columnar storage with optional version control capabilities, making it suitable for everything from real-time graph processing to complex analytical workflows.

### Core Value Proposition
- **Universal Graph Processing**: Handle static graphs, dynamic networks, temporal data, and evolving structures
- **Columnar Storage**: High-performance bulk operations optimized for ML/analytics workloads  
- **Optional History**: Choose when to track changes - from stateless processing to full version control
- **Modular Architecture**: Use only the components you need for your specific use case
- **API Composability**: Consistent interface whether working with full graphs or filtered subgraphs

---

## Core Architecture

```text
                    ┌─────────────────────────────────────────┐
                    │              Graph (API)              │
                    │        Smart Coordinator & Facade      │
                    └─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┘
                      │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
        ┌─────────────┘ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
        │               │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
   ┌────▼────┐     ┌────▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─▼─┐
   │ Pool    │     │             Core Components            │
   │(Storage)│     └─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┘
   └─────────┘       │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                 ┌───▼─┐│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                 │Space││ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                 │(Act)│└─▼─┐ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                 └─────┘  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                  ┌───────▼─┐ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                  │ChangeT  │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
                  │(Delta)  │ └─▼─┐ │ │ │ │ │ │ │ │ │ │ │
                  └─────────┘   │ │ │ │ │ │ │ │ │ │ │ │ │
                      ┌─────────▼─┐ │ │ │ │ │ │ │ │ │ │ │
                      │ History   │ │ │ │ │ │ │ │ │ │ │ │
                      │(Git-like) │ └─▼─┐ │ │ │ │ │ │ │ │
                      └───────────┘   │ │ │ │ │ │ │ │ │ │
                          ┌───────────▼─┐ │ │ │ │ │ │ │ │
                          │ RefManager  │ │ │ │ │ │ │ │ │
                          │(Branch/Tag) │ └─▼─┐ │ │ │ │ │
                          └─────────────┘   │ │ │ │ │ │ │
                              ┌─────────────▼─┐ │ │ │ │ │
                              │ Query Engine  │ │ │ │ │ │
                              │(Analysis)     │ └─▼─┐ │ │
                              └───────────────┘   │ │ │ │
                                  ┌───────────────▼─┐ │ │
                                  │ Traversal       │ │ │
                                  │(Algorithms)     │ └─▼─┐
                                  └─────────────────┘   │ │
                                      ┌─────────────────▼─┐ │
                                      │ Subgraph          │ │
                                      │(Views)            │ └─▼─┐
                                      └───────────────────┘   │
                                          ┌───────────────────▼─┐
                                          │ Strategies          │
                                          │(Pluggable Storage)  │
                                          └─────────────────────┘
```

### Modular Component Architecture

**Graph as Master Coordinator**: The Graph struct is the single entry point that intelligently coordinates between all specialized components. Each core module contributes a specific capability:

1. **Storage Components**:
   - `GraphPool` - High-performance columnar data storage 
   - `GraphSpace` - Active state tracker and workspace management

2. **Processing Components**:
   - `QueryEngine` - Filtering, aggregation, and analytical operations
   - `TraversalEngine` - High-performance graph algorithms
   - `Subgraph` - Views with full Graph API inheritance for composable operations

3. **Optional Temporal Components** (use as needed):
   - `ChangeTracker` - Transaction management with pluggable strategies
   - `HistoryForest` - Version control system with branching capabilities
   - `RefManager` - Branch and tag management

4. **System Components**:
   - `DeltaObject` - Efficient change representation
   - `Strategies` - Pluggable storage algorithms for different workloads

---

## Key Design Principles

### 1. **Modular Design**
Components can be used independently based on your needs:
- **Static Analysis**: Use Pool + Space + QueryEngine for traditional graph processing
- **Dynamic Processing**: Add ChangeTracker for transaction management
- **Version Control**: Include History + RefManager for full temporal capabilities
- **API Composability**: Subgraphs inherit full Graph API for infinite composability

### 2. **High-Performance Columnar Storage**
Optimized for bulk operations and analytical workloads:
```rust
// Columnar attribute storage enables efficient bulk processing
pub struct AttributeColumn {
    values: Vec<AttrValue>,  // Cache-friendly sequential access
    next_index: usize,       // Append-only for performance
}
```

### 3. **Optional Index-Based Temporal Storage**
When history tracking is needed, uses space-efficient index references:
```rust
// Instead of: delta.node_attrs["name"] = "Alice" → "Bob"
// We store: delta.node_attrs["name"] = index_mapping(3 → 7)
// Where pool.node_attributes["name"].values[3] = "Alice"
//   and pool.node_attributes["name"].values[7] = "Bob"
```

### 4. **Strategy Pattern for Workload Optimization**
Pluggable algorithms adapt to different use cases:
```rust
pub trait TemporalStorageStrategy {
    fn record_node_addition(&mut self, node_id: NodeId);
    fn create_delta(&self) -> DeltaObject;
    // Choose strategy based on workload characteristics
}
```

---

## Component Deep Dive

### Graph: The Master Coordinator (api/graph.rs)
The Graph is the **smart coordinator** that manages all operations across components:

```rust
pub struct Graph {
    pool: GraphPool,           // Pure data storage
    space: GraphSpace,         // Active state tracker  
    change_tracker: ChangeTracker,  // Transaction management
    history: HistoryForest,    // Git-like version control
    ref_manager: RefManager,   // Branch/tag management
    query_engine: QueryEngine, // Analysis operations
    traversal_engine: TraversalEngine,  // Graph algorithms
    config: GraphConfig,       // Performance tuning
}
```

**Key Responsibilities:**
- Single entry point for all graph operations
- Coordinate between specialized components intelligently
- Manage transactional boundaries and consistency
- Handle ID generation and entity lifecycle
- Optimize cross-component operations

### GraphPool: Pure Data Storage (core/pool.rs)
The "database" that stores ALL data with no business logic:

```rust
pub struct GraphPool {
    next_node_id: NodeId,
    next_edge_id: EdgeId,
    edges: HashMap<EdgeId, (NodeId, NodeId)>,
    // Columnar attribute storage for cache efficiency
    node_attributes: HashMap<AttrName, AttributeColumn>,
    edge_attributes: HashMap<AttrName, AttributeColumn>,
}

pub struct AttributeColumn {
    values: Vec<AttrValue>,  // Append-only for performance!
    next_index: usize,
}
```

**Architecture Role:**
- Pure storage with no concept of "active" vs "inactive"
- Grows indefinitely (append-only for performance)
- Can store soft-deleted entities
- Optimized for bulk operations and analytics workloads

**Key Operations:**
- `add_node()` → Generate new NodeId
- `set_attr()` → Append to column, return index
- `set_bulk_attrs()` → Efficient batch operations

### GraphSpace: Active State Tracker (core/space.rs)
Minimal responsibility - just tracks what's currently "active":

```rust
pub struct GraphSpace {
    active_nodes: HashSet<NodeId>,
    active_edges: HashSet<EdgeId>, 
    // Maps entities to their current attribute indices in Pool
    node_attribute_indices: HashMap<NodeId, HashMap<AttrName, usize>>,
    edge_attribute_indices: HashMap<EdgeId, HashMap<AttrName, usize>>,
    base_state: StateId,  // What state this workspace is built on
}
```

**Architecture Role:**
- Knows which entities are currently "active"
- Manages the active subset of pool data
- Provides the "current view" of the graph
- Minimal and focused (change tracking moved to Graph level)

**Key Operations:**
- `contains_node()` → Check if node exists in current view
- `get_attr_index()` → Find current attribute value index
- `set_attr_index()` → Update current mapping

### ChangeTracker: Strategy-Based Transactions (core/change_tracker.rs)
Pluggable transaction management with different temporal storage strategies:

```rust
pub struct ChangeTracker {
    strategy: Box<dyn TemporalStorageStrategy>,
}

pub trait TemporalStorageStrategy {
    fn record_node_addition(&mut self, node_id: NodeId);
    fn record_attr_change(&mut self, entity_id: u64, attr: AttrName, 
                         old_index: Option<usize>, new_index: usize);
    fn create_delta(&self) -> DeltaObject;
    // ... other operations
}
```

**Architecture Role:**
- Strategy Pattern for pluggable storage algorithms
- Delegates to selected strategy while maintaining consistent API
- Performance transparency - each strategy optimizes for different workloads
- Configuration driven strategy selection

**Current Strategies:**
- **Index Deltas**: Column indices (current default)
- **Future**: Full snapshots, hybrid, compressed storage

### HistoryForest: Git-Like Version Control (core/history.rs) 
Immutable version control backbone with branching and merging:

```rust
pub struct HistoryForest {
    states: HashMap<StateId, Arc<StateObject>>,
    children: HashMap<StateId, Vec<StateId>>, // For traversal
    deltas: HashMap<[u8; 32], Arc<DeltaObject>>, // Content addressing
}

pub struct StateObject {
    parent: Option<StateId>,
    delta: Arc<DeltaObject>,
    metadata: Arc<StateMetadata>,
}
```

**Architecture Role:**
- Immutable snapshots (states never modified after creation)
- Content-addressed storage (deduplication via hashing)
- Git-like branching model with merge capabilities
- Efficient diff-based storage (only store what changed)

**Key Operations:**
- `add_state()` → Record new state with delta
- `get_path_to_root()` → Traverse history chain  
- `view_at_state()` → Create historical view

### RefManager: Branch and Tag Management (core/ref_manager.rs)
Git-like reference system for organizing history:

```rust
pub struct RefManager {
    branches: HashMap<BranchName, Branch>,
    current_branch: BranchName,
    tags: HashMap<String, StateId>,
}

pub struct Branch {
    name: BranchName,
    head: StateId,        // Current tip of branch
    created_at: u64,      // Creation timestamp  
    created_by: String,   // Author
    description: Option<String>,
}
```

**Architecture Role:**
- Lightweight references (just pointers to states)
- Git-like workflow (branch, merge, tag operations)
- Metadata-rich references with creation info
- Concurrent-safe operations

### QueryEngine: Analysis Operations (core/query.rs)
Core filtering operations for graph data:

```rust
pub struct QueryEngine {}

pub enum NodeFilter {
    AttributeFilter { name: AttrName, filter: AttributeFilter },
    // ... other filters
}
```

**Architecture Role:**
- Read-only analysis and filtering operations
- Leverages columnar storage for performance
- Supports complex attribute-based queries
- Works with active space information

### TraversalEngine: High-Performance Algorithms (core/traversal.rs)
Graph algorithms optimized for columnar storage:

```rust
pub struct TraversalEngine {
    state_pool: TraversalStatePool,     // Reusable state
    config: TraversalConfig,            // Performance tuning
    stats: TraversalStats,              // Metrics tracking
    adjacency_cache: AdjacencyCache,    // Fast neighbor lookups
}
```

**Architecture Role:**
- Performance-first with columnar topology access
- Memory-efficient with reusable data structures  
- Modular algorithms implementing common traits
- Configurable with filtering and constraints

### Subgraph: Views with Full API (core/subgraph.rs)
Subset of nodes/edges with complete Graph API inheritance:

```rust
pub struct Subgraph {
    graph: Rc<RefCell<Graph>>,  // Reference to parent
    nodes: HashSet<NodeId>,     // Subset of nodes
    edges: HashSet<EdgeId>,     // Induced edges
    subgraph_type: String,      // Creation metadata
}
```

**Architecture Role:**  
- Subgraph IS-A Graph through delegation
- All Graph operations work: filter, traverse, analyze
- Infinite composability: `subgraph.filter().bfs().filter()`
- Column access: `subgraph[attr_name] -> Vec<AttrValue>`

### DeltaObject: Change Representation (core/delta.rs)
Efficient representation of changes between states:

```rust
pub struct DeltaObject {
    node_attrs: HashMap<AttrName, ColumnIndexDelta>,
    edge_attrs: HashMap<AttrName, ColumnIndexDelta>, 
    nodes_added: Vec<NodeId>,
    nodes_removed: Vec<NodeId>,
    edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    edges_removed: Vec<EdgeId>,
}

pub struct ColumnIndexDelta {
    // Entity ID → (old_index, new_index) mappings
    index_changes: HashMap<u64, (Option<usize>, usize)>,
}
```

**Architecture Role:**
- Sparse representation (only store what changed)
- Columnar layout for bulk operations and cache efficiency
- Content-addressed for automatic deduplication  
- Immutable design for safe sharing

### Strategies: Pluggable Storage (core/strategies.rs)
Different temporal storage approaches for different workloads:

```rust
pub trait TemporalStorageStrategy {
    fn record_node_addition(&mut self, node_id: NodeId);
    fn record_attr_change(&mut self, ...);
    fn create_delta(&self) -> DeltaObject;
}

pub struct IndexDeltaStrategy { /* Current implementation */ }
// Future: FullSnapshotStrategy, HybridStrategy, etc.
```

**Architecture Role:**
- Strategy pattern for runtime algorithm selection
- Different optimizations for different workloads:
  - **Index Deltas**: Best for frequent small changes
  - **Full Snapshots**: Best for time-travel heavy workloads  
  - **Hybrid**: Balanced for mixed access patterns

---

## Temporal Storage Strategy

### Index-Based Delta System

The core innovation is storing deltas as index mappings rather than value copies:

```rust
#[derive(Debug, Clone)]
pub struct ColumnIndexDelta {
    // Entity ID → (old_index, new_index) mappings
    index_changes: HashMap<u64, (Option<usize>, usize)>,
}

#[derive(Debug, Clone)]
pub struct DeltaObject {
    node_attrs: HashMap<AttrName, ColumnIndexDelta>,
    edge_attrs: HashMap<AttrName, ColumnIndexDelta>,
    nodes_added: Vec<NodeId>,
    nodes_removed: Vec<NodeId>,
    edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    edges_removed: Vec<EdgeId>,
}
```

### Example Flow

1. **Initial State**: Node 1 has name="Alice" (stored at index 0 in name column)
2. **Change**: Update to name="Bob"
   - Pool: Append "Bob" to name column → gets index 1
   - Space: Update mapping node1.name: 0 → 1
   - ChangeTracker: Record (node1, "name", Some(0), 1)
3. **Commit**: Create delta with index mapping 0 → 1
4. **Historical View**: To see old state, use index 0; current state uses index 1

### Benefits

- **Space Efficient**: Deltas are tiny (just index mappings)
- **Fast Reconstruction**: No need to replay all changes
- **Deduplication**: Identical values share indices
- **Bulk Operations**: Columnar storage enables vectorized operations

---

## Usage Patterns

### Static Graph Processing
```rust
// Simple graph analysis without version control
let mut graph = Graph::new();

// Build and analyze graph efficiently
let alice = graph.add_node();
let bob = graph.add_node();
let charlie = graph.add_node();

graph.add_edge(alice, bob).unwrap();
graph.add_edge(bob, charlie).unwrap();

// High-performance attribute operations using columnar storage
graph.set_node_attr(alice, "name", AttrValue::Text("Alice".into())).unwrap();
graph.set_node_attr(bob, "department", AttrValue::Text("Engineering".into())).unwrap();

// Efficient filtering and analysis
let engineers = graph.find_nodes(&NodeFilter::AttributeFilter {
    name: "department".to_string(),
    filter: AttributeFilter::Equals(AttrValue::Text("Engineering".into()))
}).unwrap();
```

### Dynamic Graph Processing
```rust
// Real-time graph updates with optional change tracking
let mut graph = Graph::with_change_tracking();

// Process streaming updates efficiently
for update in stream_of_updates {
    match update {
        GraphUpdate::AddNode(attrs) => {
            let node = graph.add_node();
            for (attr, value) in attrs {
                graph.set_node_attr(node, attr, value)?;
            }
        }
        GraphUpdate::UpdateAttribute(node, attr, value) => {
            graph.set_node_attr(node, attr, value)?;
        }
    }
    
    // Optional: create checkpoints for important states
    if update.is_milestone() {
        graph.commit("Milestone reached", "system")?;
    }
}
```

### Bulk Operations (Columnar Efficiency)
```rust
// Leverage columnar storage for analytical workloads
let node_ids = graph.add_nodes(100_000);

// Efficient bulk attribute setting
let mut bulk_attrs = HashMap::new();
bulk_attrs.insert("initialized".to_string(), 
    node_ids.iter().map(|&id| (id, AttrValue::Bool(true))).collect()
);
bulk_attrs.insert("batch_id".to_string(),
    node_ids.iter().enumerate().map(|(i, &id)| (id, AttrValue::Int(i as i64 / 1000))).collect()
);

graph.set_node_attrs(bulk_attrs)?;

// Bulk querying with columnar optimizations
let active_nodes = graph.find_nodes(&NodeFilter::AttributeFilter {
    name: "initialized".to_string(),
    filter: AttributeFilter::Equals(AttrValue::Bool(true))
})?;
```

### Advanced Analysis with Subgraphs
```rust
// Create composable subgraph views
let engineers = graph.filter_nodes(&HashMap::from([
    ("department".to_string(), AttributeFilter::Equals(AttrValue::Text("Engineering".into())))
]))?;

// Chain operations infinitely - subgraphs have full Graph API
let senior_active_engineers = engineers
    .filter_nodes_by_attribute("level", &AttrValue::Text("Senior".into()))?
    .filter_nodes_by_attribute("status", &AttrValue::Text("Active".into()))?;

// Run graph algorithms on subgraphs
let components = senior_active_engineers.connected_components()?;
let central_nodes = senior_active_engineers.compute_centrality(CentralityType::Betweenness)?;

// Column-wise attribute access for analytics
let salaries = senior_active_engineers.get_node_attribute_column("salary")?;
let avg_salary: f64 = salaries.iter()
    .filter_map(|v| if let AttrValue::Float(f) = v { Some(*f as f64) } else { None })
    .sum::<f64>() / salaries.len() as f64;
```

### Optional Version Control (When Needed)
```rust
// Enable full version control for complex workflows
let mut graph = Graph::with_version_control();

// Standard graph operations
let alice = graph.add_node();
graph.set_node_attr(alice, "role", AttrValue::Text("Lead".into()))?;

// Create versioned snapshots when needed
let v1 = graph.commit("Initial team structure", "manager")?;

// Branch for experimentation
graph.create_branch("reorganization")?;
graph.checkout_branch("reorganization")?;

// Make experimental changes
graph.set_node_attr(alice, "role", AttrValue::Text("Architect".into()))?;
let experimental_commit = graph.commit("Test new roles", "manager")?;

// Time travel to compare states
let original_view = graph.view_at_commit(v1)?;
let experimental_view = graph.view_at_commit(experimental_commit)?;

// Merge successful experiments
graph.checkout_branch("main")?;
graph.merge_branch("reorganization", "manager")?;
```

### Workload-Specific Strategies
```rust
// Configure storage strategy based on workload
let graph = match workload_type {
    WorkloadType::FrequentSmallChanges => {
        Graph::with_strategy(Box::new(IndexDeltaStrategy::new()))
    },
    WorkloadType::InfrequentLargeChanges => {
        Graph::with_strategy(Box::new(SnapshotStrategy::new()))
    },
    WorkloadType::MixedAccess => {
        Graph::with_strategy(Box::new(HybridStrategy::new()))
    }
};

// Strategy is transparent - same API regardless of storage approach
graph.add_node();
graph.set_node_attr(node, "data", value)?;
```

---

## Performance Characteristics

### Time Complexity
- **Node/Edge Operations**: O(1) amortized
- **Attribute Access**: O(1) with direct indexing
- **History Operations**: O(log n) with content addressing
- **Query Operations**: O(n) with columnar optimizations
- **Bulk Operations**: O(k) for k items (vectorized)

### Space Complexity
- **Current State**: O(nodes + edges + unique_attributes)
- **History**: O(total_changes) with delta compression
- **Memory Usage**: Configurable limits with garbage collection

### Optimizations
- **Content Addressing**: Automatic deduplication of identical changes
- **Columnar Storage**: Cache-friendly bulk operations
- **Index-Based Deltas**: Minimal storage overhead
- **Zero-Copy Views**: Historical access without materialization
- **Configurable Snapshots**: Balance between space and access speed

---

## Getting Started Examples

### Simple Network Analysis
```rust
use groggy::{Graph, AttrValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::new();
    
    // Create social network
    let alice = graph.add_node();
    let bob = graph.add_node();
    let charlie = graph.add_node();
    
    graph.add_edge(alice, bob)?;
    graph.add_edge(bob, charlie)?;
    
    // Add attributes
    graph.set_node_attr(alice, "name", AttrValue::Text("Alice".into()))?;
    graph.set_node_attr(alice, "age", AttrValue::Int(28))?;
    
    // Save checkpoint
    let v1 = graph.commit("Initial network".into(), "analyst".into())?;
    
    // Analyze
    let degree_dist = graph.degree_distribution()?;
    println!("Degree distribution: {:?}", degree_dist);
    
    Ok(())
}
```

### ML Pipeline with History
```rust
use groggy::{Graph, AttrValue, GraphConfig};

fn ml_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let config = GraphConfig::performance_optimized();
    let mut graph = Graph::with_config(config);
    
    // Load data
    let nodes = graph.add_nodes(10000);
    
    // Feature engineering branch  
    graph.create_branch("feature_engineering")?;
    
    // Bulk feature computation
    let features: Vec<_> = nodes.iter().map(|&id| {
        (id, AttrValue::Vec(compute_features(id)))
    }).collect();
    
    graph.set_node_attrs(HashMap::from([
        ("features".to_string(), features)
    ]))?;
    
    let features_commit = graph.commit("Add ML features", "ml_team")?;
    
    // Model training branch
    graph.create_branch("model_v1")?;
    // ... training logic ...
    
    // Merge successful experiment
    graph.checkout_branch("main")?;
    graph.merge_branch("model_v1", "ml_team")?;
    
    Ok(())
}

fn compute_features(node_id: u64) -> Vec<f32> {
    // Placeholder feature computation
    vec![node_id as f32, (node_id * 2) as f32]
}
```

### Historical Analysis
```rust
use groggy::{Graph, AttrValue};

fn analyze_evolution() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Graph::new();
    
    // Build initial network
    let alice = graph.add_node();
    graph.set_node_attr(alice, "influence", AttrValue::Float(0.5))?;
    let checkpoint1 = graph.commit("Day 1", "system")?;
    
    // Simulate growth over time
    for day in 2..=30 {
        let new_node = graph.add_node();
        graph.add_edge(alice, new_node)?;
        
        // Update influence
        let old_influence = graph.get_node_attr(alice, "influence")?;
        if let AttrValue::Float(val) = old_influence {
            graph.set_node_attr(alice, "influence", AttrValue::Float(val * 1.1))?;
        }
        
        graph.commit(format!("Day {}", day), "system")?;
    }
    
    // Analyze historical trend
    let states = graph.list_states();
    for state in states {
        let view = graph.view_at_state(state)?;
        let influence = view.get_node_attr(alice, "influence")?;
        println!("State {}: Alice influence = {:?}", state, influence);
    }
    
    Ok(())
}
```

---

## Next Steps

1. **Read the full API documentation** for detailed method signatures
2. **Explore the test files** for comprehensive usage examples
3. **Check the configuration options** in `GraphConfig` for performance tuning
4. **Review the error types** in `GraphError` for proper error handling
5. **Experiment with different strategies** via the `TemporalStorageStrategy` trait

## Key Takeaways

- **Universal Graph Engine**: Handles any kind of graph workload - static analysis, dynamic processing, temporal data, and evolving networks
- **Modular Architecture**: Use only the components you need - from simple static processing to full version control
- **Columnar Performance**: High-performance bulk operations optimized for analytical and ML workloads
- **API Composability**: Consistent interface whether working with full graphs or filtered subgraphs enables infinite composability
- **Optional History**: Choose when to track changes - version control is a feature, not a requirement
- **Strategy Pattern**: Pluggable algorithms adapt to different workload characteristics and performance needs
- **Component Coordination**: Graph struct intelligently coordinates specialized components rather than simple delegation
- **Workload Optimization**: Different storage strategies (index deltas, snapshots, hybrid) for different use cases

---

*For more information, see the full documentation and examples in the repository.*
