# Groggy Implementation Plan

## High-Level Implementation Phases

### Phase 1: Fix Compilation ✅
**Goal**: Get the project to compile without errors
- [x] Fix doc comment syntax (`//!` → `//`)
- [x] Add missing imports and module declarations
- [x] Implement `Hash` trait for `AttrValue`
- [x] Complete basic type accessor methods
- **Milestone**: Core compilation issues resolved (only query engine types remain)

### Phase 2: Basic Storage ✅  
**Goal**: Working data storage without versioning
- [x] Implement `GraphPool` (columnar storage)
- [x] Implement `GraphSpace` (active state management)
- [x] Basic `Graph` API (add nodes/edges, attributes)
- **Milestone**: Core graph operations working (add_node, add_edge, set/get attributes)

### Phase 3: Change Tracking ✅
**Goal**: Transaction support for commits  
- [x] Implement `DeltaObject` (change representation)
- [x] Implement `ChangeTracker` (record modifications)  
- [x] Add `Graph::commit()` functionality
- **Milestone**: Working transaction system with commit/rollback support

### Phase 4: Version Control ✅
**Goal**: Git-like branching and history
- [x] Implement `HistoryForest` and commit storage
- [x] Implement basic branching (`create_branch`, `list_branches`) 
- [x] Complete core version control operations
- **Milestone**: Working Git-like commit and branch functionality

### Phase 5: Query Engine ✅
**Goal**: Filtering and analysis capabilities
- [x] Implement `QueryEngine` core filtering (AttributeFilter::matches)
- [x] Basic node/edge filtering by attributes 
- [x] Complex filter combinations (And, Or, Not)
- [x] Vector similarity operations (cosine, euclidean, etc.)
- [x] Graph API integration (`find_nodes`, `find_edges`)
- [ ] Structural filters (neighbors, degree, patterns) - TODO
- [ ] Advanced aggregation operations - TODO
- [ ] Query optimization and caching - TODO
- **Milestone**: Core filtering infrastructure working and integrated

### Phase 6: Production Readiness ❌
**Goal**: Make the library production-ready

#### 6a. Fix Critical Compilation Issues (Priority 1)
- [ ] Fix missing HashMap imports in change_tracker.rs
- [ ] Fix missing GraphPool/GraphError imports
- [ ] Fix lib.rs example (missing AttributeFilter import)
- [ ] Implement remaining placeholder methods returning `()` instead of proper types
- **Goal**: Get `cargo test` working

#### 6b. Core Functionality Gaps (Priority 2)  
- [ ] Implement GraphSpace-Pool integration (proper attribute index mapping)
- [ ] Implement ChangeTracker-Pool integration (convert indices to values)
- [ ] Complete HistoryForest commit operations (Delta::from_changes)
- [ ] Add proper error handling throughout
- **Goal**: Core graph operations fully functional

#### 6c. Production Features (Priority 3)
- [ ] Add comprehensive test coverage
- [ ] Implement structural graph queries (neighbors, patterns, etc.)
- [ ] Performance optimization (query caching, bulk operations)
- [ ] Documentation and API examples
- [ ] Python bindings integration
- **Milestone**: Complete API matching README examples

---

## Current Status: Phase 5 Complete (Core Query Engine) - Moving to Phase 6 (Production)

## Overview
This plan implements a working graph library incrementally, ensuring each step compiles and adds functionality. We follow the dependency chain revealed by `cargo check` to build from foundation up.

---

## Step 1: Fix Immediate Compilation Issues

### 1.1 Fix Documentation Comments (`src/api/graph.rs`)
**Problem**: Misplaced `//!` inner doc comments causing E0753 errors
**Action**: Convert `//!` to `//` for comments that aren't at module level
- Lines 4-14: Change `//!` to `//` 
- Keep the first `//!` at line 1 (module-level doc is correct)

### 1.2 Add Missing Imports
**Problem**: Referenced types not in scope
**Action**: Add imports to make compilation work
- `src/api/graph.rs`: Add imports for `GraphPool`, `HistoryForest`, `QueryEngine`, etc.
- `src/lib.rs`: Ensure all module declarations are correct
- Add `use std::path::Path;` where needed for file operations

### 1.3 Implement Hash for AttrValue (`src/types.rs`)
**Problem**: `Hash` trait not implemented (line 61-64)
**Action**: Complete the Hash implementation
```rust
impl Hash for AttrValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttrValue::Float(f) => {
                0u8.hash(state);
                f.to_bits().hash(state);
            }
            AttrValue::Int(i) => {
                1u8.hash(state);
                i.hash(state);
            }
            AttrValue::Text(s) => {
                2u8.hash(state);
                s.hash(state);
            }
            AttrValue::FloatVec(v) => {
                3u8.hash(state);
                v.len().hash(state);
                for f in v {
                    f.to_bits().hash(state);
                }
            }
            AttrValue::Bool(b) => {
                4u8.hash(state);
                b.hash(state);
            }
        }
    }
}
```

**Success Criteria**: `cargo check` shows significantly fewer errors

---

## Step 2: Complete Foundation Types (`src/types.rs`)

### 2.1 Complete AttrValue Accessor Methods
**Problem**: Methods like `as_float()` are stubs (lines 67-92)
**Action**: Implement all accessor methods
```rust
impl AttrValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            AttrValue::Float(_) => "Float",
            AttrValue::Int(_) => "Int", 
            AttrValue::Text(_) => "Text",
            AttrValue::FloatVec(_) => "FloatVec",
            AttrValue::Bool(_) => "Bool",
        }
    }
    
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttrValue::Float(f) => Some(*f),
            _ => None,
        }
    }
    
    // Continue for as_int(), as_text(), as_float_vec(), as_bool()
}
```

### 2.2 Add Missing Type Aliases
**Action**: Ensure all referenced types exist or add them to `types.rs`

**Success Criteria**: All types in `src/types.rs` compile and have complete implementations

---

## Step 3: Basic Storage Implementation (`src/core/pool.rs`)

### 3.1 Define AttributeColumn Structure
**Problem**: `AttributeColumn` referenced but not defined
**Action**: Add before GraphPool struct
```rust
#[derive(Debug, Clone)]
pub struct AttributeColumn {
    pub values: Vec<AttrValue>,
}

impl AttributeColumn {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }
    
    pub fn push(&mut self, value: AttrValue) -> usize {
        let index = self.values.len();
        self.values.push(value);
        index
    }
    
    pub fn get(&self, index: usize) -> Option<&AttrValue> {
        self.values.get(index)
    }
}
```

### 3.2 Implement GraphPool Constructor
**Problem**: `GraphPool::new()` is a todo (line 97)
**Action**: Complete the constructor
```rust
impl GraphPool {
    pub fn new() -> Self {
        Self {
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            topology: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
        }
    }
}
```

### 3.3 Implement Basic Entity Creation
**Action**: Implement `add_node()` and `add_edge()` methods
```rust
impl GraphPool {
    pub fn add_node(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }
    
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> EdgeId {
        let id = self.next_edge_id;
        self.topology.insert(id, (source, target));
        self.next_edge_id += 1;
        id
    }
}
```

### 3.4 Implement Basic Attribute Operations
**Action**: Add methods for setting/getting attributes
```rust
impl GraphPool {
    pub fn set_node_attr(&mut self, attr_name: AttrName, value: AttrValue) -> usize {
        let column = self.node_attributes.entry(attr_name).or_insert_with(AttributeColumn::new);
        column.push(value)
    }
    
    pub fn get_node_attr_at_index(&self, attr_name: &AttrName, index: usize) -> Option<&AttrValue> {
        self.node_attributes.get(attr_name).and_then(|col| col.get(index))
    }
    
    // Similar methods for edges
}
```

**Success Criteria**: GraphPool compiles and provides basic storage functionality

---

## Step 4: Active State Management (`src/core/space.rs`)

### 4.1 Complete GraphSpace Constructor  
**Problem**: Methods are todos starting at line 106
**Action**: Implement real functionality
```rust
impl GraphSpace {
    pub fn new(base_state: StateId) -> Self {
        Self {
            active_nodes: HashSet::new(),
            active_edges: HashSet::new(),
            node_attribute_indices: HashMap::new(),
            edge_attribute_indices: HashMap::new(),
            base_state,
        }
    }
}
```

### 4.2 Implement Basic Active Set Operations
**Action**: Replace todos with real implementations
```rust
impl GraphSpace {
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.active_nodes.iter().copied().collect()
    }
    
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.active_edges.iter().copied().collect()
    }
    
    pub fn get_base_state(&self) -> StateId {
        self.base_state
    }
}
```

### 4.3 Fix Generic Type Issues
**Problem**: `set_attr_index` and `get_attr_index` use generic `T: Into<u64>` but nodes/edges are `usize`
**Action**: Simplify to use concrete types
```rust
pub fn set_node_attr_index(&mut self, node_id: NodeId, attr_name: AttrName, new_index: usize) {
    self.node_attribute_indices
        .entry(node_id)
        .or_insert_with(HashMap::new)
        .insert(attr_name, new_index);
}

pub fn get_node_attr_index(&self, node_id: NodeId, attr_name: &AttrName) -> Option<usize> {
    self.node_attribute_indices
        .get(&node_id)
        .and_then(|attrs| attrs.get(attr_name))
        .copied()
}
```

**Success Criteria**: GraphSpace compiles and provides active set management

---

## Step 5: Minimal Graph API (`src/api/graph.rs`)

### 5.1 Add Required Imports and Fix Module Issues
**Action**: Add all missing imports at top of file
```rust
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::change_tracker::ChangeTracker;
use crate::config::GraphConfig;
use crate::types::*;
use crate::errors::*;
use std::path::Path;
use std::collections::HashMap;
```

### 5.2 Temporarily Stub Complex Dependencies
**Action**: Create minimal stubs for components not yet implemented
```rust
// Temporary stubs - will replace in later steps
#[derive(Debug)]
struct HistoryForest;
impl HistoryForest {
    fn new() -> Self { Self }
}

#[derive(Debug)] 
struct QueryEngine;
impl QueryEngine {
    fn new() -> Self { Self }
}
```

### 5.3 Implement Graph Constructor
**Problem**: `Graph::new()` is todo (line 99)
**Action**: Initialize all components
```rust
impl Graph {
    pub fn new() -> Self {
        let config = GraphConfig::new();
        Self {
            pool: GraphPool::new(),
            history: HistoryForest::new(),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            space: GraphSpace::new(0),
            change_tracker: ChangeTracker::new(),
            config,
        }
    }
}
```

### 5.4 Implement Basic Graph Operations
**Action**: Replace todos with real coordination between pool and space
```rust
impl Graph {
    pub fn add_node(&mut self) -> NodeId {
        let node_id = self.pool.add_node();
        self.space.activate_node(node_id);
        node_id
    }
    
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> Result<EdgeId, GraphError> {
        if !self.space.contains_node(source) || !self.space.contains_node(target) {
            return Err(GraphError::node_not_found(source, "add edge"));
        }
        
        let edge_id = self.pool.add_edge(source, target);
        self.space.activate_edge(edge_id, source, target);
        Ok(edge_id)
    }
    
    pub fn contains_node(&self, node: NodeId) -> bool {
        self.space.contains_node(node)
    }
}
```

### 5.5 Implement Basic Attribute Operations  
**Action**: Coordinate between pool (storage) and space (indexing)
```rust
impl Graph {
    pub fn set_node_attr(&mut self, node: NodeId, attr: AttrName, value: AttrValue) -> Result<(), GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "set attribute"));
        }
        
        let new_index = self.pool.set_node_attr(attr.clone(), value);
        self.space.set_node_attr_index(node, attr, new_index);
        Ok(())
    }
    
    pub fn get_node_attr(&self, node: NodeId, attr: &AttrName) -> Result<Option<AttrValue>, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "get attribute"));
        }
        
        if let Some(index) = self.space.get_node_attr_index(node, attr) {
            Ok(self.pool.get_node_attr_at_index(attr, index).cloned())
        } else {
            Ok(None)
        }
    }
}
```

**Success Criteria**: Basic graph operations work - can create nodes, edges, and set/get attributes

---

## Step 6: Change Tracking (Minimal for Commits)

### 6.1 Implement Basic DeltaObject (`src/core/delta.rs`)
**Action**: Replace complex structure with minimal working version
```rust
#[derive(Debug, Clone)]
pub struct DeltaObject {
    pub nodes_added: Vec<NodeId>,
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    pub content_hash: [u8; 32],
}

impl DeltaObject {
    pub fn empty() -> Self {
        Self {
            nodes_added: Vec::new(),
            edges_added: Vec::new(), 
            content_hash: [0; 32],
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.nodes_added.is_empty() && self.edges_added.is_empty()
    }
    
    pub fn change_count(&self) -> usize {
        self.nodes_added.len() + self.edges_added.len()
    }
}
```

### 6.2 Implement Basic ChangeTracker
**Action**: Minimal change tracking for commits
```rust
// Stub the strategy pattern for now
impl ChangeTracker {
    pub fn new() -> Self {
        Self {
            nodes_added: Vec::new(),
            edges_added: Vec::new(),
        }
    }
    
    pub fn record_node_addition(&mut self, node_id: NodeId) {
        self.nodes_added.push(node_id);
    }
    
    pub fn record_edge_addition(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.edges_added.push((edge_id, source, target));
    }
    
    pub fn create_delta(&self) -> DeltaObject {
        DeltaObject {
            nodes_added: self.nodes_added.clone(),
            edges_added: self.edges_added.clone(),
            content_hash: [0; 32], // TODO: real hash
        }
    }
    
    pub fn clear(&mut self) {
        self.nodes_added.clear();
        self.edges_added.clear();
    }
}
```

### 6.3 Integrate Change Tracking in Graph
**Action**: Update Graph operations to record changes
```rust
impl Graph {
    pub fn add_node(&mut self) -> NodeId {
        let node_id = self.pool.add_node();
        self.space.activate_node(node_id);
        self.change_tracker.record_node_addition(node_id);
        node_id
    }
    
    pub fn commit(&mut self, message: String, author: String) -> Result<StateId, GraphError> {
        let delta = self.change_tracker.create_delta();
        if delta.is_empty() {
            return Err(GraphError::NoChangesToCommit);
        }
        
        // For now, just increment commit ID
        self.current_commit += 1;
        self.change_tracker.clear();
        
        Ok(self.current_commit)
    }
}
```

**Success Criteria**: Can track changes and create basic commits

---

## Step 7: Complete Version Control (Final Step)

### 7.1 Implement StateObject and History Storage
### 7.2 Implement Branching Operations
### 7.3 Complete All Remaining TODOs

---

## Testing Strategy

After each step:
1. `cargo check` - should compile without errors
2. `cargo test` - run any existing tests  
3. Manual verification - test basic functionality works

## Success Criteria

**Step 1-2**: Project compiles without errors
**Step 3-4**: Basic storage and state management works
**Step 5**: Can create graphs, add nodes/edges, set attributes
**Step 6**: Can commit changes and track history
**Step 7**: Full version control functionality

Each step builds incrementally on the previous, ensuring we always have a working system.