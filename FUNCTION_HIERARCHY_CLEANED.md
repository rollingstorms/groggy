# Groggy Function Hierarchy - Clean API Structure

*Generated after duplicate function cleanup*

## 📊 Summary

- **Total Functions**: 366
- **Total Structs**: 59
- **Total Traits**: 1
- **TODO Functions**: 218
- **Implementation Rate**: 40.4%

## 🎯 Core API (graph.rs)

**Functions**: 44 total, 43 public, 43 TODO
**Structs**: 4, **Traits**: 0

**Key Types:**
- `Graph`
- `GraphStatistics`
- `BranchInfo`
- ... and 1 more

**Public API:**

*Construction:*
- `new() -> Self ` 🚧 TODO
- `with_config(config: GraphConfig) -> Self ` 🚧 TODO
- `load_from_path(path: &Path) -> Result<Self, GraphError> ` 🚧 TODO

*Entity Creation:*
- `add_node(&mut self) -> NodeId ` 🚧 TODO
- `add_nodes(&mut self, count: usize) -> Vec<NodeId> ` 🚧 TODO
- `add_edge(&mut self, source: NodeId, target: NodeId) -> Result<EdgeId, GraphError> ` 🚧 TODO
- `add_edges(&mut self, edges: &[(NodeId, NodeId)` 🚧 TODO

*Entity Removal:*
- `remove_node(&mut self, node: NodeId) -> Result<(), GraphError> ` 🚧 TODO
- `remove_edge(&mut self, edge: EdgeId) -> Result<(), GraphError> ` 🚧 TODO

*Attribute Setting:*
- `set_node_attr(&mut self, node: NodeId, attr: AttrName, value: AttrValue) -> Result<(), GraphError> ` 🚧 TODO
- `set_node_attrs(&mut self, attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)` 🚧 TODO
- `set_edge_attr(&mut self, edge: EdgeId, attr: AttrName, value: AttrValue) -> Result<(), GraphError> ` 🚧 TODO
- `set_edge_attrs(&mut self, attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)` 🚧 TODO

*Attribute Getting:*
- `get_node_attr(&self, node: NodeId, attr: &AttrName) -> Result<Option<AttrValue>, GraphError> ` 🚧 TODO
- `get_edge_attr(&self, edge: EdgeId, attr: &AttrName) -> Result<Option<AttrValue>, GraphError> ` 🚧 TODO
- `get_node_attrs(&self, node: NodeId) -> Result<HashMap<AttrName, AttrValue>, GraphError> ` 🚧 TODO
- `get_edge_attrs(&self, edge: EdgeId) -> Result<HashMap<AttrName, AttrValue>, GraphError> ` 🚧 TODO
- `get_nodes_attrs(&self, attr: &AttrName, requested_nodes: &[NodeId]) -> GraphResult<Vec<Option<AttrValue>>> ` 🚧 TODO
- `get_edges_attrs(&self, attr: &AttrName, requested_edges: &[EdgeId]) -> GraphResult<Vec<Option<AttrValue>>> ` 🚧 TODO

*Topology Queries:*
- `contains_node(&self, node: NodeId) -> bool ` 🚧 TODO
- `contains_edge(&self, edge: EdgeId) -> bool ` 🚧 TODO
- `node_ids(&self) -> Vec<NodeId> ` 🚧 TODO
- `edge_ids(&self) -> Vec<EdgeId> ` 🚧 TODO
- `neighbors(&self, node: NodeId) -> Result<Vec<NodeId>, GraphError> ` 🚧 TODO
- `degree(&self, node: NodeId) -> Result<usize, GraphError> ` 🚧 TODO

*Version Control:*
- `commit(&mut self, message: String, author: String) -> Result<StateId, GraphError> ` 🚧 TODO
- `create_branch(&mut self, branch_name: BranchName) -> Result<(), GraphError> ` 🚧 TODO

*History & Views:*
- `list_branches(&self) -> Vec<BranchInfo> ` 🚧 TODO
- `commit_history(&self) -> Vec<CommitInfo> ` 🚧 TODO
- `view_at_commit(&self, commit_id: StateId) -> Result<HistoricalView, GraphError> ` 🚧 TODO
- `gc_history(&mut self) -> Result<usize, GraphError> ` 🚧 TODO

*Status & Config:*
- `statistics(&self) -> GraphStatistics ` 🚧 TODO
- `has_uncommitted_changes(&self) -> bool ` 🚧 TODO

*Other Operations:*
- `edge_endpoints(&self, edge: EdgeId) -> Result<(NodeId, NodeId), GraphError> ` 🚧 TODO
- `reset_hard(&mut self) -> Result<(), GraphError> ` 🚧 TODO
- `checkout_branch(&mut self, branch_name: BranchName) -> Result<(), GraphError> ` 🚧 TODO
- `find_nodes(&self, filter: NodeFilter) -> Result<Vec<NodeId>, GraphError> ` 🚧 TODO
- `find_edges(&self, filter: EdgeFilter) -> Result<Vec<EdgeId>, GraphError> ` 🚧 TODO
- `query(&self, query: GraphQuery) -> Result<QueryResult, GraphError> ` 🚧 TODO
- `create_view(&self) -> GraphView ` 🚧 TODO
- `diff_commits(&self, from: StateId, to: StateId) -> Result<CommitDiff, GraphError> ` 🚧 TODO
- `optimize(&mut self) -> Result<(), GraphError> ` 🚧 TODO
- `save_to_path(&self, path: &Path) -> Result<(), GraphError> ` 🚧 TODO

## 🏗️ Architecture Components

### core::pool - Data Storage Layer

**Functions**: 26 total, 25 public, 8 TODO
**Structs**: 3, **Traits**: 0

**Key Types:**
- `GraphPool`
- `AttributeColumn`
- `PoolStatistics`

**Key Functions:**
- `new()`
- `commit_baseline()`
- `get_node_attr_by_index()`
- `get_edge_attr_by_index()`
- `add_node()`
- ... 20 more functions

### core::space - Active State Tracking

**Functions**: 26 total, 26 public, 8 TODO
**Structs**: 1, **Traits**: 0

**Key Types:**
- `GraphSpace`

**Key Functions:**
- `new()`
- `with_strategy()`
- `activate_node()`
- `deactivate_node()`
- `activate_edge()`
- ... 21 more functions

### core::strategies - Temporal Storage Strategies

**Functions**: 30 total, 4 public, 2 TODO
**Structs**: 2, **Traits**: 1

**Key Types:**
- `StorageCharacteristics`
- `IndexDeltaStrategy`

**Traits:**
- `TemporalStorageStrategy`

**Key Functions:**
- `name()`
- `description()`
- `create_strategy()`
- `new()`

### core::change_tracker - Change Management

**Functions**: 33 total, 32 public, 22 TODO
**Structs**: 5, **Traits**: 0

**Key Types:**
- `ChangeTracker`
- `ChangeSummary`
- `MergeConflict`
- ... and 2 more

**Key Functions:**
- `new()`
- `with_strategy()`
- `with_custom_strategy()`
- `record_node_addition()`
- `record_node_removal()`
- ... 27 more functions

### core::history - Version Control System

**Functions**: 56 total, 54 public, 55 TODO
**Structs**: 8, **Traits**: 0

**Key Types:**
- `HistoryForest`
- `Commit`
- `Delta`
- ... and 5 more

**Key Functions:**
- ... 49 more functions

### core::state - State Management

**Functions**: 36 total, 33 public, 12 TODO
**Structs**: 7, **Traits**: 0

**Key Types:**
- `StateObject`
- `StateMetadata`
- `GraphSnapshot`
- ... and 4 more

**Key Functions:**
- `new_root()`
- `parent()`
- `delta()`
- `metadata()`
- `is_root()`
- ... 28 more functions

### core::delta - Change Deltas

**Functions**: 20 total, 17 public, 0 TODO
**Structs**: 3, **Traits**: 0

**Key Types:**
- `ColumnIndexDelta`
- `ColumnDelta`
- `DeltaObject`

**Key Functions:**
- `new()`
- `add_index_change()`
- `get_change()`
- `has_change()`
- `len()`
- ... 12 more functions

### core::query - Query Engine

**Functions**: 10 total, 7 public, 9 TODO
**Structs**: 18, **Traits**: 0

**Key Types:**
- `QueryEngine`
- `QueryConfig`
- `DegreeFilter`
- ... and 12 more

**Key Functions:**
- ... 2 more functions

### core::ref_manager - Branch/Tag Management

**Functions**: 35 total, 27 public, 34 TODO
**Structs**: 5, **Traits**: 0

**Key Types:**
- `Branch`
- `RefManager`
- `BranchInfo`
- ... and 2 more

**Key Functions:**
- ... 22 more functions

### config - Configuration Management

**Functions**: 16 total, 14 public, 13 TODO
**Structs**: 1, **Traits**: 0

**Key Types:**
- `GraphConfig`

**Key Functions:**
- `with_storage_strategy()`
- ... 9 more functions

### types - Core Type System

**Functions**: 6 total, 6 public, 6 TODO
**Structs**: 0, **Traits**: 0

**Key Functions:**
- ... 1 more functions

### errors - Error Handling

**Functions**: 15 total, 12 public, 1 TODO
**Structs**: 1, **Traits**: 0

**Key Types:**
- `MergeConflictDetail`

**Key Functions:**
- `node_not_found()`
- `edge_not_found()`
- `state_not_found()`
- `branch_not_found()`
- `uncommitted_changes()`
- ... 7 more functions

## 📚 Supporting Modules

### lib

**Functions**: 6 total, 2 public, 0 TODO
**Structs**: 1, **Traits**: 0

**Key Types:**
- `LibraryInfo`

**Key Functions:**
- `info()`
- `banner()`

### util

**Functions**: 7 total, 5 public, 5 TODO
**Structs**: 0, **Traits**: 0

**Key Functions:**
