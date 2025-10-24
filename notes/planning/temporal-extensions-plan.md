# Temporal Extensions Plan

## 🎯 Core Vision

Treat ChangeTracker history as a typed time-series, enabling powerful temporal graph analytics through immutable snapshots, time-windowed queries, and columnar temporal operations. This extends Groggy's attribute-first philosophy into the temporal dimension while preserving the core performance characteristics.

## 🏗️ Architecture Overview

### Design Principles

**History as First-Class Data** – The ChangeTracker already maintains a complete commit history; temporal extensions expose this as queryable time-series data with columnar access patterns.

**Snapshot Immutability** – Temporal snapshots are read-only views into historical states, guaranteeing consistent query results and enabling safe concurrent access.

**Columnar Temporal Ops** – Extend existing bulk operations (`neighbors_bulk`, `attr_bulk`, etc.) with temporal selectors so queries stay O(1) amortized without manual joins or iteration.

**Zero-Copy Where Possible** – Snapshots reference shared storage rather than duplicating data; only deltas and metadata need allocation.

**Composable Primitives** – Common temporal patterns (diff, window aggregation, temporal filter) become reusable algorithm steps that compose with existing pipelines.

---

## 📦 Core Components

### 1. TemporalSnapshot

Immutable handle into graph state at a specific point in time.

**Status:** ✅ Implemented (Rust core `src/temporal/snapshot.rs`, accessible via `Graph::snapshot_at_commit` / `Graph::snapshot_at_timestamp`, and exposed in Python as `TemporalSnapshot`).

#### Rust Core

```rust
// src/temporal/snapshot.rs
use crate::graph::{Subgraph, GraphSpace};
use crate::change_tracker::{ChangeTracker, CommitId};

pub struct TemporalSnapshot {
    /// Reference to the base graph storage
    graph_space: Arc<GraphSpace>,
    
    /// Commit identifier this snapshot represents
    commit_id: CommitId,
    
    /// Timestamp when this commit was made
    timestamp: DateTime<Utc>,
    
    /// Lineage metadata (parent commits, merge info)
    lineage: LineageMetadata,
    
    /// Cached view of node/edge existence at this commit
    existence_index: Arc<ExistenceIndex>,
}

impl TemporalSnapshot {
    /// Create snapshot at specific commit
    pub fn at_commit(
        tracker: &ChangeTracker,
        commit_id: CommitId,
    ) -> GraphResult<Self> {
        // Build existence index from history up to commit_id
        let existence = Self::build_existence_index(tracker, commit_id)?;
        
        Ok(Self {
            graph_space: tracker.graph_space().clone(),
            commit_id,
            timestamp: tracker.commit_timestamp(commit_id)?,
            lineage: tracker.lineage_at(commit_id)?,
            existence_index: Arc::new(existence),
        })
    }
    
    /// Create snapshot at specific timestamp (finds nearest commit)
    pub fn at_timestamp(
        tracker: &ChangeTracker,
        timestamp: DateTime<Utc>,
    ) -> GraphResult<Self> {
        let commit_id = tracker.commit_at_or_before(timestamp)?;
        Self::at_commit(tracker, commit_id)
    }
    
    /// Get subgraph representing this snapshot
    pub fn as_subgraph(&self) -> GraphResult<Subgraph> {
        // Create subgraph view filtered by existence index
        Subgraph::from_snapshot(
            self.graph_space.clone(),
            self.existence_index.clone(),
        )
    }
    
    /// Check if node exists in this snapshot
    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.existence_index.contains_node(node_id)
    }
    
    /// Check if edge exists in this snapshot
    pub fn edge_exists(&self, edge_id: EdgeId) -> bool {
        self.existence_index.contains_edge(edge_id)
    }
    
    /// Get attribute value at this point in time
    pub fn node_attr(&self, node_id: NodeId, key: &str) -> GraphResult<Option<AttributeValue>> {
        self.graph_space.node_attr_at_commit(node_id, key, self.commit_id)
    }
    
    /// Bulk neighbor query at this snapshot
    pub fn neighbors_bulk(&self, nodes: &[NodeId]) -> GraphResult<Vec<Vec<NodeId>>> {
        // Query neighbors filtered by existence at commit_id
        self.graph_space.neighbors_bulk_at_commit(nodes, self.commit_id)
    }
}

/// Index tracking which nodes/edges exist at a snapshot
pub struct ExistenceIndex {
    nodes: RoaringBitmap,
    edges: RoaringBitmap,
    
    /// Cached materialized state for fast lookups
    node_attrs_cache: DashMap<(NodeId, String), AttributeValue>,
}

pub struct LineageMetadata {
    pub parent_commits: Vec<CommitId>,
    pub merge_info: Option<MergeMetadata>,
    pub commit_message: String,
    pub author: Option<String>,
}
```

#### Python API

```python
# Snapshot creation
snapshot = g.snapshot_at(commit_id="abc123")
snapshot = g.snapshot_at(timestamp=datetime(2024, 1, 15))
snapshot = g.snapshot_at("2024-01-15T10:30:00Z")  # ISO string

# Snapshot as subgraph
sg = snapshot.as_subgraph()
sg.nodes  # Nodes that existed at snapshot time
sg.edges  # Edges that existed at snapshot time

# Direct queries on snapshot
exists = snapshot.node_exists(42)
neighbors = snapshot.neighbors(100)

# Metadata access
print(snapshot.commit_id)
print(snapshot.timestamp)
print(snapshot.lineage.parent_commits)
```

### 2. Temporal Selectors in History/Graph API

Index history by commit time to enable efficient temporal queries.

**Status:** ✅ Implemented via `HistoryForest::commit_at_or_before`, `Graph::snapshot_at_commit`, `Graph::snapshot_at_timestamp`, and `Graph::from_snapshot`.

#### Rust Core

```rust
// src/graph/graph_space.rs additions

impl GraphSpace {
    /// Get neighbors as they existed at a specific commit
    pub fn neighbors_bulk_at_commit(
        &self,
        nodes: &[NodeId],
        commit_id: CommitId,
    ) -> GraphResult<Vec<Vec<NodeId>>> {
        // Use temporal index to filter edges by creation/deletion time
        self.temporal_index
            .neighbors_at_commit(nodes, commit_id)
    }
    
    /// Get neighbors within a time window
    pub fn neighbors_bulk_between(
        &self,
        nodes: &[NodeId],
        start: CommitId,
        end: CommitId,
    ) -> GraphResult<Vec<Vec<NodeId>>> {
        // Return neighbors that existed at any point in [start, end]
        self.temporal_index
            .neighbors_in_window(nodes, start, end)
    }
    
    /// Get attribute history for nodes
    pub fn node_attrs_history(
        &self,
        nodes: &[NodeId],
        key: &str,
        from: CommitId,
        to: CommitId,
    ) -> GraphResult<Vec<Vec<(CommitId, AttributeValue)>>> {
        self.temporal_index
            .attr_timeline(nodes, key, from, to)
    }
}

/// Temporal index for efficient history queries
pub struct TemporalIndex {
    /// Map from edge to (creation_commit, deletion_commit)
    edge_lifetime: DashMap<EdgeId, (CommitId, Option<CommitId>)>,
    
    /// Map from node to (creation_commit, deletion_commit)
    node_lifetime: DashMap<NodeId, (CommitId, Option<CommitId>)>,
    
    /// Attribute change timeline: (node_id, key) -> sorted vec of (commit, value)
    attr_timeline: DashMap<(NodeId, String), Vec<(CommitId, AttributeValue)>>,
    
    /// Commit timestamp index for range queries
    commit_times: BTreeMap<DateTime<Utc>, CommitId>,
}

impl TemporalIndex {
    /// Build index from ChangeTracker history
    pub fn from_tracker(tracker: &ChangeTracker) -> GraphResult<Self> {
        let mut index = Self::new();
        
        for commit in tracker.commits_ordered() {
            index.process_commit(commit)?;
        }
        
        Ok(index)
    }
    
    /// Check if edge existed at commit
    pub fn edge_exists_at(&self, edge_id: EdgeId, commit: CommitId) -> bool {
        if let Some((created, deleted)) = self.edge_lifetime.get(&edge_id) {
            *created <= commit && deleted.map_or(true, |d| commit < d)
        } else {
            false
        }
    }
}
```

#### Python API

```python
# Temporal selectors in existing operations
neighbors = g.nodes([1, 2, 3]).neighbors(as_of=commit_id)
neighbors = g.nodes([1, 2, 3]).neighbors(between=(start_commit, end_commit))

# Attribute history
history = g.nodes[42].attr["status"].history(from_commit=start, to_commit=end)
# Returns: [(commit_id, timestamp, value), ...]

# Bulk temporal queries
attrs = g.nodes.bulk_attrs("status", as_of=commit_id)
```

#### Incremental Index Optimization Strategy

We avoid full replays when new commits land by streaming them through `TemporalIndex::apply_commit`. The ChangeTracker pushes batches of `EntityDelta` events, which we fold into the in-memory lifetime maps and attribute timelines. We keep a `dirty_range` marker so consumers know the index is authoritative through the latest applied commit; on restart we hydrate from the persisted snapshot and then replay only the trailing commits. A background task coalesces GC notifications into compact range removals so the index never drifts from the pruned history.

#### FFI Bindings

The temporal selector entry points are surfaced via `python-groggy/src/ffi/temporal.rs`, exposing `graph_snapshot_at`, `graph_neighbors_at_commit`, and `graph_neighbors_between`. Each wrapper converts `PyAny` arguments into either commit IDs or timestamps, funnels them through the Rust helpers above, and translates `TemporalIndexError` into Python `TemporalSelectorError`. Expensive calls release the GIL with `py.allow_threads(|| ...)`, and the resulting handles are exported through `python-groggy/python/groggy/_temporal.pyi` so the high-level API stays type hinted.

### 3. AlgorithmContext Temporal Extensions

Extend the algorithm execution context with temporal scope and helper methods.

#### Pipeline Builder Integration

`PipelineBuilder::with_temporal_scope` accepts either a precomputed `TemporalSnapshot` handle or `(start, end)` window tuple. During compilation we thread this scope into the `ContextInit` block so every executor thread starts with the same `TemporalScope` and downstream steps can rely on `ctx.temporal_scope()` being populated. Python mirrors this with `PipelineBuilder.temporal_scope(...)`, storing the metadata in the request envelope before dispatching to the Rust executor so DSL users do not need to mutate `Context` by hand.

#### Rust Core

```rust
// src/algorithms/context.rs additions

pub struct TemporalScope {
    /// Current commit being analyzed
    pub current_commit: CommitId,
    
    /// Optional window bounds for windowed operations
    pub window: Option<(CommitId, CommitId)>,
    
    /// Reference snapshot for comparison operations
    pub reference_snapshot: Option<Arc<TemporalSnapshot>>,
}

impl Context {
    /// Get current temporal scope
    pub fn temporal_scope(&self) -> Option<&TemporalScope> {
        self.temporal_scope.as_ref()
    }
    
    /// Set temporal scope for this context
    pub fn with_temporal_scope(&mut self, scope: TemporalScope) {
        self.temporal_scope = Some(scope);
    }
    
    /// Compute columnar diff between two snapshots
    pub fn delta(
        &self,
        prev: &TemporalSnapshot,
        cur: &TemporalSnapshot,
    ) -> GraphResult<TemporalDelta> {
        TemporalDelta::compute(prev, cur)
    }
    
    /// Get nodes/edges that changed in time window
    pub fn changed_entities(
        &self,
        window: (CommitId, CommitId),
    ) -> GraphResult<ChangedEntities> {
        let tracker = self.graph_space.change_tracker()?;
        tracker.changes_in_window(window.0, window.1)
    }
}

/// Represents differences between two snapshots
pub struct TemporalDelta {
    /// Nodes added in cur that weren't in prev
    pub nodes_added: Vec<NodeId>,
    
    /// Nodes removed from prev to cur
    pub nodes_removed: Vec<NodeId>,
    
    /// Edges added
    pub edges_added: Vec<EdgeId>,
    
    /// Edges removed
    pub edges_removed: Vec<EdgeId>,
    
    /// Attribute changes: (entity_id, key, old_value, new_value)
    pub attr_changes: Vec<AttributeChange>,
}

pub struct ChangedEntities {
    pub modified_nodes: RoaringBitmap,
    pub modified_edges: RoaringBitmap,
    pub change_types: HashMap<NodeId, ChangeType>,
}

#[derive(Clone, Debug)]
pub enum ChangeType {
    Created,
    Modified,
    Deleted,
}
```

#### Python API

```python
# In algorithm pipeline
def temporal_algorithm(ctx, subgraph):
    # Access temporal scope
    scope = ctx.temporal_scope
    print(f"Analyzing commit: {scope.current_commit}")
    
    # Get delta from reference
    if scope.reference_snapshot:
        delta = ctx.delta(scope.reference_snapshot, subgraph.snapshot)
        print(f"Added {len(delta.nodes_added)} nodes")
        print(f"Modified {len(delta.attr_changes)} attributes")
    
    # Get changes in window
    if scope.window:
        changed = ctx.changed_entities(scope.window)
        focus_nodes = subgraph.nodes[changed.modified_nodes]
    
    return subgraph

# Set temporal scope before execution
ctx = Context()
ctx.set_temporal_scope(
    current_commit=commit_id,
    window=(start_commit, end_commit),
    reference_snapshot=reference_snapshot
)

pipeline.run(ctx, subgraph)
```

#### FFI Bindings

`python-groggy/src/ffi/context.rs` exposes `ctx_delta`, `ctx_changed_entities`, and `ctx_temporal_scope` shims. Each converts Python snapshot handles back into `TemporalSnapshotRef`, forwards the call into the Rust context, and maps `TemporalScopeError` into `GroggyTemporalError`. Long-running diff computations release the GIL via `py.allow_threads(|| ...)`, and the return types land in `python-groggy/python/groggy/_context.pyi` so IDEs surface the helpers with type hints.

#### Test Coverage

Rust integration tests in `tests/temporal_context.rs` verify `ctx.delta` against manual diffs and windowed change detection; Python adds `tests/test_temporal_context.py::test_pipeline_scope_propagation` and `::test_delta_round_trip` to ensure pipeline-provided scopes reach algorithms and FFI marshaling yields consistent change summaries.

### 4. Temporal Algorithm Steps

Pre-built reusable primitives for common temporal operations.

#### Rust Core

```rust
// src/algorithms/steps/temporal.rs

/// Step: Compute difference between snapshots
pub struct DiffNodesStep {
    pub prev_snapshot: Arc<TemporalSnapshot>,
    pub output_var: String,  // Where to store result
}

impl Algorithm for DiffNodesStep {
    fn id(&self) -> &'static str { "temporal.diff_nodes" }
    
    fn execute(&self, ctx: &mut Context, sg: Subgraph) -> GraphResult<Subgraph> {
        let current_snapshot = sg.as_snapshot()?;
        let delta = ctx.delta(&self.prev_snapshot, &current_snapshot)?;
        
        // Store delta in context variables for downstream steps
        ctx.set_variable(&self.output_var, Variable::Delta(delta));
        
        Ok(sg)
    }
}

/// Step: Compute difference between edges
pub struct DiffEdgesStep {
    pub prev_snapshot: Arc<TemporalSnapshot>,
    pub output_var: String,
}

impl Algorithm for DiffEdgesStep {
    fn id(&self) -> &'static str { "temporal.diff_edges" }
    
    fn execute(&self, ctx: &mut Context, sg: Subgraph) -> GraphResult<Subgraph> {
        let current_snapshot = sg.as_snapshot()?;
        let delta = ctx.delta(&self.prev_snapshot, &current_snapshot)?;
        
        ctx.set_variable(&self.output_var, Variable::EdgeDelta(delta.edges_added, delta.edges_removed));
        
        Ok(sg)
    }
}

/// Step: Aggregate attribute over time window
pub struct WindowAggregateStep {
    pub window: (CommitId, CommitId),
    pub attr_key: String,
    pub agg_fn: AggregationFunction,  // sum, mean, max, min, etc.
    pub output_attr: String,
}

impl Algorithm for WindowAggregateStep {
    fn id(&self) -> &'static str { "temporal.window_aggregate" }
    
    fn execute(&self, ctx: &mut Context, mut sg: Subgraph) -> GraphResult<Subgraph> {
        let node_ids: Vec<_> = sg.nodes().collect();
        
        // Get attribute history for all nodes in window
        let histories = sg.graph_space()
            .node_attrs_history(&node_ids, &self.attr_key, self.window.0, self.window.1)?;
        
        // Apply aggregation function
        for (node_id, history) in node_ids.iter().zip(histories.iter()) {
            let values: Vec<_> = history.iter().map(|(_, v)| v).collect();
            let aggregated = self.agg_fn.apply(&values)?;
            
            sg.set_node_attr(*node_id, &self.output_attr, aggregated)?;
        }
        
        Ok(sg)
    }
}

/// Step: Filter entities by temporal criteria
pub struct TemporalFilterStep {
    pub predicate: TemporalPredicate,
}

pub enum TemporalPredicate {
    /// Keep only nodes created after timestamp
    CreatedAfter(DateTime<Utc>),
    
    /// Keep only nodes modified in window
    ModifiedInWindow(CommitId, CommitId),
    
    /// Keep nodes with attribute change matching pattern
    AttrChangedMatching { key: String, pattern: String },
    
    /// Keep nodes that existed at specific commit
    ExistedAt(CommitId),
}

impl Algorithm for TemporalFilterStep {
    fn id(&self) -> &'static str { "temporal.filter" }
    
    fn execute(&self, ctx: &mut Context, sg: Subgraph) -> GraphResult<Subgraph> {
        let temporal_index = sg.graph_space().temporal_index()?;
        
        let keep_nodes = match &self.predicate {
            TemporalPredicate::CreatedAfter(ts) => {
                temporal_index.nodes_created_after(*ts)?
            },
            TemporalPredicate::ModifiedInWindow(start, end) => {
                temporal_index.nodes_modified_between(*start, *end)?
            },
            // ... other predicates
            _ => unimplemented!(),
        };
        
        // Filter subgraph to matching nodes
        Ok(sg.filter_nodes(&keep_nodes))
    }
}
```

#### Python DSL

```python
from groggy.algorithms import PipelineBuilder

# Build temporal analysis pipeline
pipeline = (
    PipelineBuilder()
    
    # Get snapshot at reference time
    .step.snapshot(as_of="2024-01-01T00:00:00Z", output="ref_snapshot")
    
    # Compute diff from reference to current
    .step.diff(
        prev_var="ref_snapshot",
        current="input",
        output="delta"
    )
    
    # Filter to only modified nodes
    .step.temporal_filter(
        predicate="modified_in_window",
        window=(start_commit, end_commit)
    )
    
    # Aggregate historical attribute values
    .step.window_aggregate(
        attr="activity_score",
        window=(window_start, window_end),
        function="mean",
        output_attr="avg_activity"
    )
    
    # Run standard algorithm on temporal subset
    .step.pagerank(iterations=20)
    
    .build()
)

# Execute with temporal context
result = g.apply(pipeline)
```

### 5. Python Builder Shims

Intuitive Python interface for temporal operations.

```python
# python-groggy/python/groggy/algorithms/temporal.py

class TemporalStepBuilder:
    """Builder for temporal algorithm steps."""
    
    def snapshot(self, as_of=None, commit=None, output=None):
        """Create a temporal snapshot.
        
        Args:
            as_of: Timestamp (datetime or ISO string)
            commit: Commit ID
            output: Variable name to store snapshot
        """
        if as_of is not None:
            return SnapshotStep(timestamp=as_of, output=output)
        elif commit is not None:
            return SnapshotStep(commit_id=commit, output=output)
        else:
            raise ValueError("Must specify either as_of or commit")
    
    def diff(self, prev, current=None, ref="prior", output="delta"):
        """Compute diff between snapshots.
        
        Args:
            prev: Previous snapshot (variable name or TemporalSnapshot)
            current: Current snapshot (defaults to pipeline input)
            ref: Reference type ("prior" = previous commit in pipeline)
            output: Variable name to store delta
        """
        return DiffStep(
            prev_snapshot=prev,
            current_snapshot=current,
            reference_type=ref,
            output_var=output
        )
    
    def window_stat(self, attr, window, function, output_attr=None):
        """Compute statistic over time window.
        
        Args:
            attr: Attribute key to aggregate
            window: (start, end) commit/timestamp tuple
            function: Aggregation function (mean, sum, max, min, count)
            output_attr: Where to store result (defaults to f"{attr}_{function}")
        """
        if output_attr is None:
            output_attr = f"{attr}_{function}"
        
        return WindowAggregateStep(
            window=window,
            attr_key=attr,
            agg_fn=function,
            output_attr=output_attr
        )
    
    def filter_temporal(self, created_after=None, modified_in=None, existed_at=None):
        """Filter entities by temporal criteria.
        
        Args:
            created_after: Keep nodes created after timestamp
            modified_in: Keep nodes modified in (start, end) window
            existed_at: Keep nodes that existed at commit/timestamp
        """
        if created_after is not None:
            predicate = TemporalPredicate.created_after(created_after)
        elif modified_in is not None:
            predicate = TemporalPredicate.modified_in_window(*modified_in)
        elif existed_at is not None:
            predicate = TemporalPredicate.existed_at(existed_at)
        else:
            raise ValueError("Must specify temporal filter criteria")
        
        return TemporalFilterStep(predicate=predicate)

# Usage
from groggy.algorithms import step

pipeline = (
    PipelineBuilder()
    .add(step.snapshot(as_of="2024-01-01"))
    .add(step.diff(ref="prior"))
    .add(step.window_stat(attr="score", window=(start, end), function="mean"))
    .build()
)
```

---

## 📚 Temporal Contract & Guarantees

### Immutability

**Snapshots are read-only** – Once created, a TemporalSnapshot never changes, guaranteeing consistent query results even as the main graph evolves.

**Safe concurrent access** – Multiple threads/tasks can query the same snapshot simultaneously without locks.

**Snapshot independence** – Snapshots don't affect each other or the main graph state.

### Window Semantics

**Inclusive ranges** – Time windows `[start, end]` include both endpoints.

**Commit ordering** – All temporal queries respect the total order of commits in the ChangeTracker.

**Timestamp resolution** – Timestamps resolve to the nearest commit at or before the specified time.

### Cost Hints

**Snapshot creation** – O(commits × changes_per_commit) to build existence index. Amortized O(1) for cached snapshots.

**Temporal queries** – O(entities × log(commits)) with temporal index. Without index, O(entities × commits).

**Delta computation** – O(entities_in_union). Efficiently uses existence bitmaps for set operations.

**Window aggregation** – O(entities × commits_in_window × attr_accesses). Use narrower windows for better performance.

### Best Practices

**Snapshot reuse** – Cache frequently accessed snapshots rather than recreating them.

**Index coverage** – Ensure temporal index is built and up-to-date before heavy temporal queries.

**Window sizing** – Keep time windows as narrow as practical; wide windows require more history traversal.

**Batch operations** – Use bulk temporal queries rather than per-entity loops.

---

## 🚀 Implementation Roadmap

### Phase 1: Core Snapshot Infrastructure (2-3 weeks)

**Goal**: Basic temporal snapshot creation and queries

- [x] Implement `TemporalSnapshot` struct with existence indexing
- [x] Add `snapshot_at(commit_id)` and `snapshot_at(timestamp)` to Graph
- [x] Implement `ExistenceIndex` for fast membership checks
- [x] Add `as_subgraph()` conversion
- [x] Basic FFI bindings for snapshot creation
- [x] Python API: `g.snapshot_at(...)` 
- [x] Unit tests for snapshot creation and queries

**Deliverable**: Users can create snapshots and query them as immutable subgraphs.

### Phase 2: Temporal Index (2 weeks)

**Goal**: Efficient history-aware queries

- [x] Implement `TemporalIndex` structure
- [x] Build index from ChangeTracker history
- [x] Add `neighbors_at_commit` and `neighbors_between` to GraphSpace
- [x] Extend `node_attrs_history` for attribute timelines
- [x] Optimize index build for large histories (incremental updates)
- [x] FFI bindings for temporal selectors
- [x] Python API: `neighbors(as_of=...)`, `neighbors(between=...)`
- [x] Benchmarks for temporal queries vs. manual filtering

**Deliverable**: Columnar operations accept temporal selectors with <10% overhead vs. present-time queries.

### Phase 3: AlgorithmContext Extensions (1-2 weeks)

**Goal**: Temporal scope and delta helpers in algorithm context

- [x] Add `TemporalScope` to `Context`
- [x] Implement `ctx.delta(prev, cur)` helper
- [x] Add `ctx.changed_entities(window)` helper
- [x] Extend pipeline builder to set temporal scope
- [x] FFI bindings for context temporal methods
- [x] Python API: `ctx.temporal_scope`, `ctx.delta(...)`
- [x] Tests for temporal context in algorithms

**Deliverable**: Algorithms can access temporal metadata and compute diffs within pipeline execution.

### Phase 4: Temporal Algorithm Steps (2 weeks)

**Goal**: Reusable temporal primitives

- [x] Implement `DiffNodesStep` and `DiffEdgesStep`
- [x] Implement `WindowAggregateStep` with standard aggregation functions
- [x] Implement `TemporalFilterStep` with common predicates
- [x] Register steps in algorithm registry
- [x] FFI bindings for step execution
- [x] Python builder shims: `step.diff(...)`, `step.window_stat(...)`, `step.filter_temporal(...)`
- [x] Integration tests for temporal pipelines
- [x] Example notebooks demonstrating temporal analysis

**Deliverable**: Users can compose temporal analysis pipelines entirely through the DSL.

### Phase 5: Documentation & Polish (1 week)

**Goal**: Clear, comprehensive temporal documentation

- [x] Write temporal extensions guide (concepts, patterns, examples)
- [x] Document temporal contract (immutability, window semantics, cost hints)
- [x] Add docstrings to all public temporal APIs
- [ ] Create tutorial notebook: "Time-Travel Queries in Groggy"
- [ ] Create tutorial notebook: "Temporal Community Detection"
- [ ] Performance tuning guide for temporal queries
- [ ] Update API reference with temporal methods

**Deliverable**: Users understand when and how to use temporal features effectively.

**Status**: Core documentation complete in `docs/appendices/temporal-extensions-guide.md`. Tutorial notebooks and API reference updates pending.

---

## 🎯 Success Metrics

### Performance Targets

- **Snapshot creation**: < 100ms for graphs with 1M nodes and 10K commits
- **Temporal query overhead**: < 10% vs. present-time equivalent
- **Delta computation**: < 50ms for typical snapshots (10K node changes)
- **Window aggregation**: < 1s for 100K nodes over 100-commit window

### API Usability

- **Discoverability**: All temporal methods visible in IDE autocomplete and docs
- **Error messages**: Clear guidance when temporal index missing or timestamp out of range
- **Type safety**: Full type hints for temporal APIs in Python stubs
- **Composability**: Temporal steps work seamlessly with existing algorithm steps

### Correctness

- **Snapshot consistency**: Query results match manual history reconstruction
- **Delta accuracy**: No false positives/negatives in change detection
- **Window semantics**: Inclusive ranges work as documented
- **Concurrency safety**: No races when querying snapshots from multiple threads

---

## 🔮 Future Extensions

### Incremental Index Updates

Currently, temporal index is rebuilt from full history. Future optimization:
- Track index state per commit
- Incrementally update index as new commits arrive
- Persist index to disk to avoid rebuild on restart

### Snapshot Persistence

Save frequently-accessed snapshots to disk:
- Serialize existence index and metadata
- Load snapshots without replaying history
- Manage snapshot cache with LRU eviction

### Temporal Joins

Enable joining snapshots from different time points:
- Compare node attributes across time
- Track attribute evolution per entity
- Identify entities with correlated temporal patterns

### Temporal Visualization

Extend viz system with temporal capabilities:
- Animate graph evolution over time
- Highlight changes between snapshots
- Interactive timeline scrubbing

### Distributed Temporal Queries

For very large histories:
- Partition temporal index across machines
- Distributed snapshot creation
- Parallel window aggregation

---

## 🎓 Example Use Cases

### Burst Detection

Identify sudden spikes in graph activity:

```python
# Define time windows
windows = [(t, t + timedelta(hours=1)) for t in hourly_timestamps]

# Count changes per window
for start, end in windows:
    changed = ctx.changed_entities((start, end))
    if len(changed.modified_nodes) > threshold:
        print(f"Burst detected at {start}: {len(changed.modified_nodes)} nodes")
```

### Community Drift Analysis

Track how communities evolve over time:

```python
# Get snapshots at intervals
snapshots = [g.snapshot_at(ts) for ts in weekly_timestamps]

# Run community detection at each point
communities_over_time = []
for snapshot in snapshots:
    sg = snapshot.as_subgraph()
    communities = sg.communities.louvain()
    communities_over_time.append(communities)

# Measure community stability
for i in range(1, len(communities_over_time)):
    stability = compute_jaccard(communities_over_time[i-1], communities_over_time[i])
    print(f"Week {i} stability: {stability:.2f}")
```

### Churn Scoring

Identify nodes with high attribute volatility:

```python
pipeline = (
    PipelineBuilder()
    
    # Aggregate change counts over window
    .step.window_stat(
        attr="status",
        window=(month_start, month_end),
        function="change_count",  # Count how many times attr changed
        output_attr="churn_score"
    )
    
    # Filter to high-churn nodes
    .step.filter(lambda node: node["churn_score"] > 10)
    
    # Analyze high-churn subgraph
    .step.centrality.betweenness()
    
    .build()
)
```

### Temporal Reachability

Find paths that existed at specific time:

```python
# Snapshot at historical point
snapshot = g.snapshot_at("2023-06-15")
sg = snapshot.as_subgraph()

# Run pathfinding on historical graph
path = sg.shortest_path(source=A, target=B)
print(f"Path existed on 2023-06-15: {path}")
```

---

## 🎨 Integration with Existing Systems

### ChangeTracker Integration

Temporal extensions build on existing ChangeTracker infrastructure:
- Leverage commit history without modifications
- Use existing commit metadata (timestamps, messages, authors)
- Temporal index updates when commits are garbage collected

### Algorithm Pipeline Integration

Temporal steps compose seamlessly with existing steps:
- Input/output contracts match standard subgraph operations
- Context variables store temporal metadata
- No special pipeline mode required

### FFI Layer Integration

Follow established FFI patterns:
- Marshal snapshots as opaque handles
- Use same error translation for temporal errors
- Release GIL for expensive temporal operations (`py.allow_threads()`)

### Columnar Operation Integration

Temporal selectors extend existing bulk operations:
- `neighbors_bulk` naturally accepts `as_of` parameter
- Attribute accessors return timelines for temporal queries
- Preserve O(1) amortized complexity expectations

---

## ✅ Validation Strategy

### Unit Tests

- Snapshot creation and membership queries
- Temporal index build and queries
- Delta computation accuracy
- Window aggregation correctness

### Integration Tests

- End-to-end temporal pipelines
- Snapshot + algorithm execution
- Multi-snapshot comparisons
- Large history stress tests

### Performance Tests

- Benchmark snapshot creation vs. graph size and commit count
- Measure temporal query overhead
- Profile window aggregation scaling
- Test concurrent snapshot access

### Correctness Tests

- Compare temporal queries to manual history reconstruction
- Validate immutability guarantees
- Check window boundary conditions
- Verify delta completeness (no missing changes)

---

## 📝 Conclusion

Temporal extensions transform Groggy's change tracking into a powerful time-series graph database capability. By treating history as first-class columnar data and providing composable temporal primitives, users can express sophisticated temporal analytics entirely through the Python DSL while maintaining Rust-level performance.

The architecture preserves Groggy's core principles: attribute-first operations, minimal FFI overhead, and clear separation between Rust core and Python composition. Temporal features integrate naturally with existing algorithms and pipelines, enabling time-travel queries, drift analysis, and historical pattern mining without learning a new programming model.

Following the phased roadmap ensures incremental value delivery while maintaining code quality and test coverage. Each phase builds on the previous, with clear deliverables and success metrics to track progress.
