# Temporal Extensions Guide

## Overview

Groggy's temporal extensions transform it into a time-series graph database, enabling powerful temporal analytics on evolving graphs. This guide covers the concepts, APIs, and patterns for working with temporal features.

## Core Concepts

### 1. Temporal Snapshots

A **TemporalSnapshot** is an immutable view of your graph at a specific point in time. Think of it as a photograph of your graph's state at a particular commit or timestamp.

```python
import groggy as gr

g = gr.Graph()
# ... build and modify graph ...
commit_id = g.commit("Checkpoint 1", "user")

# Create snapshot at specific commit
snapshot = g.snapshot_at_commit(commit_id)

# Create snapshot at timestamp
snapshot = g.snapshot_at_timestamp(1704067200)  # Unix timestamp
```

**Key Properties:**
- **Immutable**: Snapshots never change, ensuring consistent queries
- **Lightweight**: Reference shared storage where possible
- **Queryable**: Can be converted to subgraphs for algorithm execution

### 2. Temporal Index

The **TemporalIndex** enables efficient O(log n) queries on historical data without reconstructing full snapshots.

```python
# Build index once
index = g.build_temporal_index()

# Fast temporal queries
exists = index.node_exists_at(node_id, commit_id)
neighbors = index.neighbors_at_commit(node_id, commit_id)
history = index.node_attr_history(node_id, "status", start_commit, end_commit)
```

**What it Tracks:**
- When nodes and edges were created/deleted
- Attribute value changes over time
- Entity existence at any commit
- Neighbors at specific points in time

**When to Use:**
- Making many temporal queries
- Analyzing change patterns
- Time-window aggregations
- Historical pathfinding

### 3. Temporal Scope

A **TemporalScope** defines the temporal context for algorithm execution, allowing algorithms to be time-aware.

```python
from groggy import TemporalScope

# Scope at a specific commit
scope = TemporalScope(current_commit=42, window=None)

# Scope with a time window
scope = TemporalScope(current_commit=50, window=(10, 50))

# Use in algorithm context
ctx = gr.Context.with_temporal_scope(scope)
```

**Use Cases:**
- Running algorithms on historical states
- Windowed temporal analysis
- Change-aware computations
- Temporal filtering

### 4. Temporal Delta

A **TemporalDelta** represents the differences between two snapshots, capturing what changed.

```python
snapshot1 = g.snapshot_at_commit(commit1)
snapshot2 = g.snapshot_at_commit(commit2)

# Compute delta (would need direct API exposure)
# For now, use index to track changes
nodes_at_c1 = index.nodes_at_commit(commit1)
nodes_at_c2 = index.nodes_at_commit(commit2)
nodes_added = set(nodes_at_c2) - set(nodes_at_c1)
```

## Common Patterns

### Pattern 1: Time-Travel Queries

Analyze your graph as it existed at a specific point in time.

```python
# Get snapshot from the past
snapshot = g.snapshot_at(timestamp="2024-01-15T10:00:00Z")
subgraph = snapshot.as_subgraph()

# Run algorithms on historical state
pagerank = subgraph.centrality.pagerank()
communities = subgraph.communities.louvain()

# Query historical topology
historical_neighbors = snapshot.neighbors(node_id)
existed = snapshot.node_exists(node_id)
```

**Use Cases:**
- Debugging: "What did the graph look like when X happened?"
- Compliance: "Show me the network state on date Y"
- Analysis: "How have communities evolved?"

### Pattern 2: Change Detection

Identify what changed over time.

```python
index = g.build_temporal_index()

# Track changes in a single commit
changed_nodes = index.nodes_changed_in_commit(commit_id)
changed_edges = index.edges_changed_in_commit(commit_id)

# Find nodes that changed in a window
for commit_id in range(start_commit, end_commit + 1):
    changed = index.nodes_changed_in_commit(commit_id)
    if len(changed) > threshold:
        print(f"Burst detected at commit {commit_id}: {len(changed)} changes")
```

**Use Cases:**
- Burst detection: Find periods of rapid change
- Audit trails: Track which entities were modified
- Change propagation: Analyze cascading effects

### Pattern 3: Attribute Timelines

Query how attributes changed over time.

```python
index = g.build_temporal_index()

# Get full history for an attribute
history = index.node_attr_history(
    node_id, 
    "status", 
    start_commit=1, 
    end_commit=100
)

# Each entry is (commit_id, value)
for commit_id, value in history:
    print(f"At commit {commit_id}: status = {value}")

# Count how many times an attribute changed
change_count = len(history)
```

**Use Cases:**
- Churn analysis: Identify high-volatility attributes
- Lifecycle tracking: Monitor entity state transitions
- Anomaly detection: Find unusual change patterns

### Pattern 4: Temporal Neighbor Queries

Query graph topology as it existed at different times.

```python
index = g.build_temporal_index()

# Neighbors at a specific commit
neighbors_then = index.neighbors_at_commit(node_id, past_commit)

# Neighbors that existed at ANY point in a window
neighbors_in_window = index.neighbors_in_window(
    node_id, 
    start_commit, 
    end_commit
)

# Bulk queries for efficiency
bulk_neighbors = index.neighbors_bulk_at_commit(
    [node1, node2, node3], 
    commit_id
)
```

**Use Cases:**
- Historical reachability: "Was path A→B possible at time T?"
- Influence analysis: "Who were X's neighbors during period Y?"
- Temporal motifs: Find recurring connection patterns

### Pattern 5: Window Aggregations

Aggregate values over temporal windows.

```python
index = g.build_temporal_index()

# Get all changes in a time window
all_commits = range(start_commit, end_commit + 1)
all_changed_nodes = set()

for commit in all_commits:
    changed = index.nodes_changed_in_commit(commit)
    all_changed_nodes.update(changed)

# Compute statistics
change_rate = len(all_changed_nodes) / (end_commit - start_commit + 1)
```

**Use Cases:**
- Activity metrics: Measure graph dynamism
- Rolling statistics: Moving averages over time
- Trend detection: Identify accelerating/decelerating change

## Best Practices

### Performance Optimization

1. **Build Index Once**: If making many temporal queries, build the index once and reuse it.
   ```python
   index = g.build_temporal_index()  # One-time cost
   # Make many queries...
   ```

2. **Use Bulk Operations**: Prefer bulk queries over loops.
   ```python
   # Good: O(n) with single call
   neighbors = index.neighbors_bulk_at_commit(nodes, commit)
   
   # Avoid: O(n²) with many calls
   for node in nodes:
       n = index.neighbors_at_commit(node, commit)
   ```

3. **Limit History Queries**: Don't query the entire history unnecessarily.
   ```python
   # Good: Query specific window
   history = index.node_attr_history(node, attr, start, end)
   
   # Avoid: Query all commits
   history = index.node_attr_history(node, attr, 0, latest_commit)
   ```

4. **Snapshot Reuse**: Convert snapshots to subgraphs once and reuse.
   ```python
   snapshot = g.snapshot_at_commit(commit_id)
   sg = snapshot.as_subgraph()  # One conversion
   # Run multiple algorithms on sg
   ```

### Memory Management

1. **Index Memory**: The temporal index stores timelines proportional to the number of changes.
   - Expect ~50-100 bytes per tracked change
   - Monitoring large histories may require significant memory

2. **Snapshot Lifecycle**: Snapshots hold references to shared data.
   - Safe to create many snapshots
   - Automatically cleaned up when no longer referenced

3. **Commit Strategy**: More frequent commits = more temporal granularity but larger index.
   - Balance between query precision and memory usage
   - Consider committing at natural boundaries (transactions, time intervals)

### API Guidelines

1. **Timestamps**: Use Unix timestamps (seconds since epoch) or ISO 8601 strings.
   ```python
   # Both work
   snapshot = g.snapshot_at_timestamp(1704067200)
   snapshot = g.snapshot_at(timestamp="2024-01-15T10:00:00Z")
   ```

2. **Commit IDs**: Commits are sequential integers starting from 1.
   ```python
   commit_id = g.commit("message", "author")
   # commit_id is an integer: 1, 2, 3, ...
   ```

3. **Error Handling**: Check for existence before querying.
   ```python
   if snapshot.node_exists(node_id):
       neighbors = snapshot.neighbors(node_id)
   ```

## Temporal Contract

### Immutability Guarantees

- **Snapshots** are immutable: Once created, they represent a fixed point in time
- **Deltas** are immutable: Represent differences between two fixed states
- **Index** is immutable: Reflects state at build time; rebuild if graph changes

### Window Semantics

- **Inclusive ranges**: `[start, end]` includes both endpoints
- **Empty windows**: Window (x, x) contains only commit x
- **Ordering**: start must be ≤ end

### Cost Hints

- **Snapshot creation**: O(commit_depth) - reconstructs state from history
- **Index build**: O(total_changes) - processes all commits once
- **Temporal queries**: O(log n) - binary search on timelines
- **Window aggregations**: O(window_size × entities) - proportional to window and entity count

## Troubleshooting

### "No commit found at or before timestamp"

**Cause**: Querying a timestamp before any commits.

**Solution**: Ensure commits exist before the query timestamp.

```python
# Check available commits
commits = g.list_commits()  # If API exists
# Or create an initial commit
g.commit("Initial state", "system")
```

### "No temporal scope set in context"

**Cause**: Temporal algorithm step running without temporal scope.

**Solution**: Set temporal scope before running temporal algorithms.

```python
ctx = Context.with_temporal_scope(TemporalScope.at_commit(42))
```

### High Memory Usage

**Cause**: Large temporal index for graphs with many changes.

**Solution**:
1. Commit less frequently
2. Garbage collect old history if not needed
3. Query specific windows instead of full history

### Slow Temporal Queries

**Cause**: Not using the temporal index, or querying very large windows.

**Solution**:
1. Build and reuse the temporal index
2. Narrow time windows
3. Use bulk operations
4. Filter entities before temporal queries

## Examples

### Example 1: Community Drift Analysis

Track how communities evolve over time.

```python
g = gr.Graph()
# ... populate graph over time with commits ...

# Take snapshots at regular intervals
snapshot_commits = list(range(10, 100, 10))  # Every 10 commits
snapshots = [g.snapshot_at_commit(c) for c in snapshot_commits]

# Run community detection at each point
communities_over_time = []
for snapshot in snapshots:
    sg = snapshot.as_subgraph()
    communities = sg.communities.louvain()
    communities_over_time.append(communities)

# Measure stability between consecutive snapshots
# (Requires community comparison logic)
```

### Example 2: Churn Scoring

Identify high-volatility nodes.

```python
index = g.build_temporal_index()

# Count changes per node
churn_scores = {}
for node in g.nodes():
    # Count commits where this node changed
    changes = 0
    for commit in range(1, latest_commit + 1):
        if node in index.nodes_changed_in_commit(commit):
            changes += 1
    churn_scores[node] = changes

# Find high-churn nodes
threshold = 10
high_churn = {n: s for n, s in churn_scores.items() if s > threshold}
```

### Example 3: Temporal Reachability

Find if a path existed at a specific time.

```python
snapshot = g.snapshot_at_commit(historical_commit)
sg = snapshot.as_subgraph()

# Check if path existed then
try:
    path = sg.shortest_path(source=node_a, target=node_b)
    print(f"Path existed at commit {historical_commit}: {path}")
except:
    print(f"No path existed at commit {historical_commit}")
```

## Integration with Algorithms

### Using Temporal Steps in Pipelines

```python
# Future: When pipeline DSL supports temporal steps
# pipeline = (
#     PipelineBuilder()
#     .step.temporal.diff_nodes(before=var1, after=var2, output_prefix="delta")
#     .step.temporal.mark_changed_nodes(output="changed")
#     .step.filter(lambda node: node["changed"] == 1)
#     .step.centrality.betweenness()
#     .build()
# )
```

### Custom Temporal Algorithms

Create algorithms that leverage temporal context:

```python
from groggy.algorithms import Context, TemporalScope

def temporal_analysis_algorithm(ctx: Context, subgraph):
    # Check if temporal scope is set
    if ctx.temporal_scope():
        scope = ctx.temporal_scope()
        print(f"Analyzing at commit: {scope.current_commit}")
        
        # Use temporal features
        if scope.has_window():
            start, end = scope.window
            print(f"Window: [{start}, {end}]")
    
    # Run algorithm logic
    # ...
```

## Future Enhancements

Areas for potential extension:

1. **Temporal Joins**: Correlate events across different time series
2. **Forecasting**: Predict future graph states based on historical patterns
3. **Anomaly Detection**: Identify unusual temporal patterns automatically
4. **Temporal Constraints**: Define validity periods for relationships
5. **Bi-temporal Support**: Separate transaction time from valid time

## See Also

- API reference (see main documentation navigation)
- Performance guide (see main documentation navigation)
- Tutorial notebooks (external)
