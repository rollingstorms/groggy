# History/Temporal Analysis Tutorial Plan

## Objective
Create a comprehensive tutorial demonstrating Groggy's history module for temporal graph analysis, focusing on commits, branches, and version control workflows.

## Tutorial Structure

### 1. **Introduction to Graph Versioning** (5 minutes)
- Why version control matters for graph analysis
- Groggy's git-like approach to graph history
- Use cases: A/B testing, scenario modeling, temporal evolution

### 2. **Basic History Operations** (10 minutes)
```python
# Initialize graph with version control
g = gr.Graph()
# Build initial network
g.commit("Initial network", "analyst@company.com")

# Check status and history
g.has_uncommitted_changes()
g.commit_history()
```

### 3. **Branching for Scenario Analysis** (15 minutes)
```python
# Main timeline: Organic growth
g.create_branch("organic-growth")
g.checkout_branch("organic-growth")
# Add gradual connections
g.commit("Month 1: Organic connections", "analyst@company.com")

# Alternative timeline: Intervention
g.create_branch("intervention")
g.checkout_branch("intervention")  
# Add strategic connections
g.commit("Month 1: Strategic intervention", "analyst@company.com")

# Compare outcomes
g.branches()  # List all branches
```

### 4. **Historical Views and Time Travel** (10 minutes)
```python
# Access historical states
history = g.commit_history()
early_state = g.historical_view(history[0])  # First commit

# Compare past vs present
past_density = early_state.density()
current_density = g.density()
evolution = current_density - past_density
```

### 5. **Temporal Metrics Tracking** (15 minutes)
```python
def capture_temporal_snapshot(graph, timestamp):
    return {
        "timestamp": timestamp,
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "density": graph.density(),
        "components": len(graph.analytics.connected_components()),
        "avg_degree": sum(graph.degree()) / graph.node_count()
    }

# Track evolution across commits
timeline = []
for i, commit in enumerate(g.commit_history()):
    historical_g = g.historical_view(commit)
    snapshot = capture_temporal_snapshot(historical_g, f"commit_{i}")
    timeline.append(snapshot)
```

### 6. **Real-World Scenario: Social Media Platform Evolution** (20 minutes)

#### Phase 1: Launch (Commit 1)
- Small user base, high engagement
- Dense connections among early adopters

#### Phase 2: Growth (Commit 2) 
- User acquisition, declining density
- Emergence of communities

#### Phase 3: Maturity (Commit 3)
- Platform algorithms influence connections
- Different growth patterns in different branches

```python
# Main branch: Algorithm-driven growth
g.checkout_branch("main")
# Simulate algorithmic recommendations
g.commit("Phase 3: Algorithm recommendations", "product@company.com")

# Alternative branch: Organic growth only  
g.create_branch("organic-only")
g.checkout_branch("organic-only")
# Simulate natural growth patterns
g.commit("Phase 3: Organic growth", "research@company.com")
```

### 7. **Branch Comparison and Analysis** (15 minutes)
```python
def compare_branches(graph, branch1, branch2):
    # Switch to first branch
    graph.checkout_branch(branch1)
    metrics1 = get_network_metrics(graph)
    
    # Switch to second branch  
    graph.checkout_branch(branch2)
    metrics2 = get_network_metrics(graph)
    
    # Compare
    comparison = {}
    for metric in metrics1:
        if isinstance(metrics1[metric], (int, float)):
            comparison[metric] = {
                branch1: metrics1[metric],
                branch2: metrics2[metric], 
                "difference": metrics2[metric] - metrics1[metric],
                "percent_change": ((metrics2[metric] - metrics1[metric]) / metrics1[metric]) * 100
            }
    
    return comparison

# Compare algorithmic vs organic growth
results = compare_branches(g, "main", "organic-only")
```

### 8. **Temporal Network Analysis** (10 minutes)
```python
# Analyze how node importance changes over time
def track_centrality_evolution(graph):
    history = graph.commit_history()
    centrality_timeline = []
    
    for commit in history:
        historical_g = graph.historical_view(commit)
        degrees = historical_g.degree()
        
        # Track top nodes
        node_centrality = {}
        for i, degree in enumerate(degrees):
            if historical_g.has_node(i):
                node = historical_g.nodes[i]
                name = node.get('name', f'Node_{i}')
                node_centrality[name] = degree
                
        centrality_timeline.append({
            "commit": commit,
            "centrality": node_centrality
        })
    
    return centrality_timeline

# Track how influencers emerge over time
centrality_evolution = track_centrality_evolution(g)
```

### 9. **Predictive Temporal Modeling** (10 minutes)
```python
# Extract temporal features for prediction
def extract_temporal_features(graph):
    features = []
    history = graph.commit_history()
    
    for i, commit in enumerate(history):
        historical_g = graph.historical_view(commit)
        
        # Network-level features
        feature_vector = {
            "time_step": i,
            "node_count": historical_g.node_count(),
            "edge_count": historical_g.edge_count(),
            "density": historical_g.density(),
            "components": len(historical_g.analytics.connected_components()),
            "max_degree": max(historical_g.degree()) if historical_g.node_count() > 0 else 0
        }
        
        features.append(feature_vector)
    
    return features

# Could be used for:
# - Predicting future network growth
# - Identifying critical time points
# - Comparing intervention effectiveness
temporal_features = extract_temporal_features(g)
```

### 10. **Best Practices and Workflows** (5 minutes)
- Commit frequency strategies
- Branch naming conventions
- Merge strategies (if supported)
- Performance considerations for large histories

## Key Learning Outcomes

By the end of this tutorial, users will:

1. **Understand version control for graphs** - Why and how to use git-like workflows
2. **Master branching strategies** - Create alternative timelines for analysis
3. **Perform temporal comparisons** - Quantify how networks change over time
4. **Extract temporal features** - Prepare data for predictive modeling
5. **Implement real-world workflows** - Apply to business and research scenarios

## Tutorial Assets

- **Sample datasets**: Social network, organizational growth, epidemic spread
- **Code templates**: Reusable functions for temporal analysis
- **Visualization helpers**: Export data for temporal plotting
- **Performance tips**: Handle large histories efficiently

## Testing Requirements

Each code example must:
- ✅ Execute without errors
- ✅ Produce meaningful results
- ✅ Include error handling where appropriate
- ✅ Work with the current Groggy implementation
- ⚠️ Note any limitations or stub implementations

## Prerequisites

- Basic Groggy graph operations
- Understanding of network analysis concepts
- Familiarity with version control concepts (helpful but not required)

This tutorial will showcase Groggy's unique temporal analysis capabilities that set it apart from other graph libraries.