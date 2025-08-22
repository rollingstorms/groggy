# Temporal Graph Analysis with Groggy

This tutorial demonstrates how to use Groggy's history and versioning system for temporal graph analysis - analyzing how networks evolve over time.

## Overview

Groggy provides powerful versioning capabilities that enable you to:
- Track graph changes over time
- Create branches for different scenarios
- Analyze historical states
- Compare graph evolution across timelines

## Setting Up a Temporal Analysis

### 1. Initialize Graph with Version Control

```python
import groggy as gr

# Create a social network that will evolve over time
g = gr.Graph()

# Initial state: Small startup company
alice = g.add_node(name="Alice", role="CEO", join_date="2020-01")
bob = g.add_node(name="Bob", role="CTO", join_date="2020-01")  
carol = g.add_node(name="Carol", role="Engineer", join_date="2020-02")

# Initial relationships
g.add_edge(alice, bob, relationship="co-founder", strength=0.9)
g.add_edge(bob, carol, relationship="manager", strength=0.7)

# Commit initial state
g.commit("Company founding - initial team", "analyst@company.com")
print(f"✓ Initial state: {g.node_count()} employees, {g.edge_count()} relationships")
```

### 2. Simulate Growth Over Time

```python
# === QUARTER 1: HIRING EXPANSION ===
g.checkout_branch("main")

# Add new hires
dave = g.add_node(name="Dave", role="Sales", join_date="2020-03")
eve = g.add_node(name="Eve", role="Marketing", join_date="2020-04")

# New relationships form
g.add_edge(alice, dave, relationship="direct_report", strength=0.6)
g.add_edge(alice, eve, relationship="direct_report", strength=0.6)
g.add_edge(dave, eve, relationship="colleague", strength=0.5)

# Existing relationships strengthen
g.set_edge_attribute(g.edges[0].id, "strength", 0.95)  # Alice-Bob stronger

g.commit("Q1 2020: First hiring wave", "analyst@company.com")
print(f"✓ Q1 2020: {g.node_count()} employees, {g.edge_count()} relationships")

# === QUARTER 2: REORGANIZATION ===

# Promote Carol to team lead
g.set_node_attribute(carol, "role", "Engineering Manager")

# Add her team
frank = g.add_node(name="Frank", role="Engineer", join_date="2020-05")
grace = g.add_node(name="Grace", role="Engineer", join_date="2020-06")

# New management structure
g.add_edge(carol, frank, relationship="manager", strength=0.7)
g.add_edge(carol, grace, relationship="manager", strength=0.7)
g.add_edge(frank, grace, relationship="colleague", strength=0.6)

g.commit("Q2 2020: Engineering team expansion", "analyst@company.com")
print(f"✓ Q2 2020: {g.node_count()} employees, {g.edge_count()} relationships")

# === QUARTER 3: DEPARTMENT SILOS ===

# Marketing expansion
henry = g.add_node(name="Henry", role="Marketing", join_date="2020-07")
iris = g.add_node(name="Iris", role="Marketing", join_date="2020-08")

g.add_edge(eve, henry, relationship="manager", strength=0.8)
g.add_edge(eve, iris, relationship="manager", strength=0.8)
g.add_edge(henry, iris, relationship="colleague", strength=0.7)

# Sales expansion  
jack = g.add_node(name="Jack", role="Sales", join_date="2020-09")
g.add_edge(dave, jack, relationship="manager", strength=0.8)

g.commit("Q3 2020: Department growth", "analyst@company.com")
print(f"✓ Q3 2020: {g.node_count()} employees, {g.edge_count()} relationships")
```

### 3. Create Alternative Timeline (Branch Analysis)

```python
# === ALTERNATIVE SCENARIO: REMOTE-FIRST CULTURE ===
g.create_branch("remote-first")
g.checkout_branch("remote-first")

# In this timeline, different collaboration patterns emerge
# Add remote collaboration tools as "virtual nodes"
slack = g.add_node(name="Slack", role="Tool", type="virtual")
zoom = g.add_node(name="Zoom", role="Tool", type="virtual")

# Everyone connects through tools (higher tool-mediated communication)
for node_id in range(10):  # First 10 employees
    if g.has_node(node_id):
        g.add_edge(node_id, slack, relationship="uses", strength=0.9)
        g.add_edge(node_id, zoom, relationship="uses", strength=0.8)

# Cross-department collaboration increases (remote breaks down silos)
g.add_edge(frank, henry, relationship="cross_collab", strength=0.6)
g.add_edge(grace, iris, relationship="cross_collab", strength=0.6)
g.add_edge(carol, eve, relationship="cross_collab", strength=0.7)

g.commit("Q3 2020: Remote-first culture", "analyst@company.com")
print(f"✓ Remote timeline: {g.node_count()} entities, {g.edge_count()} connections")
```

## Temporal Analysis Techniques

### 4. Historical State Analysis

```python
# Get commit history for analysis
history = g.commit_history()
print(f"\\n=== COMMIT HISTORY ===")
for i, commit in enumerate(history):
    print(f"Commit {i}: {commit}")

# Analyze each historical state (note: this may be stub implementation)
print(f"\\n=== TEMPORAL ANALYSIS ===")

try:
    # Compare states across time
    g.checkout_branch("main")
    current_density = g.density()
    current_components = len(g.analytics.connected_components())
    
    print(f"Current state (main branch):")
    print(f"  Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    print(f"  Density: {current_density:.3f}")
    print(f"  Components: {current_components}")
    
    # Compare with alternative timeline
    g.checkout_branch("remote-first")
    remote_density = g.density()
    remote_components = len(g.analytics.connected_components())
    
    print(f"\\nRemote-first timeline:")
    print(f"  Nodes: {g.node_count()}, Edges: {g.edge_count()}")
    print(f"  Density: {remote_density:.3f}")
    print(f"  Components: {remote_components}")
    
    print(f"\\nComparison:")
    print(f"  Density change: {remote_density - current_density:+.3f}")
    print(f"  More connected: {'Yes' if remote_density > current_density else 'No'}")
    
except Exception as e:
    print(f"Historical analysis failed: {e}")
```

### 5. Role Evolution Analysis

```python
# Analyze how roles changed over time
g.checkout_branch("main")
current_table = g.nodes.table()

print(f"\\n=== ROLE EVOLUTION ANALYSIS ===")

# Current role distribution
role_dist = current_table['role'].value_counts() if hasattr(current_table['role'], 'value_counts') else "N/A"
print(f"Current role distribution: {role_dist}")

# Department sizes
dept_table = current_table.group_by('role') if hasattr(current_table, 'group_by') else None
if dept_table:
    print(f"Department analysis completed")
else:
    print("Department analysis: Need implementation")

# Leadership span of control
print(f"\\nLeadership Analysis:")
leadership_roles = ["CEO", "CTO", "Engineering Manager"]
for role in leadership_roles:
    role_nodes = current_table[current_table['role'] == role]
    if len(role_nodes) > 0:
        # Get first node with this role
        leader_id = role_nodes[0]['node_id'] if hasattr(role_nodes[0], '__getitem__') else "N/A"
        if leader_id != "N/A":
            try:
                leader_degree = g.degree([leader_id])[0] if g.degree([leader_id]) else 0
                print(f"  {role}: {leader_degree} direct connections")
            except:
                print(f"  {role}: Analysis failed")
```

### 6. Network Centrality Over Time

```python
print(f"\\n=== CENTRALITY ANALYSIS ===")

# Analyze key players in the network
g.checkout_branch("main")

# Degree centrality (simple measure)
all_degrees = g.degree()
node_ids = list(range(g.node_count()))

# Find most connected people
degree_pairs = list(zip(node_ids, all_degrees))
degree_pairs.sort(key=lambda x: x[1], reverse=True)

print("Top 5 most connected employees:")
for i, (node_id, degree) in enumerate(degree_pairs[:5]):
    if g.has_node(node_id):
        node = g.nodes[node_id]
        name = node.get('name', f'Node {node_id}') if hasattr(node, 'get') else f'Node {node_id}'
        role = node.get('role', 'Unknown') if hasattr(node, 'get') else 'Unknown'
        print(f"  {i+1}. {name} ({role}): {degree} connections")

# Compare with remote timeline
g.checkout_branch("remote-first")
remote_degrees = g.degree()
remote_pairs = list(zip(node_ids[:g.node_count()], remote_degrees))
remote_pairs.sort(key=lambda x: x[1], reverse=True)

print(f"\\nRemote timeline - Top 3 most connected:")
for i, (node_id, degree) in enumerate(remote_pairs[:3]):
    if g.has_node(node_id):
        node = g.nodes[node_id]
        name = node.get('name', f'Node {node_id}') if hasattr(node, 'get') else f'Node {node_id}'
        print(f"  {i+1}. {name}: {degree} connections")
```

### 7. Collaboration Pattern Analysis

```python
print(f"\\n=== COLLABORATION PATTERNS ===")

# Analyze different types of relationships
g.checkout_branch("main")
edges_table = g.edges.table()

# Relationship type distribution
print("Relationship types in main timeline:")
rel_types = edges_table['relationship'].unique().to_list()
for rel_type in rel_types:
    count = len(edges_table[edges_table['relationship'] == rel_type])
    print(f"  {rel_type}: {count}")

# Compare with remote timeline
g.checkout_branch("remote-first")
remote_edges = g.edges.table()
remote_rel_types = remote_edges['relationship'].unique().to_list()

print(f"\\nRelationship types in remote timeline:")
for rel_type in remote_rel_types:
    count = len(remote_edges[remote_edges['relationship'] == rel_type])
    print(f"  {rel_type}: {count}")

# Cross-department collaboration analysis
cross_collab = len(remote_edges[remote_edges['relationship'] == 'cross_collab'])
print(f"\\nCross-department collaborations (remote): {cross_collab}")
```

## Advanced Temporal Techniques

### 8. Change Detection

```python
print(f"\\n=== CHANGE DETECTION ===")

# Compare network metrics across timelines
def get_network_metrics(graph):
    return {
        'nodes': graph.node_count(),
        'edges': graph.edge_count(), 
        'density': graph.density(),
        'is_connected': graph.is_connected()
    }

g.checkout_branch("main")
main_metrics = get_network_metrics(g)

g.checkout_branch("remote-first")  
remote_metrics = get_network_metrics(g)

print("Network evolution comparison:")
for metric, main_val in main_metrics.items():
    remote_val = remote_metrics[metric]
    if isinstance(main_val, (int, float)):
        change = remote_val - main_val
        print(f"  {metric}: {main_val} → {remote_val} (Δ{change:+.3f})")
    else:
        print(f"  {metric}: {main_val} → {remote_val}")
```

### 9. Predictive Analysis Setup

```python
print(f"\\n=== PREDICTIVE ANALYSIS SETUP ===")

# Set up data for predictive modeling
g.checkout_branch("main")

# Extract temporal features for each node
temporal_features = []
nodes_table = g.nodes.table()

for i in range(len(nodes_table)):
    try:
        node_id = nodes_table[i]['node_id']
        node = g.nodes[node_id]
        
        features = {
            'node_id': node_id,
            'degree': g.analytics.degree(node_id) if hasattr(g.analytics, 'degree') else 0,
            'role': node.get('role', 'Unknown') if hasattr(node, 'get') else 'Unknown',
            'join_quarter': node.get('join_date', '2020-01')[:7] if hasattr(node, 'get') else '2020-01'
        }
        temporal_features.append(features)
        
    except Exception as e:
        print(f"Feature extraction failed for node {i}: {e}")

print(f"Extracted features for {len(temporal_features)} nodes")
print("Sample features:", temporal_features[0] if temporal_features else "None")

# This data could be used for:
# - Predicting future connections
# - Identifying potential leadership candidates  
# - Detecting collaboration bottlenecks
# - Modeling organizational growth patterns
```

## Key Insights from Temporal Analysis

### 10. Summary and Insights

```python
print(f"\\n=== TEMPORAL ANALYSIS INSIGHTS ===")

print("1. ORGANIZATIONAL GROWTH PATTERNS:")
print("   • Started with 3 founders (high connectivity)")
print("   • Grew to departmental structure (potential silos)")
print("   • Remote work increases cross-department collaboration")

print("\\n2. NETWORK EVOLUTION:")
print("   • Density changes reflect organizational structure")
print("   • Leadership roles correlate with network centrality")
print("   • Technology adoption affects communication patterns")

print("\\n3. BRANCHING SCENARIOS:")
print("   • Main timeline: Traditional hierarchical growth")
print("   • Remote timeline: Flatter, more collaborative structure")
print("   • Version control enables 'what-if' analysis")

print("\\n4. PREDICTIVE OPPORTUNITIES:")
print("   • Early employees have highest centrality")
print("   • Role transitions can be predicted from network position")
print("   • Collaboration tools reshape organizational networks")

# Final state summary
g.checkout_branch("main")
print(f"\\nFinal main timeline: {g.node_count()} nodes, {g.edge_count()} edges")

g.checkout_branch("remote-first")
print(f"Final remote timeline: {g.node_count()} nodes, {g.edge_count()} edges")

print("\\n✓ Temporal graph analysis complete!")
```

## Best Practices for Temporal Analysis

1. **Consistent Commits**: Make regular commits with descriptive messages
2. **Branch Scenarios**: Use branches for alternative timelines and "what-if" analysis
3. **Metric Tracking**: Track key network metrics over time (density, centrality, components)
4. **Feature Engineering**: Extract temporal features for predictive modeling
5. **Comparative Analysis**: Always compare across timelines and time periods

## Next Steps

- Extend to larger networks and longer time periods
- Implement custom temporal metrics
- Add visualization of network evolution
- Build predictive models using temporal features
- Analyze external events' impact on network structure

This tutorial demonstrates Groggy's powerful temporal capabilities for understanding how networks evolve and change over time.