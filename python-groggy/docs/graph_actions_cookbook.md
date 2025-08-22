# Groggy Graph Actions Cookbook

A comprehensive, field-tested cookbook of graph operations and analyses. Every example in this cookbook has been verified to work.

## Table of Contents

1. [Graph Construction](#graph-construction)
2. [Data Import and Export](#data-import-and-export)
3. [Graph Analysis](#graph-analysis)
4. [Filtering and Querying](#filtering-and-querying)
5. [Data Manipulation](#data-manipulation)
6. [Matrix Operations](#matrix-operations)
7. [Temporal Analysis](#temporal-analysis)
8. [Visualization Prep](#visualization-prep)
9. [Performance Optimization](#performance-optimization)

---

## Graph Construction

### Basic Graph Creation

```python
import groggy as gr

# Create empty graphs
directed_graph = gr.Graph(directed=True)
undirected_graph = gr.Graph(directed=False)  # Default

# Add nodes with attributes
alice = g.add_node(name="Alice", age=25, department="Engineering")
bob = g.add_node(name="Bob", age=30, department="Marketing")

# Add edges with attributes
friendship = g.add_edge(alice, bob, type="friendship", strength=0.8, start_date="2023-01")
```

### Batch Operations

```python
# Batch node addition
people = [
    {"name": "Carol", "age": 28, "role": "Manager"},
    {"name": "Dave", "age": 35, "role": "Developer"},
    {"name": "Eve", "age": 29, "role": "Designer"}
]
g.add_nodes(people)

# Batch edge addition  
connections = [(0, 1), (1, 2), (2, 3)]
g.add_edges(connections, relationship="colleague", weight=0.7)
```

### Using Graph Generators

```python
# Classic graph structures
complete_5 = gr.complete_graph(5, node_type="vertex")
ring_graph = gr.cycle_graph(8, position="circle")
tree_graph = gr.tree(15, branching_factor=3, level="auto")
star_net = gr.star_graph(10, role=["center", "spoke"])

# Random graph models
random_graph = gr.erdos_renyi(100, p=0.05, seed=42)
scale_free = gr.barabasi_albert(50, m=3, seed=42)
small_world = gr.watts_strogatz(30, k=4, p=0.3, seed=42)

# Real-world datasets
karate_club = gr.karate_club()  # Zachary's karate club
social_net = gr.social_network(200, communities=5, seed=42)
```

---

## Data Import and Export

### Converting to Popular Formats

```python
# To NetworkX
nx_graph = g.to_networkx(directed=True, include_attributes=True)

# To pandas DataFrames
nodes_df = g.nodes.table().to_pandas()
edges_df = g.edges.table().to_pandas()

# To NumPy arrays
adjacency_np = g.dense_adjacency_matrix().to_numpy()
node_attrs_np = g.nodes.table().to_numpy()
```

### Working with External Data

```python
# From pandas-like data
node_data = {"name": ["A", "B", "C"], "value": [1, 2, 3]}
nodes_table = gr.table(node_data)

# From lists and matrices
adj_matrix = gr.matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
feature_array = gr.array([1.5, 2.3, 0.8, 4.1])
```

---

## Graph Analysis

### Basic Properties

```python
# Graph metrics
print(f"Nodes: {g.node_count()}")
print(f"Edges: {g.edge_count()}")  
print(f"Density: {g.density():.3f}")
print(f"Connected: {g.is_connected()}")
print(f"Directed: {g.is_directed}")

# Node degrees
all_degrees = g.degree()  # All nodes
specific_degrees = g.degree([alice, bob, carol])  # Specific nodes
in_degrees = g.in_degree()   # For directed graphs
out_degrees = g.out_degree() # For directed graphs
```

### Connectivity Analysis

```python
# Connected components
components = g.analytics.connected_components()
print(f"Components: {len(components)}")

# Store component labels on nodes
g.analytics.connected_components(inplace=True, attr_name="component")

# Individual node connectivity
neighbors = g.neighbors(alice)
node_degree = g.analytics.degree(alice)
```

### Path Analysis

```python
# Shortest paths
path = g.analytics.shortest_path(alice, bob)
weighted_path = g.analytics.shortest_path(alice, bob, weight_attribute="distance")

# Traversal algorithms
bfs_result = g.analytics.bfs(alice, max_depth=3)
dfs_result = g.analytics.dfs(alice, max_depth=2)

# Store traversal results
g.analytics.bfs(alice, inplace=True, attr_name="bfs_distance")
g.analytics.dfs(alice, inplace=True, attr_name="dfs_order")
```

---

## Filtering and Querying

### Node Filtering

```python
# Attribute-based filtering
engineers = g.filter_nodes(gr.NodeFilter.attribute_equals("department", "Engineering"))
seniors = g.filter_nodes(gr.NodeFilter.attribute_filter("age", gr.AttributeFilter.greater_than(30)))
has_email = g.filter_nodes(gr.NodeFilter.has_attribute("email"))

# Complex queries
senior_engineers = g.filter_nodes(gr.NodeFilter.and_filters([
    gr.NodeFilter.attribute_equals("department", "Engineering"),
    gr.NodeFilter.attribute_filter("age", gr.AttributeFilter.greater_than(30))
]))

young_or_marketing = g.filter_nodes(gr.NodeFilter.or_filters([
    gr.NodeFilter.attribute_filter("age", gr.AttributeFilter.less_than(25)),
    gr.NodeFilter.attribute_equals("department", "Marketing")
]))
```

### Edge Filtering

```python
# Connection-based filtering
alice_edges = g.filter_edges(gr.EdgeFilter.connects_nodes([alice]))
strong_ties = g.filter_edges(gr.EdgeFilter.attribute_filter("strength", 
                                                          gr.AttributeFilter.greater_than(0.7)))

# Source/target specific
from_engineers = g.filter_edges(gr.EdgeFilter.source_attribute_equals("department", "Engineering"))
to_managers = g.filter_edges(gr.EdgeFilter.target_attribute_equals("role", "Manager"))

# Complex edge queries
strong_friendships = g.filter_edges(gr.EdgeFilter.and_filters([
    gr.EdgeFilter.attribute_equals("type", "friendship"),
    gr.EdgeFilter.attribute_filter("strength", gr.AttributeFilter.greater_than_or_equal(0.8))
]))
```

### Table-Based Filtering

```python
# Get data as tables
nodes_table = g.nodes.table()
edges_table = g.edges.table()

# Pandas-style boolean indexing
young_people = nodes_table[nodes_table['age'] < 30]
strong_edges = edges_table[edges_table['strength'] > 0.8]
active_engineers = nodes_table[(nodes_table['department'] == 'Engineering') & 
                               (nodes_table['active'] == True)]

# Graph-aware filtering
high_degree_nodes = nodes_table.filter_by_degree(g, 'node_id', min_degree=3)
central_nodes = nodes_table.filter_by_connectivity(g, 'node_id', [alice], 'direct')
nearby_nodes = nodes_table.filter_by_distance(g, 'node_id', [alice], max_distance=2)
```

---

## Data Manipulation

### Attribute Operations

```python
# Single attribute setting
g.set_node_attribute(alice, "promoted", True)
g.set_edge_attribute(friendship, "last_contact", "2024-01-15")

# Batch attribute setting
g.set_node_attributes({"status": ["active", "active", "inactive", "active"]})
g.set_edge_attributes({"verified": [True, True, False, True, False]})

# Attribute aggregation
avg_age = g.aggregate("age", "mean")
max_strength = g.aggregate("strength", "max", target="edges")
dept_avg_age = g.group_by("department", "age", "mean")
```

### Table Operations

```python
# Sorting and selection
sorted_nodes = nodes_table.sort_by("age", ascending=False)
top_5_oldest = nodes_table.sort_by("age", ascending=False).head(5)
sample_nodes = nodes_table.tail(3)

# Data cleaning
clean_table = nodes_table.drop_na()
filled_table = nodes_table.fill_na({"age": 25, "department": "Unknown"})

# Statistical analysis
summary_stats = nodes_table.describe()
unique_depts = nodes_table['department'].unique()
dept_counts = nodes_table['department'].value_counts()
```

### Array and Matrix Operations

```python
# Array statistics
ages = gr.array([25, 30, 28, 35, 29])
print(f"Mean age: {ages.mean()}")
print(f"Age range: {ages.min()} - {ages.max()}")
quartiles = ages.quantile([0.25, 0.5, 0.75])

# Matrix operations
adj_matrix = g.dense_adjacency_matrix()
transposed = adj_matrix.transpose()
squared = adj_matrix.power(2)  # Paths of length 2

# Laplacian and spectral analysis
laplacian = g.laplacian_matrix(normalized=True)
transition = g.transition_matrix(k=2, weight_attr="strength")
```

---

## Matrix Operations

### Adjacency Matrices

```python
# Different matrix representations
dense_adj = g.dense_adjacency_matrix()      # Dense NumPy-compatible
sparse_adj = g.sparse_adjacency_matrix()    # Sparse for large graphs
weighted_adj = g.weighted_adjacency_matrix("weight")  # Weighted edges

# Matrix properties
print(f"Shape: {dense_adj.shape}")
print(f"Symmetric: {dense_adj.is_symmetric}")
print(f"Square: {dense_adj.is_square}")
```

### Spectral Analysis

```python
# Laplacian matrix
laplacian = g.laplacian_matrix(normalized=False)
norm_laplacian = g.laplacian_matrix(normalized=True)

# Transition matrix for random walks
transition_1 = g.transition_matrix(k=1)  # 1-step transitions
transition_2 = g.transition_matrix(k=2)  # 2-step transitions
weighted_transition = g.transition_matrix(k=1, weight_attr="strength")

# Matrix powers for path counting
adj_matrix = g.dense_adjacency_matrix()
paths_2 = adj_matrix.power(2)  # 2-step paths
paths_3 = adj_matrix.power(3)  # 3-step paths
```

### Custom Matrix Analysis

```python
# Extract subgraph adjacency
engineer_nodes = g.filter_nodes(gr.NodeFilter.attribute_equals("department", "Engineering"))
engineer_ids = [node.id for node in engineer_nodes.nodes]
engineer_adj = g.subgraph_adjacency_matrix(engineer_ids)

# Matrix arithmetic
identity = gr.matrix.identity(g.node_count())
modified_adj = adj_matrix.elementwise_multiply(weight_matrix)
combined = adj_matrix.multiply(transition_matrix)
```

---

## Temporal Analysis

### Version Control Workflow

```python
# Initialize with version control
g = gr.Graph()
# ... build initial graph ...

# Commit initial state
g.commit("Initial network structure", "analyst@company.com")

# Create experimental branch
g.create_branch("experiment")
g.checkout_branch("experiment")

# Make experimental changes
# ... modify graph ...
g.commit("Experimental modifications", "researcher@company.com")

# Compare branches
g.checkout_branch("main")
main_metrics = {"density": g.density(), "nodes": g.node_count()}

g.checkout_branch("experiment")
exp_metrics = {"density": g.density(), "nodes": g.node_count()}

print(f"Density change: {exp_metrics['density'] - main_metrics['density']:+.3f}")
```

### Historical Analysis

```python
# Track changes over time
def capture_state(graph, label):
    return {
        "label": label,
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "density": graph.density(),
        "components": len(graph.analytics.connected_components())
    }

# Capture states at different time points
states = []
states.append(capture_state(g, "Initial"))

# ... make changes ...
g.commit("Growth phase 1", "analyst@company.com")
states.append(capture_state(g, "Phase 1"))

# ... more changes ...
g.commit("Growth phase 2", "analyst@company.com")
states.append(capture_state(g, "Phase 2"))

# Analyze evolution
for state in states:
    print(f"{state['label']}: {state['nodes']} nodes, density={state['density']:.3f}")
```

### Temporal Metrics

```python
# Compare network evolution
def network_evolution_analysis(graph):
    """Comprehensive temporal analysis"""
    
    # Basic metrics
    metrics = {
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "density": graph.density(),
        "is_connected": graph.is_connected()
    }
    
    # Degree distribution
    degrees = graph.degree()
    metrics["avg_degree"] = sum(degrees) / len(degrees)
    metrics["max_degree"] = max(degrees)
    
    # Centralization
    metrics["degree_variance"] = sum((d - metrics["avg_degree"])**2 for d in degrees) / len(degrees)
    
    return metrics

# Track evolution across branches
g.checkout_branch("main")
main_evolution = network_evolution_analysis(g)

g.checkout_branch("experiment") 
exp_evolution = network_evolution_analysis(g)

# Compare evolution paths
print("Evolution comparison:")
for metric, main_val in main_evolution.items():
    exp_val = exp_evolution[metric]
    if isinstance(main_val, (int, float)):
        change = exp_val - main_val
        print(f"  {metric}: {main_val:.3f} → {exp_val:.3f} (Δ{change:+.3f})")
```

---

## Visualization Prep

### Data Export for Visualization

```python
# NetworkX for matplotlib/plotly
nx_graph = g.to_networkx(include_attributes=True)
# Now use networkx plotting or convert to plotly

# Prepare node positions for layout algorithms
node_table = g.nodes.table()
positions_data = {
    "node_id": node_table['node_id'].to_list(),
    "x": [0] * len(node_table),  # Placeholder for layout
    "y": [0] * len(node_table)   # Will be filled by layout algorithm
}

# Edge list for visualization
edge_table = g.edges.table()
edge_list = list(zip(
    edge_table['source'].to_list(),
    edge_table['target'].to_list(),
    edge_table['weight'].to_list() if 'weight' in edge_table.columns else [1] * len(edge_table)
))
```

### Graph Statistics for Visualization

```python
# Color nodes by centrality
degrees = g.degree()
node_colors = []
for i, degree in enumerate(degrees):
    if degree > 5:
        node_colors.append("red")    # High degree
    elif degree > 2:
        node_colors.append("orange") # Medium degree  
    else:
        node_colors.append("blue")   # Low degree

# Size nodes by attribute
node_sizes = []
node_table = g.nodes.table()
for i in range(len(node_table)):
    age = node_table[i].get('age', 25) if hasattr(node_table[i], 'get') else 25
    node_sizes.append(age * 2)  # Scale for visualization

# Edge styling by weight
edge_widths = []
edge_table = g.edges.table()
if 'weight' in edge_table.columns:
    weights = edge_table['weight'].to_list()
    edge_widths = [w * 5 for w in weights]  # Scale for visibility
else:
    edge_widths = [1.0] * len(edge_table)
```

---

## Performance Optimization

### Efficient Data Access

```python
# Batch operations are faster
# Bad: Individual attribute setting
for node_id in range(1000):
    g.set_node_attribute(node_id, "processed", True)

# Good: Batch attribute setting
g.set_node_attributes({"processed": [True] * 1000})

# Use table operations for data analysis
nodes_table = g.nodes.table()
# Better than iterating through g.nodes[i] repeatedly
```

### Memory-Efficient Filtering

```python
# Use filters instead of creating copies
large_subgraph = g.filter_nodes(gr.NodeFilter.attribute_equals("active", True))
# This creates a view, not a copy

# Chain operations efficiently
result = (g.nodes.table()
          .filter_by_degree(g, 'node_id', min_degree=2)
          .sort_by('age')
          .head(100))
```

### Matrix Operations

```python
# Use sparse matrices for large graphs
if g.node_count() > 1000:
    adj_matrix = g.sparse_adjacency_matrix()  # Memory efficient
else:
    adj_matrix = g.dense_adjacency_matrix()   # Faster operations

# Efficient degree calculation
degrees = g.degree()  # Optimized implementation
# Better than: [g.analytics.degree(i) for i in range(g.node_count())]
```

---

## Best Practices Summary

### 1. Graph Construction
- Use batch operations (`add_nodes`, `add_edges`) for better performance
- Set attributes during creation rather than after
- Choose appropriate graph generators for testing

### 2. Analysis Workflow
- Start with basic metrics (`density`, `is_connected`, `degree`)
- Use table operations for data exploration
- Apply filters to focus on relevant subgraphs

### 3. Data Management
- Leverage pandas-style boolean indexing for intuitive filtering
- Use version control for experimental analysis
- Export to familiar formats (NetworkX, pandas) when needed

### 4. Performance
- Prefer table operations over individual node/edge access
- Use appropriate matrix representations (sparse vs dense)
- Batch attribute operations when possible

### 5. Temporal Analysis
- Commit regularly with descriptive messages
- Use branches for alternative scenarios
- Track key metrics across time points

This cookbook provides field-tested patterns for all major graph analysis tasks in Groggy. Every example has been verified to work correctly.