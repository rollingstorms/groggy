# Groggy Python API Primer: High-Level Graph Processing Made Simple

## Table of Contents
1. [Overview & Philosophy](#overview--philosophy)  
2. [Python vs Rust: Why Both?](#python-vs-rust-why-both)
3. [Pythonic Design Principles](#pythonic-design-principles)
4. [Quick Start Guide](#quick-start-guide)
5. [Core Python API](#core-python-api)
6. [Advanced Python Features](#advanced-python-features)
7. [Real-World Workflows](#real-world-workflows)
8. [Python-Specific Performance](#python-specific-performance)
9. [Ecosystem Integration](#ecosystem-integration)

---

## Overview & Philosophy

The **Groggy Python API** provides a high-level, intuitive interface to the powerful Rust graph engine. While the Rust implementation focuses on raw performance and memory efficiency, the Python API emphasizes **developer productivity**, **ecosystem integration**, and **ease of use**.

### Core Value Proposition
- **Pythonic Interface**: Natural, readable syntax that follows Python conventions
- **Zero Boilerplate**: Complex graph operations in single lines of code
- **Ecosystem Native**: Seamless integration with pandas, numpy, networkx, and ML libraries
- **Flexible Input Formats**: Accept data in any format - dicts, lists, DataFrames, or raw values
- **Rust Performance**: All heavy computation happens in optimized Rust code

---

## Python vs Rust: Why Both?

### When to Use the Python API
✅ **Data Science & Analytics**: Graph analysis in Jupyter notebooks  
✅ **Rapid Prototyping**: Quickly explore graph algorithms and patterns  
✅ **Business Applications**: Build graph-powered applications with minimal code  
✅ **Ecosystem Integration**: Work with existing Python data science stack  
✅ **Interactive Analysis**: Explore large networks interactively  

### When to Use the Rust Core
✅ **Performance Critical**: Maximum throughput for large-scale processing  
✅ **Memory Constrained**: Embedded systems or tight memory budgets  
✅ **Custom Algorithms**: Implement specialized graph algorithms  
✅ **Library Development**: Build other language bindings or frameworks  

### The Best of Both Worlds
```python
import groggy as gr
import pandas as pd
import numpy as np

# Python: Easy data import and setup
df = pd.read_csv('network_data.csv')
g = gr.Graph()
# groggy doesn't store string ids, so we need to map string ids to internal int ids
node_mapping = g.add_nodes(df.to_dict('records'), uid_key='employee_id')

# Python: Intuitive graph operations  
engineers = g.filter_nodes('department == "Engineering"')
# we can store labels on each component inplace
components = engineers.connected_components(inplace=True)
# or add them manually with kwargs
components = engineers.connected_components()
for i, component in enumerate(components):
    component.nodes[:].set(component_id=i)
    print(f"Team {i}: {len(component.nodes)} people")

# Python: Seamless ecosystem integration
result_df = pd.DataFrame({
    'node_id': g.nodes.ids,
    'name': g.nodes['name'], 
    'department': g.nodes['department']
})
```

---

## Pythonic Design Principles

### 1. **Natural Syntax Over Performance**
```python
# Pythonic: Reads like natural language
engineers = g.filter_nodes('role == "engineer"')
senior_engineers = engineers.filter_nodes('experience > 5')
```

### 2. **Flexible Input Types**  
```python
# Accept any reasonable format
g.add_node(name="Alice", age=30, role="engineer")  # ✅ kwargs
g.add_node({"name": "Bob", "age": 35})             # ✅ dict
g.set_node_attr(0, "salary", 120000)              # ✅ individual
# or set in node view
g.nodes[0].set(age=30)
g.nodes[0].set({"age": 30, "role": "engineer"})
g.nodes[1].update({"age": 35, "role": "engineer"})

# Bulk operations with mixed formats
nodes_data = [
    {"id": "alice", "name": "Alice", "dept": "Engineering"},
    {"id": "bob", "name": "Bob", "dept": "Sales"}
]
g.add_nodes(nodes_data, uid_key="id")  # ✅ Auto ID mapping

g.add_edge(0, 1, relationship="collaborates", strength=0.8)
g.add_edge("Alice", "Bob", uid_key="name", type="friendship")  # String IDs
edge_data = [
    (0, 1, {"relationship": "collaborates", "strength": 0.8}),
    {"source": "alice", "target": "bob", "type": "friendship"}
]
g.add_edges(edge_data, uid_key="id")
```

### 3. **Intelligent Defaults**
```python
# No configuration needed for common cases
g = gr.Graph()  # ✅ Sensible defaults

# Optional version control when needed
g = gr.Graph(version_control=True)  # ✅ Opt-in complexity
```

### 4. **Property Access Over Method Calls**
```python
# Python: Property-style access
len(g.nodes)        # ✅ Pythonic
g.nodes[0]          # ✅ Dictionary-like
g.edges.source      # ✅ Column access

# vs Method-heavy APIs
g.node_count()      # ❌ Less Pythonic
g.get_node(0)       # ❌ Verbose
```

### 5. **String Queries for Readability**
```python
# Python: Natural language queries
high_performers = g.filter_nodes('performance > 0.8 and tenure > 2')
recent_hires = g.filter_nodes('start_date > "2024-01-01"')
```

---

## Quick Start Guide

### Installation
```bash
pip install groggy  # From PyPI (future)
# or 
pip install -e python-groggy/  # Development install
```

### 30-Second Graph Analysis
```python
import groggy as gr

# Create graph with real data
g = gr.Graph()

# Add nodes with attributes (multiple ways)
alice = g.add_node(name="Alice", role="Engineer", salary=120000)
bob = g.add_node(name="Bob", role="Manager", salary=140000) 
charlie = g.add_node(name="Charlie", role="Engineer", salary=100000)

# Add relationships
g.add_edge(alice, bob, relationship="reports_to", strength=0.9)
g.add_edge(alice, charlie, relationship="collaborates", strength=0.7)

# Instant analysis with string queries
engineers = g.filter_nodes('role == "Engineer"')
print(f"Engineers: {len(engineers.nodes)}")

high_earners = g.filter_nodes('salary > 110000')  
print(f"High earners: {len(high_earners.nodes)}")

# Graph algorithms made simple
components = g.connected_components()
print(f"Connected groups: {len(components)}")

# Network analysis
centrality = g.betweenness_centrality()
most_central = max(centrality.items(), key=lambda x: x[1])
print(f"Most central person: node {most_central[0]} (score: {most_central[1]:.3f})")
```

---

## Core Python API

### Graph Construction

#### Simple Node/Edge Creation
```python
g = gr.Graph()

# Individual nodes with attributes
node_id = g.add_node(name="Alice", age=30, department="Engineering")

# Bulk node creation
employee_data = [
    {"name": "Alice", "age": 30, "dept": "Engineering"},
    {"name": "Bob", "age": 35, "dept": "Sales"}
]
node_mapping = g.add_nodes(employee_data, uid_key="name")  
# Returns: {"Alice": 0, "Bob": 1}

# Flexible edge creation
g.add_edge(0, 1, relationship="collaborates", strength=0.8)
g.add_edge("Alice", "Bob", uid_key="name", type="friendship")  # String IDs

# Bulk edge creation with different formats
edge_data = [
    (0, 1, {"type": "reports_to"}),  # Tuple format
    {"source": "Alice", "target": "Bob", "strength": 0.9}  # Dict format
]
g.add_edges(edge_data, node_mapping=node_mapping)
```

#### Smart Data Import
```python
# From pandas DataFrame
import pandas as pd
df = pd.read_csv('employees.csv')
node_mapping = g.add_nodes(df.to_dict('records'), uid_key='employee_id')

# From NetworkX (future feature)
import networkx as nx
nx_graph = nx.karate_club_graph()
g = gr.from_networkx(nx_graph)

# From JSON/dict structures
org_data = {
    "nodes": [{"id": "alice", "role": "engineer"}],
    "edges": [{"source": "alice", "target": "bob"}]
}
g = gr.from_dict(org_data, node_uid_key="id")
```

### Graph Querying & Filtering

#### String-Based Queries (Most Pythonic)
```python
# Simple attribute filtering
engineers = g.filter_nodes('role == "engineer"')
high_earners = g.filter_nodes('salary > 100000')
new_employees = g.filter_nodes('start_date > "2024-01-01"')

# Numeric and string operations
experienced = g.filter_nodes('years_experience >= 5')
senior_titles = g.filter_nodes('title.contains("Senior")')  # Future

# Edge filtering by attributes
strong_connections = g.filter_edges('strength > 0.8')
recent_interactions = g.filter_edges('last_contact > "2024-06-01"')

# Edge topology filtering
outgoing_from_alice = g.filter_edges('source == 0')  # From node 0
incoming_to_bob = g.filter_edges('target == 1')      # To node 1
```

#### Chainable Subgraph Operations
```python
# Multi-step filtering with method chaining
senior_engineers = (g.filter_nodes('role == "engineer"')
                     .filter_nodes('years_experience >= 5') 
                     .filter_nodes('performance_score > 0.8'))

# Subgraphs maintain full Graph API
eng_components = senior_engineers.connected_components()
eng_centrality = senior_engineers.betweenness_centrality()

# Extract data from any subgraph level
names = senior_engineers.nodes['name']        # List of names
salaries = senior_engineers.nodes['salary']   # List of salaries
avg_salary = sum(salaries) / len(salaries)
```

#### Object-Based Filtering (Advanced)
```python
from groggy import NodeFilter, EdgeFilter, AttributeFilter

# Explicit filter construction for complex cases
filter_obj = NodeFilter.attribute_filter("salary", AttributeFilter.greater_than(100000))
high_earners = g.filter_nodes(filter_obj)

# Logical combinations
complex_filter = NodeFilter.and_([
    NodeFilter.attribute_equals("department", "Engineering"),
    NodeFilter.attribute_greater_than("experience", 3)
])
results = g.filter_nodes(complex_filter)
```

### Graph Analysis & Algorithms

#### Network Metrics
```python
# Basic statistics
print(f"Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
print(f"Density: {g.density():.3f}")
print(f"Average degree: {g.average_degree():.1f}")

# Centrality measures
betweenness = g.betweenness_centrality()
pagerank = g.pagerank(damping=0.85, iterations=100)
closeness = g.closeness_centrality()

# Find most important nodes
most_central = max(betweenness.items(), key=lambda x: x[1])
highest_pagerank = max(pagerank.items(), key=lambda x: x[1])
```

#### Structural Analysis
```python
# Connected components
components = g.connected_components()
largest_component = max(components, key=lambda c: len(c.nodes))
print(f"Largest component: {len(largest_component.nodes)} nodes")

# Graph traversal
visited = g.bfs(start_node=0, max_depth=3)
dfs_tree = g.dfs(start_node=0, max_depth=None)

# Path finding
path = g.shortest_path(source=0, target=5)
all_paths = g.all_shortest_paths(source=0, target=5)
```

#### Community Detection
```python
# Built-in community algorithms
communities = g.detect_communities(method="modularity")
community_sizes = [len(comm.nodes) for comm in communities]

# Modularity scoring
modularity = g.modularity(communities)
print(f"Modularity: {modularity:.3f}")
```

---

## Advanced Python Features

### Property-Based Access

#### Node and Edge Properties
```python
# Access nodes as collections
g.nodes              # NodesAccessor object
len(g.nodes)         # Number of nodes
g.nodes.ids          # List of all node IDs

# Individual node access
g.nodes[0]           # All attributes for node 0
g.nodes[0]['name']   # Specific attribute
g.nodes[0].name      # Property access (if implemented)

# Batch node access
g.nodes[[0, 1, 2]]   # Subgraph of specific nodes
g.nodes[0:5]         # Subgraph of node range

# Column-wise access
all_names = g.nodes['name']      # List of all names
all_salaries = g.nodes['salary'] # List of all salaries
```

#### Attribute Collections
```python
# Bulk attribute access
g.attributes["department"]  # All department values
g.attributes["salary"]      # All salary values

# Statistical operations on columns
import numpy as np
salary_array = np.array(g.attributes["salary"])
print(f"Average salary: ${salary_array.mean():,.0f}")
print(f"Salary std dev: ${salary_array.std():,.0f}")

# Filter using numpy/pandas operations
high_salary_mask = salary_array > salary_array.mean() + salary_array.std()
high_salary_nodes = np.array(g.nodes.ids)[high_salary_mask]
```

### Fluent Updates & Method Chaining

#### Single Node Updates
```python
# Fluent attribute updates
g.nodes[0].set(promoted=True, new_salary=130000) \
          .set(promotion_date="2024-07-01") \
          .update(review_complete=True)

# Dictionary-based updates  
g.nodes[0].set({
    'performance_rating': 'excellent',
    'bonus_eligible': True,
    'next_review': '2025-01-01'
})
```

#### Batch Updates
```python
# Update multiple nodes at once
engineers = g.filter_nodes('role == "engineer"')
engineers.nodes.set(department='Engineering', certified=True)

# Range-based updates
g.nodes[0:10].set(batch_processed=True, process_date="2024-07-01")

# Conditional batch updates
high_performers = g.filter_nodes('performance_score > 0.9')
high_performers.nodes.set(promotion_track='leadership')
```

### Algorithm Integration

#### In-Place Attribute Generation
```python
# Algorithms that add attributes to nodes/edges
g.connected_components(inplace=True, attr_name="component_id")
g.pagerank(inplace=True, attr_name="influence_score")
g.betweenness_centrality(inplace=True, attr_name="bridge_score")

# Check results
print(g.nodes[0])  # Now includes component_id, influence_score, bridge_score

# Use algorithm results in subsequent queries
influential = g.filter_nodes('influence_score > 0.1')
bridges = g.filter_nodes('bridge_score > 0.05')
```

#### Algorithm Chaining
```python
# Chain algorithms with filtering
large_components = g.connected_components()
largest = max(large_components, key=lambda c: len(c.nodes))

# Run algorithms on subgraphs
central_in_largest = largest.betweenness_centrality()
influential_in_largest = largest.pagerank()

# Multi-level analysis
for i, component in enumerate(g.connected_components()):
    if len(component.nodes) > 5:  # Only analyze large components
        communities = component.detect_communities()
        print(f"Component {i}: {len(communities)} communities")
```

---

## Real-World Workflows

### Corporate Network Analysis
```python
import groggy as gr
import pandas as pd

# Load employee data
employees_df = pd.read_csv('employees.csv')
connections_df = pd.read_csv('collaborations.csv')

# Build graph
g = gr.Graph()
employee_map = g.add_nodes(employees_df.to_dict('records'), uid_key='employee_id')
g.add_edges(connections_df.to_dict('records'), node_mapping=employee_map)

# Department-level analysis
departments = g.group_by_attribute('department')
for dept_subgraph in departments:
    dept_name = dept_subgraph.nodes[0]['department']
    
    # Analyze team structure
    components = dept_subgraph.connected_components()
    influence = dept_subgraph.pagerank()
    
    # Find department leaders
    most_influential = max(influence.items(), key=lambda x: x[1])
    leader_name = dept_subgraph.nodes[most_influential[0]]['name']
    
    print(f"{dept_name}: {len(dept_subgraph.nodes)} people, "
          f"{len(components)} teams, leader: {leader_name}")

# Cross-department collaboration
cross_dept_edges = g.filter_edges(lambda e: 
    g.nodes[e['source']]['department'] != g.nodes[e['target']]['department']
)
print(f"Cross-department connections: {len(cross_dept_edges.edges)}")
```

### Social Network Mining
```python
import groggy as gr
import networkx as nx

# Load social network data 
nx_graph = nx.read_gml('social_network.gml')
g = gr.from_networkx(nx_graph)

# Find influential users
g.pagerank(inplace=True, attr_name='influence')
g.betweenness_centrality(inplace=True, attr_name='bridging')

# Identify user types based on network position
influencers = g.filter_nodes('influence > 0.01')  # Top 1% by influence
bridges = g.filter_nodes('bridging > 0.05')      # High bridging score
isolates = g.filter_nodes(lambda n: g.degree(n) <= 2)

# Community detection and analysis
communities = g.detect_communities()
community_sizes = [len(comm.nodes) for comm in communities]

print(f"Network: {len(g.nodes)} users, {len(g.edges)} connections")
print(f"Communities: {len(communities)} (avg size: {sum(community_sizes)/len(community_sizes):.1f})")
print(f"Influencers: {len(influencers.nodes)}, Bridges: {len(bridges.nodes)}")

# Export results for visualization
influence_df = pd.DataFrame({
    'user_id': g.nodes.ids,
    'influence': g.nodes['influence'],
    'bridging': g.nodes['bridging'],
    'community': g.nodes['community_id']  # From community detection
})
```

### Knowledge Graph Construction
```python
import groggy as gr
from typing import Dict, List

class KnowledgeGraph:
    def __init__(self):
        self.graph = gr.Graph(version_control=True)
        self.entity_mapping = {}
    
    def add_entity(self, entity_id: str, entity_type: str, **attributes):
        """Add entity with type and attributes"""
        node_id = self.graph.add_node(
            entity_id=entity_id,
            entity_type=entity_type,
            **attributes
        )
        self.entity_mapping[entity_id] = node_id
        return node_id
    
    def add_relation(self, subject: str, predicate: str, object: str, **metadata):
        """Add relationship between entities"""
        subj_node = self.entity_mapping[subject]  
        obj_node = self.entity_mapping[object]
        
        return self.graph.add_edge(
            subj_node, obj_node,
            predicate=predicate,
            **metadata
        )
    
    def query_relations(self, entity_type: str = None, predicate: str = None):
        """Query the knowledge graph"""
        if entity_type:
            entities = self.graph.filter_nodes(f'entity_type == "{entity_type}"')
        else:
            entities = self.graph
            
        if predicate:
            relations = entities.filter_edges(f'predicate == "{predicate}"')
        else:
            relations = entities
            
        return relations
    
    def save_version(self, description: str):
        """Save current state"""
        return self.graph.commit(description, "knowledge_engineer")

# Usage
kg = KnowledgeGraph()

# Add entities
kg.add_entity("person:alice", "Person", name="Alice", age=30)
kg.add_entity("company:acme", "Company", name="Acme Corp", industry="Tech")
kg.add_entity("product:widget", "Product", name="Super Widget", price=99.99)

# Add relationships
kg.add_relation("person:alice", "works_for", "company:acme", since="2020-01-01")
kg.add_relation("person:alice", "developed", "product:widget", role="lead_engineer")
kg.add_relation("company:acme", "produces", "product:widget", primary_product=True)

# Query and analyze
employees = kg.query_relations(entity_type="Person", predicate="works_for")
products = kg.query_relations(entity_type="Product")

print(f"Employees: {len(employees.nodes)}")
print(f"Products: {len(products.nodes)}")

# Save version
kg.save_version("Initial knowledge base")
```

---

## Python-Specific Performance

### Performance Best Practices

#### Batch Operations
```python
# ✅ GOOD: Batch operations are fastest
nodes_data = [{"name": f"user_{i}", "value": i} for i in range(1000)]
node_ids = g.add_nodes(nodes_data)

# ❌ AVOID: Individual operations in loops
for i in range(1000):
    g.add_node(name=f"user_{i}", value=i)  # 1000x slower
```

#### String Queries vs Filter Objects
```python
# ✅ GOOD: String queries are optimized
engineers = g.filter_nodes('role == "engineer"')

# ✅ ALSO GOOD: Filter objects for complex logic
from groggy import NodeFilter
complex_filter = NodeFilter.and_([
    NodeFilter.attribute_equals("role", "engineer"),
    NodeFilter.attribute_greater_than("experience", 5)
])
senior_engineers = g.filter_nodes(complex_filter)

# ❌ AVOID: Python lambda functions (can't optimize)
engineers = g.filter_nodes(lambda n: n['role'] == 'engineer')  # Slower
```

#### Memory-Efficient Column Access
```python
# ✅ GOOD: Direct column access (no Python objects created)
salaries = g.attributes["salary"]  # Returns numpy-compatible list
avg_salary = sum(salaries) / len(salaries)

# ❌ AVOID: Individual node access in loops  
total = 0
for node_id in g.nodes.ids:
    total += g.nodes[node_id]['salary']  # Creates Python objects
avg_salary = total / len(g.nodes)
```

### Performance Monitoring
```python
import time
import groggy as gr

# Time graph operations
start = time.time()
g = gr.Graph()
g.add_nodes(10000)
g.add_random_edges(50000)  # Future feature
construction_time = time.time() - start

start = time.time()
components = g.connected_components()
analysis_time = time.time() - start

print(f"Construction: {construction_time:.2f}s")
print(f"Analysis: {analysis_time:.2f}s")
print(f"Performance: {len(g.nodes) / analysis_time:.0f} nodes/sec")

# Memory usage
memory_stats = g.memory_statistics()
print(f"Memory: {memory_stats.total_bytes / 1024 / 1024:.1f} MB")
```

---

## Ecosystem Integration

### Pandas Integration
```python
import groggy as gr
import pandas as pd

# DataFrame → Graph
df = pd.read_csv('network_data.csv')
g = gr.Graph()
node_mapping = g.add_nodes(df.to_dict('records'), uid_key='user_id')

# Graph → DataFrame  
results_df = pd.DataFrame({
    'node_id': g.nodes.ids,
    'name': g.nodes['name'],
    'department': g.nodes['department'],
    'centrality': g.betweenness_centrality()
})

# Analyze with pandas
dept_centrality = results_df.groupby('department')['centrality'].mean()
print(dept_centrality)
```

### NumPy Integration
```python
import groggy as gr
import numpy as np

g = gr.Graph()
# ... build graph ...

# Extract numeric data as numpy arrays
ages = np.array(g.attributes['age'])
salaries = np.array(g.attributes['salary'])
performance = np.array(g.attributes['performance_score'])

# NumPy-powered analysis
high_performers = np.where((performance > np.percentile(performance, 90)) & 
                          (salaries < np.percentile(salaries, 75)))[0]

# Use results to filter graph
undervalued_stars = g.nodes[high_performers]
undervalued_stars.set(promotion_eligible=True)
```

### Scikit-learn Integration
```python
import groggy as gr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

g = gr.Graph()
# ... build employee network ...

# Extract features for ML
features = np.column_stack([
    g.attributes['age'],
    g.attributes['salary'], 
    g.attributes['performance_score'],
    [g.degree(node) for node in g.nodes.ids]  # Network features
])

# ML clustering
scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(X)

# Add ML results back to graph
g.nodes.set_batch('ml_cluster', dict(zip(g.nodes.ids, clusters)))

# Analyze clusters using graph operations
for cluster_id in range(4):
    cluster_nodes = g.filter_nodes(f'ml_cluster == {cluster_id}')
    avg_degree = np.mean([g.degree(n) for n in cluster_nodes.nodes.ids])
    print(f"Cluster {cluster_id}: {len(cluster_nodes.nodes)} people, "
          f"avg connections: {avg_degree:.1f}")
```

### Visualization Integration
```python
import groggy as gr
import matplotlib.pyplot as plt
import networkx as nx

# Convert to NetworkX for visualization
g = gr.Graph()
# ... build graph ...

nx_graph = g.to_networkx()  # Future feature

# Create layout with NetworkX
pos = nx.spring_layout(nx_graph)

# Visualize with node attributes from Groggy
node_colors = [g.nodes[node]['performance_score'] for node in nx_graph.nodes()]
node_sizes = [g.degree(node) * 100 for node in nx_graph.nodes()]

plt.figure(figsize=(12, 8))
nx.draw(nx_graph, pos, 
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=True,
        cmap='viridis')
plt.colorbar(label='Performance Score')
plt.title('Employee Network - Size=Connections, Color=Performance')
plt.show()
```

---

## Next Steps

### Learning Path
1. **Start Simple**: Try the Quick Start examples
2. **Explore Filtering**: Master string queries and subgraph operations  
3. **Add Algorithms**: Experiment with centrality and community detection
4. **Scale Up**: Work with larger datasets and performance optimization
5. **Integrate**: Combine with your existing Python data science workflow

### Advanced Topics
- **Custom Algorithms**: Implement domain-specific graph algorithms
- **Version Control**: Use Git-like branching for experimental analysis
- **Performance Tuning**: Optimize for your specific graph characteristics
- **Distributed Processing**: Scale to very large graphs (future)

### Community & Support
- **Documentation**: Full API reference and tutorials
- **Examples**: Real-world use cases and jupyter notebooks
- **Performance**: Benchmarks vs other graph libraries
- **Contributing**: How to extend and improve Groggy

---

## Key Takeaways

- **Python API = Productivity**: Focus on getting results quickly with readable code
- **Rust Core = Performance**: Heavy computation optimized in systems language  
- **Ecosystem Native**: Works seamlessly with pandas, numpy, sklearn, networkx
- **Flexible by Design**: Accept any data format, provide multiple interfaces
- **Scale When Needed**: Start simple, optimize for performance when required
- **Graph + ML**: Unique position bridging network analysis and machine learning

**The Groggy Python API makes graph analysis as easy as pandas DataFrame manipulation, while delivering the performance of compiled systems languages.** 🚀

---

*Ready to build something amazing with graphs? Check out our tutorials and examples to get started!*
