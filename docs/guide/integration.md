# Integration with Other Libraries

Groggy integrates seamlessly with the Python data science ecosystem. Export to pandas, numpy, NetworkX, and other popular libraries.

---

## pandas Integration

### Graph to DataFrame

Convert tables to pandas DataFrames:

```python
import groggy as gr
import pandas as pd

g = gr.generators.karate_club()

# Nodes to DataFrame
nodes_df = g.nodes.table().to_pandas()
print(type(nodes_df))  # pandas.DataFrame
print(nodes_df.head())

# Edges to DataFrame
edges_df = g.edges.table().to_pandas()
print(edges_df.head())
```

### DataFrame to Graph

Build graphs from DataFrames:

```python
import pandas as pd
import groggy as gr

# Create DataFrames
nodes_df = pd.DataFrame({
    'node_id': [0, 1, 2],
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [29, 55, 31]
})

edges_df = pd.DataFrame({
    'source': [0, 0, 1],
    'target': [1, 2, 2],
    'weight': [5.0, 2.0, 1.0]
})

# Build graph
g = gr.Graph()

# Add nodes from DataFrame
for _, row in nodes_df.iterrows():
    g.add_node(**row.to_dict())

# Add edges from DataFrame
for _, row in edges_df.iterrows():
    src = int(row['source'])
    tgt = int(row['target'])
    attrs = {k: v for k, v in row.items() if k not in ['source', 'target']}
    g.add_edge(src, tgt, **attrs)
```

### Bulk Import

For better performance, use bulk operations:

```python
# Prepare data
nodes_data = nodes_df.to_dict('records')
edges_data = edges_df.to_dict('records')

# Add in bulk
g = gr.Graph()
g.add_nodes(nodes_data)

# For edges, may need to extract src/dst separately
for edge in edges_data:
    src = int(edge.pop('source'))
    tgt = int(edge.pop('target'))
    g.add_edge(src, tgt, **edge)
```

---

## NumPy Integration

### Arrays to NumPy

Convert Groggy arrays to numpy:

```python
import numpy as np
import groggy as gr

g = gr.generators.karate_club()

# Get attribute as numpy array
ages_groggy = g.nodes["age"]
ages_numpy = np.array(ages_groggy.to_list())

print(type(ages_numpy))  # numpy.ndarray
```

### Matrices to NumPy

Convert matrices:

```python
# Adjacency matrix
A_groggy = g.adjacency_matrix()

# Convert to numpy
A_data = A_groggy.data()  # List of lists
A_numpy = np.array(A_data)

print(A_numpy.shape)

# Or via flatten and reshape
flat = A_groggy.flatten()
shape = A_groggy.shape()
A_numpy = np.array(flat.to_list()).reshape(shape)
```

### NumPy to Groggy

Create Groggy objects from numpy:

```python
# NumPy array to Groggy array
np_array = np.array([1, 2, 3, 4, 5])
groggy_array = gr.num_array(np_array.tolist())

# NumPy matrix to GraphMatrix
np_matrix = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 0, 1]])
groggy_matrix = gr.matrix(np_matrix.tolist())
```

---

## NetworkX Integration

### Groggy to NetworkX

Convert to NetworkX for their algorithms:

```python
import networkx as nx
import groggy as gr

g = gr.generators.karate_club()

# Method 1: Via edge list
edges = list(zip(
    g.edges.sources().to_list(),
    g.edges.targets().to_list()
))
G_nx = nx.Graph(edges)

# Add node attributes
for nid in g.nodes.ids().to_list():
    # Get attributes for this node
    # (Would need accessor for individual node)
    pass

# Method 2: Manual construction
G_nx = nx.Graph()

# Add nodes with attributes
nodes_df = g.nodes.table().to_pandas()
for _, row in nodes_df.iterrows():
    node_id = row['id'] if 'id' in row else row.name
    attrs = row.to_dict()
    G_nx.add_node(node_id, **attrs)

# Add edges with attributes
edges_df = g.edges.table().to_pandas()
for _, row in edges_df.iterrows():
    src = int(row['source'])
    tgt = int(row['target'])
    attrs = {k: v for k, v in row.items() if k not in ['source', 'target']}
    G_nx.add_edge(src, tgt, **attrs)
```

### NetworkX to Groggy

Import NetworkX graphs:

```python
import networkx as nx
import groggy as gr

# Create NetworkX graph
G_nx = nx.karate_club_graph()

# Convert to Groggy
g = gr.Graph()

# Add nodes with attributes
for node, attrs in G_nx.nodes(data=True):
    g.add_node(**attrs)

# Add edges with attributes
for src, tgt, attrs in G_nx.edges(data=True):
    g.add_edge(src, tgt, **attrs)
```

---

## SciPy Integration

### Sparse Matrices

Use scipy for sparse operations:

```python
from scipy.sparse import csr_matrix
import groggy as gr

g = gr.generators.karate_club()

# Get adjacency
A = g.adjacency_matrix()

# Convert to scipy sparse
# Method: via data
data = A.data()
scipy_matrix = csr_matrix(data)

print(type(scipy_matrix))  # scipy.sparse.csr_matrix

# Use scipy algorithms
from scipy.sparse import linalg
eigenvalues, eigenvectors = linalg.eigsh(scipy_matrix, k=5)
```

### Spectral Methods

```python
# Laplacian for spectral clustering
L = g.laplacian_matrix()
L_scipy = csr_matrix(L.data())

# Compute eigenvectors
k = 10
eigenvalues, eigenvectors = linalg.eigsh(L_scipy, k=k, which='SM')

# Use for clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(eigenvectors)

# Store back in graph
g.nodes.set_attrs({
    int(nid): {"cluster": int(label)}
    for nid, label in zip(g.nodes.ids(), labels)
})
```

---

## Scikit-learn Integration

### Feature Matrices

Use graph attributes for ML:

```python
from sklearn.ensemble import RandomForestClassifier
import groggy as gr
import numpy as np

g = gr.generators.karate_club()

# Extract features
ages = g.nodes["age"].to_list()
# Add more features as needed
X = np.array(ages).reshape(-1, 1)

# Labels (example)
labels = np.random.randint(0, 2, size=len(ages))

# Train model
clf = RandomForestClassifier()
clf.fit(X, labels)

# Predict
predictions = clf.predict(X)

# Store predictions
g.nodes.set_attrs({
    int(nid): {"prediction": int(pred)}
    for nid, pred in zip(g.nodes.ids(), predictions)
})
```

### Graph Features

Use graph-derived features:

```python
# Extract graph features
degrees = g.degree().to_list()
# clustering = g.clustering_coefficient()  # If available

# Combine features
X = np.column_stack([
    degrees,
    # clustering,
    ages
])

# Use in ML models
from sklearn.svm import SVC
model = SVC()
model.fit(X, labels)
```

---

## PyTorch Integration

See [Neural Networks Guide](neural.md) for detailed PyTorch integration.

Quick example:

```python
import torch
import groggy as gr

g = gr.generators.karate_club()

# Convert to PyTorch tensors
A_data = g.adjacency_matrix().data()
A_torch = torch.tensor(A_data, dtype=torch.float32)

ages = g.nodes["age"].to_list()
X_torch = torch.tensor(ages, dtype=torch.float32).reshape(-1, 1)

# Use with PyTorch models
# model = MyGNN()
# out = model(A_torch, X_torch)
```

---

## File Format Integration

### CSV

Import/export CSV files:

```python
import groggy as gr

g = gr.generators.karate_club()

# Export to CSV
g.nodes.table().to_pandas().to_csv("nodes.csv", index=False)
g.edges.table().to_pandas().to_csv("edges.csv", index=False)

# Import from CSV
import pandas as pd
nodes_df = pd.read_csv("nodes.csv")
edges_df = pd.read_csv("edges.csv")

# Build graph (see pandas section)
```

### Parquet

Efficient columnar format:

```python
# Export to Parquet
g.nodes.table().to_pandas().to_parquet("nodes.parquet")
g.edges.table().to_pandas().to_parquet("edges.parquet")

# Import from Parquet
nodes_df = pd.read_parquet("nodes.parquet")
edges_df = pd.read_parquet("edges.parquet")
```

### JSON

```python
# Export to JSON
g.nodes.table().to_pandas().to_json("nodes.json", orient="records")
g.edges.table().to_pandas().to_json("edges.json", orient="records")

# Import from JSON
nodes_df = pd.read_json("nodes.json")
edges_df = pd.read_json("edges.json")
```

### Graph Bundles

Groggy's native format:

```python
# Save complete graph
g.save_bundle("graph.bundle")

# Load complete graph
table = gr.GraphTable.load_bundle("graph.bundle")
g_restored = table.to_graph()
```

---

## Common Patterns

### Pattern 1: pandas → Groggy → Analysis → pandas

```python
# Start with DataFrames
nodes_df = pd.read_csv("nodes.csv")
edges_df = pd.read_csv("edges.csv")

# Build graph
g = build_graph_from_dfs(nodes_df, edges_df)

# Run graph algorithms
g.connected_components(inplace=True, label="component")
degrees = g.degree()

# Back to pandas for analysis
result_df = g.nodes.table().to_pandas()
print(result_df.groupby('component')['age'].mean())
```

### Pattern 2: NetworkX Algorithm → Groggy Storage

```python
import networkx as nx
import groggy as gr

# Use NetworkX algorithm
G_nx = nx.karate_club_graph()
pagerank = nx.pagerank(G_nx)

# Store in Groggy
g = convert_nx_to_groggy(G_nx)
g.nodes.set_attrs({
    int(nid): {"pagerank": float(pr)}
    for nid, pr in pagerank.items()
})

# Export
g.save_bundle("graph_with_pagerank.bundle")
```

### Pattern 3: Groggy Features → sklearn Model

```python
# Extract features from graph
degrees = g.degree().to_list()
# Add graph metrics
features = np.column_stack([
    degrees,
    # other features...
])

# Train model
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(features, labels)

# Predict and store
predictions = clf.predict(features)
g.nodes.set_attrs({
    int(nid): {"ml_prediction": int(pred)}
    for nid, pred in zip(g.nodes.ids(), predictions)
})
```

### Pattern 4: Multi-Library Pipeline

```python
# 1. Load with pandas
df = pd.read_csv("network.csv")

# 2. Build graph with Groggy
g = build_from_dataframe(df)

# 3. Compute features with NetworkX
G_nx = convert_to_networkx(g)
centrality = nx.betweenness_centrality(G_nx)

# 4. Back to Groggy
for node, cent in centrality.items():
    g.nodes.set_attrs({int(node): {"centrality": float(cent)}})

# 5. ML with sklearn
X = extract_features(g)
model = train_model(X, labels)

# 6. Export with pandas
result = g.nodes.table().to_pandas()
result.to_csv("results.csv")
```

---

## Helper Functions

### pandas → Groggy

```python
def graph_from_dataframes(nodes_df, edges_df,
                         node_id_col='id',
                         edge_src_col='source',
                         edge_tgt_col='target'):
    """Build Groggy graph from pandas DataFrames."""
    g = gr.Graph()

    # Add nodes
    for _, row in nodes_df.iterrows():
        attrs = row.to_dict()
        if node_id_col in attrs:
            attrs.pop(node_id_col)
        g.add_node(**attrs)

    # Add edges
    for _, row in edges_df.iterrows():
        src = int(row[edge_src_col])
        tgt = int(row[edge_tgt_col])
        attrs = {k: v for k, v in row.items()
                if k not in [edge_src_col, edge_tgt_col]}
        g.add_edge(src, tgt, **attrs)

    return g
```

### NetworkX → Groggy

```python
def networkx_to_groggy(G_nx):
    """Convert NetworkX graph to Groggy."""
    g = gr.Graph()

    # Add nodes
    for node, attrs in G_nx.nodes(data=True):
        g.add_node(**attrs)

    # Add edges
    for src, tgt, attrs in G_nx.edges(data=True):
        g.add_edge(src, tgt, **attrs)

    return g
```

### Groggy → NetworkX

```python
def groggy_to_networkx(g, directed=False):
    """Convert Groggy graph to NetworkX."""
    G_nx = nx.DiGraph() if directed else nx.Graph()

    # Add nodes with attributes
    nodes_df = g.nodes.table().to_pandas()
    for _, row in nodes_df.iterrows():
        node_id = int(row.get('id', row.name))
        attrs = row.to_dict()
        G_nx.add_node(node_id, **attrs)

    # Add edges with attributes
    edges_df = g.edges.table().to_pandas()
    for _, row in edges_df.iterrows():
        src = int(row['source'])
        tgt = int(row['target'])
        attrs = {k: v for k, v in row.items()
                if k not in ['source', 'target']}
        G_nx.add_edge(src, tgt, **attrs)

    return G_nx
```

---

## Quick Reference

### pandas

```python
# To pandas
df = g.nodes.table().to_pandas()

# From pandas
g = graph_from_dataframes(nodes_df, edges_df)
```

### NumPy

```python
# To numpy
arr = np.array(g.nodes["age"].to_list())
mat = np.array(g.adjacency_matrix().data())

# From numpy
groggy_arr = gr.num_array(np_array.tolist())
```

### NetworkX

```python
# To NetworkX
G_nx = groggy_to_networkx(g)

# From NetworkX
g = networkx_to_groggy(G_nx)
```

### Files

```python
# CSV
df.to_csv("file.csv")
df = pd.read_csv("file.csv")

# Parquet
df.to_parquet("file.parquet")
df = pd.read_parquet("file.parquet")

# Bundle (native)
g.save_bundle("graph.bundle")
g = gr.GraphTable.load_bundle("graph.bundle").to_graph()
```

---

## See Also

- **[Tables Guide](tables.md)**: Working with tabular data
- **[Arrays Guide](arrays.md)**: Array conversions
- **[Matrices Guide](matrices.md)**: Matrix operations
- **[Neural Guide](neural.md)**: PyTorch integration
- **[Graph Core Guide](graph-core.md)**: Import/export operations
