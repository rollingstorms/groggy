# Neural Networks and Graph Learning

Groggy provides support for graph neural networks (GNNs) and deep learning on graphs. The neural module integrates with the graph structure to enable modern machine learning workflows.

---

## Overview

Graph Neural Networks extend deep learning to graph-structured data:

```python
import groggy as gr

g = gr.generators.karate_club()

# Neural operations on graphs
# (Check availability in your version)
```

**Key concepts:**
- **Node features**: Attributes as input vectors
- **Message passing**: Aggregating neighbor information
- **Graph convolutions**: Learning on graph structure
- **Automatic differentiation**: Backpropagation through graph ops

---

## Neural Module Architecture

Groggy's neural capabilities are built on:

1. **Rust core**: Fast tensor operations
2. **Automatic differentiation**: Built-in gradient computation
3. **Graph-aware ops**: Message passing primitives
4. **PyTorch compatibility**: Integration with existing frameworks

---

## Node Features

### Feature Matrices

Convert node attributes to feature matrices:

```python
g = gr.Graph()
g.add_node(name="Alice", age=29, score=0.8)
g.add_node(name="Bob", age=55, score=0.6)
g.add_node(name="Carol", age=31, score=0.9)

# Extract features as matrix
import numpy as np

ages = g.nodes["age"].to_list()
scores = g.nodes["score"].to_list()

# Stack features
X = np.column_stack([ages, scores])
print(X.shape)  # (3, 2)

# Or use GraphMatrix if available
# X = gr.matrix_from_attributes(g, ["age", "score"])
```

### Feature Normalization

```python
# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Or use matrix ops
from groggy import matrix
X_mat = matrix(X.tolist())
# normalized = X_mat.standardize()  # If available
```

---

## Message Passing

### Concept

Message passing aggregates information from neighbors:

```
For each node:
  1. Get messages from neighbors
  2. Aggregate messages (sum, mean, max)
  3. Update node representation
```

### Manual Implementation

```python
# Get adjacency matrix
A = g.adjacency_matrix()

# Node features
X = get_features(g)  # (num_nodes, feature_dim)

# Message passing: H = A @ X
# Each node gets sum of neighbor features
H = A.matmul(X)  # If matmul available

# Or normalize by degree
D = g.degree()
# H_normalized = D_inv @ A @ X
```

### Built-in Message Passing

```python
# If neural module provides message passing
# H = g.neural.message_pass(
#     features=X,
#     aggregation="mean"  # or "sum", "max"
# )
```

---

## Graph Convolutions

### Graph Convolutional Layer (GCN)

```python
# Conceptual GCN layer
# H_out = σ(D^(-1/2) A D^(-1/2) H_in W)

# Where:
# - A: adjacency matrix
# - D: degree matrix
# - H_in: input features
# - W: learnable weights
# - σ: activation (ReLU, etc.)
```

### Implementation

```python
# If GCN layers available in neural module
# from groggy.neural import GCNLayer

# layer = GCNLayer(
#     in_features=16,
#     out_features=32,
#     activation="relu"
# )

# H_out = layer(g, H_in)
```

---

## Graph Attention

### Attention Mechanism

Graph Attention Networks (GAT) learn attention weights:

```python
# Conceptual attention
# α_ij = softmax(attention_score(h_i, h_j))
# h_i' = Σ_j α_ij W h_j

# If GAT available
# from groggy.neural import GATLayer

# layer = GATLayer(
#     in_features=16,
#     out_features=32,
#     num_heads=8
# )
```

---

## Neural Network Example

### Node Classification

```python
# Example GNN for node classification

import groggy as gr
import numpy as np

# Load graph
g = gr.generators.karate_club()

# Prepare features
ages = g.nodes["age"].to_list()
X = np.array(ages).reshape(-1, 1)  # (num_nodes, 1)

# Prepare labels (example)
labels = np.random.randint(0, 2, size=g.node_count())

# Build GNN (conceptual)
# model = gr.neural.GNN(
#     layers=[
#         GCNLayer(1, 16),
#         GCNLayer(16, 32),
#         GCNLayer(32, 2)  # 2 classes
#     ]
# )

# Training loop (conceptual)
# for epoch in range(100):
#     # Forward pass
#     logits = model(g, X)
#
#     # Compute loss
#     loss = cross_entropy(logits, labels)
#
#     # Backward pass
#     loss.backward()
#
#     # Update weights
#     optimizer.step()
```

---

## Automatic Differentiation

### Gradient Computation

If the neural module supports autograd:

```python
# Operations track gradients
# from groggy.neural import Tensor

# x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
# y = x ** 2
# loss = y.sum()

# Backward pass
# loss.backward()

# Get gradients
# print(x.grad)  # dL/dx
```

### Matrix Operations with Gradients

```python
# If GraphMatrix supports gradients
A = g.adjacency_matrix()
# A.requires_grad = True

# Forward pass
# H = A.matmul(X)
# loss = compute_loss(H)

# Backward
# loss.backward()

# Get gradient
# dL_dA = A.grad()
```

---

## Integration with PyTorch

### Converting to PyTorch

```python
import torch
import groggy as gr

g = gr.generators.karate_club()

# Get adjacency as PyTorch tensor
A_data = g.adjacency_matrix().data()
A_torch = torch.tensor(A_data, dtype=torch.float32)

# Get features
X_list = g.nodes["age"].to_list()
X_torch = torch.tensor(X_list, dtype=torch.float32).reshape(-1, 1)

# Now use PyTorch GNN libraries
# from torch_geometric.nn import GCNConv
# model = GCNConv(in_channels=1, out_channels=16)
# out = model(X_torch, edge_index)
```

### Edge Index Format

PyTorch Geometric uses edge index format:

```python
# Convert Groggy edges to PyTorch Geometric format
sources = g.edges.sources().to_list()
targets = g.edges.targets().to_list()

edge_index = torch.tensor([sources, targets], dtype=torch.long)
print(edge_index.shape)  # (2, num_edges)

# Use in PyG
# model = GCNConv(in_features, out_features)
# out = model(x, edge_index)
```

---

## Common Patterns

### Pattern 1: Feature Engineering

```python
# Extract multiple features
features = {}
for attr in ["age", "score", "activity"]:
    if attr in g.nodes.attribute_names():
        features[attr] = g.nodes[attr].to_list()

# Combine into matrix
import pandas as pd
df = pd.DataFrame(features)
X = df.values  # (num_nodes, num_features)
```

### Pattern 2: Train/Val/Test Split

```python
# Random split
num_nodes = g.node_count()
indices = np.random.permutation(num_nodes)

train_size = int(0.6 * num_nodes)
val_size = int(0.2 * num_nodes)

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size+val_size]
test_idx = indices[train_size+val_size:]

# Create masks
train_mask = np.zeros(num_nodes, dtype=bool)
train_mask[train_idx] = True

# Use in training
# loss = compute_loss(predictions[train_mask], labels[train_mask])
```

### Pattern 3: Subgraph Sampling

```python
# Sample subgraph for mini-batch training
sample_size = 100
sample_nodes = g.nodes.sample(sample_size)
sample_graph = sample_nodes.to_graph()

# Train on sample
# predictions = model(sample_graph, X_sample)
```

### Pattern 4: Embedding Visualization

```python
# After training, extract embeddings
# embeddings = model.get_embeddings(g, X)  # (num_nodes, embedding_dim)

# Visualize with dimensionality reduction
from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2)
# coords_2d = tsne.fit_transform(embeddings)

# Plot
# import matplotlib.pyplot as plt
# plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
```

---

## Performance Considerations

### GPU Acceleration

For large graphs and deep networks:

```python
# Move to GPU (if supported)
# g_gpu = g.to("cuda")
# model_gpu = model.to("cuda")

# Training on GPU
# for epoch in range(100):
#     logits = model_gpu(g_gpu, X_gpu)
#     loss = compute_loss(logits, labels_gpu)
#     loss.backward()
```

### Memory Management

```python
# For very large graphs
# - Use subgraph sampling
# - Batch processing
# - GraphSAINT sampling
# - Neighbor sampling

# Example: K-hop neighborhood sampling
k_hop = 2
sample = g.nodes[:1000].neighborhood(depth=k_hop)
```

---

## Graph Learning Tasks

### Node Classification

Predict labels for nodes:

```python
# Use cases:
# - User categorization
# - Molecule properties
# - Document topics
```

### Link Prediction

Predict missing or future edges:

```python
# Approach:
# 1. Encode nodes with GNN
# 2. Score node pairs: score(u, v) = f(h_u, h_v)
# 3. Predict edges with high scores
```

### Graph Classification

Classify entire graphs:

```python
# Use cases:
# - Molecule classification
# - Social network analysis
# - Program classification

# Approach:
# 1. Node embeddings via GNN
# 2. Graph pooling (mean, max, attention)
# 3. Classification head
```

---

## Future Capabilities

Potential neural features in Groggy:

- **Built-in GNN layers**: GCN, GAT, GraphSAGE
- **Automatic differentiation**: Native autograd
- **Optimizers**: Adam, SGD, etc.
- **Loss functions**: Cross-entropy, MSE, etc.
- **Sampling strategies**: Neighbor sampling, GraphSAINT
- **Pooling operations**: DiffPool, SAGPool
- **Pre-trained models**: Transfer learning

Check the API reference and release notes for availability.

---

## Quick Reference

### Feature Extraction

```python
# Single feature
ages = g.nodes["age"].to_list()

# Multiple features
X = np.column_stack([
    g.nodes["age"].to_list(),
    g.nodes["score"].to_list()
])
```

### Matrix Operations

```python
# Adjacency
A = g.adjacency_matrix()

# Laplacian
L = g.laplacian_matrix()

# Activations
relu_A = A.relu()
sigmoid_A = A.sigmoid()
```

### PyTorch Integration

```python
# Edge index
edge_index = torch.tensor([
    g.edges.sources().to_list(),
    g.edges.targets().to_list()
], dtype=torch.long)

# Features
X = torch.tensor(features, dtype=torch.float32)
```

---

## See Also

- **[Matrices Guide](matrices.md)**: Matrix operations for GNNs
- **[Graph Core Guide](graph-core.md)**: Graph structure
- **[Algorithms Guide](algorithms.md)**: Traditional graph algorithms
- **[Performance Guide](performance.md)**: Optimizing neural network training
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **DGL**: https://www.dgl.ai/
