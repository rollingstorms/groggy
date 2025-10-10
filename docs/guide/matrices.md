# Working with Matrices

Matrices in Groggy provide **matrix representations** of graph structure and enable matrix-based graph algorithms. Think linear algebra operations on graph topology and attributes.

---

## What are GraphMatrices?

GraphMatrix is Groggy's matrix type for graph-related linear algebra:

```python
import groggy as gr

g = gr.generators.karate_club()

# Adjacency matrix
A = g.adjacency_matrix()  # GraphMatrix
print(type(A))

# Laplacian matrix
L = g.laplacian_matrix()  # GraphMatrix

# Shape
print(A.shape())  # (num_nodes, num_nodes)
```

**Common matrix types:**
- **Adjacency matrix**: Who connects to whom
- **Laplacian matrix**: For spectral analysis
- **Weight matrix**: Weighted connections
- **Feature matrices**: Node/edge attributes as matrices

---

## Creating Matrices

### From Graph Structure

Get structural matrices:

```python
g = gr.Graph()
n0, n1, n2 = g.add_node(), g.add_node(), g.add_node()
g.add_edge(n0, n1)
g.add_edge(n0, n2)
g.add_edge(n1, n2)

# Adjacency matrix (0/1 for connections)
A = g.adjacency_matrix()
print(A.shape())  # (3, 3)

# Alternative access
A = g.adj()  # Same as adjacency_matrix()

# Laplacian matrix
L = g.laplacian_matrix()
```

### From Subgraphs

Subgraphs also have matrix methods:

```python
# Filter to subgraph
sub = g.nodes[:10]

# Get matrices for subgraph
A_sub = sub.adjacency_matrix()
L_sub = sub.to_matrix()  # Generic matrix conversion
```

### From Attributes

Edge weights can create weighted matrices:

```python
g = gr.Graph()
n0, n1, n2 = g.add_node(), g.add_node(), g.add_node()
g.add_edge(n0, n1, weight=5.0)
g.add_edge(n0, n2, weight=2.0)
g.add_edge(n1, n2, weight=1.0)

# Weight matrix
W = g.edges.weight_matrix()  # GraphMatrix with weights
```

### Manual Creation

Create matrices directly:

```python
# From data
data = [[1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]]

M = gr.matrix(data)  # GraphMatrix

# Identity matrix
I = gr.GraphMatrix.identity(3)  # 3x3 identity

# From array
arr = gr.num_array([1, 2, 3, 4, 5, 6])
M = arr.reshape((2, 3))  # 2x3 matrix (if supported)
```

---

## Matrix Properties

### Shape and Type

```python
A = g.adjacency_matrix()

# Shape
rows, cols = A.shape()
print(f"Shape: {rows}x{cols}")

# Data type
dtype = A.dtype()
print(f"Type: {dtype}")

# Sparsity
if A.is_sparse():
    print("Sparse matrix")
else:
    print("Dense matrix")
```

### Checking Properties

```python
# Square matrix?
if A.is_square():
    print("Square matrix")

# Symmetric?
if A.is_symmetric():
    print("Symmetric")

# Empty?
if A.is_empty():
    print("No elements")

# Numeric?
if A.is_numeric():
    print("Contains numbers")
```

---

## Matrix Operations

### Basic Arithmetic

```python
A = g.adjacency_matrix()

# Element-wise operations
abs_A = A.abs()      # Absolute values
exp_A = A.exp()      # Exponential
log_A = A.log()      # Logarithm (careful with zeros!)

# Matrix norms
frobenius = A.norm()      # Frobenius norm
l1_norm = A.norm_l1()     # L1 norm
inf_norm = A.norm_inf()   # Infinity norm
```

### Activation Functions

Neural network-style activations:

```python
# Activation functions (useful for GNNs)
relu_A = A.relu()
leaky_A = A.leaky_relu()
elu_A = A.elu()
gelu_A = A.gelu()
sigmoid_A = A.sigmoid()
softmax_A = A.softmax()
tanh_A = A.tanh()
```

### Statistical Operations

```python
# Global statistics
max_val = A.max()
min_val = A.min()
mean_val = A.mean()
sum_val = A.sum()

print(f"Matrix stats: min={min_val}, max={max_val}, mean={mean_val:.2f}")

# Axis-wise (if supported)
# row_max = A.max_axis(0)  # Max per column
# col_max = A.max_axis(1)  # Max per row
```

### Matrix Multiplication

```python
# Matrix-matrix multiply
C = A.matmul(B)  # A @ B

# Matrix-vector multiply
v = gr.num_array([1, 2, 3])
result = A.matmul_vec(v)  # A @ v
```

---

## Accessing Matrix Data

### Row and Column Access

```python
A = g.adjacency_matrix()

# Get row (if supported)
# row_0 = A.get_row(0)

# Get column (if supported)
# col_0 = A.get_column(0)

# Iterate rows
for row in A.iter_rows():
    print(row)

# Iterate columns
for col in A.iter_columns():
    print(col)
```

### Conversion

```python
# To dense (if sparse)
dense_A = A.dense()

# Flatten to array
flat = A.flatten()  # NumArray
print(flat.head())

# Get raw data
data = A.data()  # List of lists
print(data)

# Column names (if applicable)
cols = A.columns()
```

---

## Sparse vs Dense

### Checking Sparsity

```python
A = g.adjacency_matrix()

if A.is_sparse():
    print("Stored as sparse matrix")
    # Efficient for large graphs with few edges
else:
    print("Stored as dense matrix")
    # Better for small or dense graphs
```

### Converting

```python
# Force to dense
dense_A = A.dense()

# Sparse is automatic based on sparsity
# Groggy chooses representation internally
```

---

## Common Patterns

### Pattern 1: Spectral Analysis

```python
# Get Laplacian
L = g.laplacian_matrix()

# Eigenvalue decomposition (if supported)
# eigenvalues, eigenvectors = L.eigenvalue_decomposition()

# For spectral clustering, embeddings, etc.
```

### Pattern 2: Power Iteration

```python
# Matrix power (if supported)
# A2 = A.power(2)  # A^2
# A3 = A.power(3)  # A^3

# For path counting, centrality, etc.
```

### Pattern 3: Adjacency Properties

```python
A = g.adjacency_matrix()

# Density from matrix
edges = A.sum() / 2  # Undirected graph
n = A.shape()[0]
max_edges = n * (n - 1) / 2
density = edges / max_edges

print(f"Graph density: {density:.3f}")
```

### Pattern 4: Degree Calculation

```python
A = g.adjacency_matrix()

# Row sums = out-degree (if supported)
# out_degrees = A.sum_axis(1)

# Column sums = in-degree
# in_degrees = A.sum_axis(0)

# Or use graph methods
degrees = g.degree()  # NumArray
```

### Pattern 5: Matrix to NumPy

```python
A = g.adjacency_matrix()

# Convert to numpy
import numpy as np

# Via data
data = A.data()
np_matrix = np.array(data)

# Or via flatten and reshape
flat = A.flatten()
shape = A.shape()
np_matrix = np.array(flat.to_list()).reshape(shape)
```

### Pattern 6: Feature Matrix

```python
# Collect node features as matrix
ages = g.nodes["age"]
scores = g.nodes["score"]

# Stack into matrix (manually)
import numpy as np
X = np.column_stack([
    ages.to_list(),
    scores.to_list()
])

print(X.shape)  # (num_nodes, 2)
```

### Pattern 7: Weighted Adjacency

```python
# Get weighted adjacency
W = g.edges.weight_matrix()

# Use in algorithms
# pagerank_scores = compute_pagerank(W)

# Or normalize
max_weight = W.max()
W_normalized = W.map(lambda x: x / max_weight) if max_weight > 0 else W
```

---

## Matrix Transformations

### Apply Functions

```python
A = g.adjacency_matrix()

# Apply function to each element
squared = A.map(lambda x: x ** 2)
doubled = A.map(lambda x: x * 2)

# Or use apply
transformed = A.apply(lambda x: max(0, x - 1))
```

### Neural Network Ops

```python
# For GNN implementations
hidden = A.relu()
activated = hidden.sigmoid()
normalized = activated.softmax()

# Dropout (if supported)
# dropped = hidden.dropout(0.5)
```

---

## Performance Considerations

### Sparse Matrices

For large graphs, sparse representation is crucial:

```python
# Large graph
g = gr.generators.erdos_renyi(n=10000, p=0.001)

# Adjacency is sparse
A = g.adjacency_matrix()

if A.is_sparse():
    # Efficient storage and operations
    print("Using sparse representation")
```

### Dense Operations

Small or dense graphs use dense matrices:

```python
# Complete graph - all edges exist
g = gr.generators.complete_graph(100)

A = g.adjacency_matrix()
# Likely stored as dense since almost all entries are 1
```

### Memory

```python
# Sparse: ~O(num_edges) memory
# Dense: O(n^2) memory

# For n=10,000:
# - Sparse with 100,000 edges: ~1 MB
# - Dense: ~400 MB (10k x 10k floats)
```

---

## Limitations

### Matrix Size

Large graphs can create huge matrices:

```python
# 100,000 nodes = 10 billion matrix entries
# Even sparse, operations can be expensive

# Consider:
# - Working with subgraphs
# - Using specialized algorithms
# - Approximate methods
```

### Missing Operations

Some operations may not be implemented:

```python
# Check API reference for available methods
# Not all numpy/scipy operations available
```

---

## Quick Reference

### Creating Matrices

```python
A = g.adjacency_matrix()        # 0/1 adjacency
L = g.laplacian_matrix()        # Graph Laplacian
W = g.edges.weight_matrix()     # Weighted edges
M = g.to_matrix()               # Generic conversion
I = gr.GraphMatrix.identity(n)  # Identity matrix
```

### Properties

```python
shape = A.shape()         # (rows, cols)
is_sparse = A.is_sparse() # True/False
is_square = A.is_square() # True/False
is_sym = A.is_symmetric() # True/False
dtype = A.dtype()         # "float", "int", etc.
```

### Operations

```python
# Element-wise
abs_A = A.abs()
exp_A = A.exp()
log_A = A.log()

# Norms
frobenius = A.norm()
l1 = A.norm_l1()
inf = A.norm_inf()

# Statistics
max_val = A.max()
min_val = A.min()
mean_val = A.mean()
sum_val = A.sum()

# Activations
relu_A = A.relu()
sigmoid_A = A.sigmoid()
softmax_A = A.softmax()
```

### Conversion

```python
dense_A = A.dense()       # To dense
flat = A.flatten()        # To NumArray
data = A.data()           # To list of lists
```

---

## See Also

- **[GraphMatrix API Reference](../api/graphmatrix.md)**: Complete method reference
- **[Graph Core Guide](graph-core.md)**: Getting matrices from graphs
- **[Subgraphs Guide](subgraphs.md)**: Subgraph matrices
- **[Arrays Guide](arrays.md)**: Working with array data
- **[Algorithms Guide](algorithms.md)**: Matrix-based algorithms
