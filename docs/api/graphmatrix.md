# GraphMatrix API Reference

**Type**: `groggy.GraphMatrix`

---

## Overview

Matrix representation of graph data (adjacency, Laplacian, embeddings).

**Primary Use Cases:**
- Matrix-based graph algorithms
- Spectral analysis
- Graph embeddings

**Related Objects:**
- `Graph`
- `Subgraph`

---

## Complete Method Reference

The following methods are available on `GraphMatrix` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `abs()` | `GraphMatrix` | ✓ |
| `apply()` | `GraphMatrix` | ✓ |
| `backward()` | `?` | ✗ |
| `cholesky_decomposition()` | `?` | ✗ |
| `columns()` | `list` | ✓ |
| `concatenate()` | `?` | ✗ |
| `data()` | `list` | ✓ |
| `dense()` | `GraphMatrix` | ✓ |
| `dense_html_repr()` | `str` | ✓ |
| `determinant()` | `?` | ✗ |
| `dropout()` | `?` | ✗ |
| `dtype()` | `str` | ✓ |
| `eigenvalue_decomposition()` | `?` | ✗ |
| `elementwise_multiply()` | `?` | ✗ |
| `elu()` | `GraphMatrix` | ✓ |
| `exp()` | `GraphMatrix` | ✓ |
| `filter()` | `?` | ✗ |
| `flatten()` | `NumArray` | ✓ |
| `from_base_array()` | `?` | ✗ |
| `from_data()` | `?` | ✗ |
| `from_flattened()` | `?` | ✗ |
| `from_graph_attributes()` | `?` | ✗ |
| `gelu()` | `GraphMatrix` | ✓ |
| `get()` | `?` | ✗ |
| `get_cell()` | `?` | ✗ |
| `get_column()` | `?` | ✗ |
| `get_column_by_name()` | `?` | ✗ |
| `get_row()` | `?` | ✗ |
| `grad()` | `?` | ✓ |
| `identity()` | `GraphMatrix` | ✓ |
| `inverse()` | `?` | ✗ |
| `is_empty()` | `bool` | ✓ |
| `is_numeric()` | `bool` | ✓ |
| `is_sparse()` | `bool` | ✓ |
| `is_square()` | `bool` | ✓ |
| `is_symmetric()` | `bool` | ✓ |
| `iter_columns()` | `list` | ✓ |
| `iter_rows()` | `list` | ✓ |
| `leaky_relu()` | `GraphMatrix` | ✓ |
| `log()` | `GraphMatrix` | ✓ |
| `lu_decomposition()` | `?` | ✗ |
| `map()` | `GraphMatrix` | ✓ |
| `max()` | `float` | ✓ |
| `max_axis()` | `?` | ✗ |
| `mean()` | `float` | ✓ |
| `mean_axis()` | `?` | ✗ |
| `min()` | `float` | ✓ |
| `min_axis()` | `?` | ✗ |
| `multiply()` | `?` | ✗ |
| `norm()` | `float` | ✓ |
| `norm_inf()` | `float` | ✓ |
| `norm_l1()` | `float` | ✓ |
| `ones()` | `?` | ✗ |
| `power()` | `?` | ✗ |
| `preview()` | `list` | ✓ |
| `qr_decomposition()` | `tuple` | ✓ |
| `rank()` | `int` | ✓ |
| `relu()` | `GraphMatrix` | ✓ |
| `repeat()` | `?` | ✗ |
| `requires_grad()` | `bool` | ✓ |
| `requires_grad_()` | `?` | ✗ |
| `reshape()` | `?` | ✗ |
| `rich_display()` | `str` | ✓ |
| `scalar_multiply()` | `?` | ✗ |
| `set()` | `?` | ✗ |
| `shape()` | `tuple` | ✓ |
| `sigmoid()` | `GraphMatrix` | ✓ |
| `softmax()` | `GraphMatrix` | ✓ |
| `solve()` | `?` | ✗ |
| `split()` | `?` | ✗ |
| `sqrt()` | `GraphMatrix` | ✓ |
| `stack()` | `?` | ✗ |
| `std_axis()` | `?` | ✗ |
| `sum()` | `float` | ✓ |
| `sum_axis()` | `?` | ✗ |
| `summary()` | `str` | ✓ |
| `svd()` | `tuple` | ✓ |
| `tanh()` | `GraphMatrix` | ✓ |
| `tile()` | `?` | ✗ |
| `to_base_array()` | `BaseArray` | ✓ |
| `to_degree_matrix()` | `?` | ✗ |
| `to_dict()` | `dict` | ✓ |
| `to_laplacian()` | `?` | ✗ |
| `to_list()` | `list` | ✓ |
| `to_normalized_laplacian()` | `?` | ✗ |
| `to_numpy()` | `ndarray` | ✓ |
| `to_pandas()` | `DataFrame` | ✓ |
| `to_table_for_streaming()` | `BaseTable` | ✓ |
| `trace()` | `?` | ✗ |
| `transpose()` | `GraphMatrix` | ✓ |
| `var_axis()` | `?` | ✗ |
| `zero_grad()` | `?` | ✗ |
| `zeros()` | `?` | ✗ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating GraphMatrix

GraphMatrices are typically created from graphs:

```python
import groggy as gr

g = gr.generators.karate_club()

# From graph structure
A = g.adjacency_matrix()    # Binary adjacency
L = g.laplacian_matrix()     # Graph Laplacian
W = g.edges.weight_matrix()  # Weighted adjacency

# Manual creation
data = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
M = gr.matrix(data)

# Identity matrix
I = gr.GraphMatrix.identity(5)  # 5x5 identity
```

---

### Properties

#### `shape()`

Get matrix dimensions.

**Returns:**
- `tuple[int, int]`: (rows, columns)

**Example:**
```python
A = g.adjacency_matrix()
rows, cols = A.shape()
print(f"Matrix is {rows}x{cols}")
```

**Performance:** O(1)

---

#### `dtype()`

Get data type.

**Returns:**
- `str`: Type name ("float", "int", etc.)

**Example:**
```python
A = g.adjacency_matrix()
print(A.dtype())  # "float" or "int"
```

---

#### `is_sparse()`

Check if stored as sparse matrix.

**Returns:**
- `bool`: True if sparse

**Example:**
```python
A = g.adjacency_matrix()
if A.is_sparse():
    print("Sparse representation (efficient for large graphs)")
else:
    print("Dense representation")
```

**Notes:**
- Groggy automatically chooses representation based on sparsity
- Sparse storage saves memory for graphs with few edges

---

#### `is_square()`

Check if matrix is square.

**Returns:**
- `bool`: True if n×n

**Example:**
```python
A = g.adjacency_matrix()
print(A.is_square())  # True (adjacency always square)
```

---

#### `is_symmetric()`

Check if matrix is symmetric.

**Returns:**
- `bool`: True if A = A^T

**Example:**
```python
A = g.adjacency_matrix()
if A.is_symmetric():
    print("Undirected graph")
else:
    print("Directed graph")
```

---

#### `is_numeric()`

Check if contains numeric values.

**Returns:**
- `bool`: True if numeric

**Example:**
```python
A = g.adjacency_matrix()
print(A.is_numeric())  # True
```

---

#### `is_empty()`

Check if matrix has no elements.

**Returns:**
- `bool`: True if empty

**Example:**
```python
if A.is_empty():
    print("Empty matrix")
```

---

### Statistical Methods

#### `sum()`

Sum of all elements.

**Returns:**
- `float`: Sum

**Example:**
```python
A = g.adjacency_matrix()
total_edges = A.sum() / 2  # Divide by 2 for undirected
print(f"{total_edges} edges")
```

---

#### `mean()`

Mean of all elements.

**Returns:**
- `float`: Mean value

**Example:**
```python
A = g.adjacency_matrix()
density = A.mean()
print(f"Matrix density: {density:.4f}")
```

---

#### `min()` / `max()`

Minimum/maximum values.

**Returns:**
- `float`: Min or max value

**Example:**
```python
W = g.edges.weight_matrix()
min_weight = W.min()
max_weight = W.max()
print(f"Weight range: [{min_weight}, {max_weight}]")
```

---

### Norms

#### `norm()`

Frobenius norm.

**Returns:**
- `float`: ||A||_F = sqrt(sum(A_ij^2))

**Example:**
```python
A = g.adjacency_matrix()
frob_norm = A.norm()
print(f"Frobenius norm: {frob_norm:.2f}")
```

---

#### `norm_l1()`

L1 norm (max column sum).

**Returns:**
- `float`: L1 norm

**Example:**
```python
A = g.adjacency_matrix()
l1 = A.norm_l1()
```

---

#### `norm_inf()`

Infinity norm (max row sum).

**Returns:**
- `float`: L∞ norm

**Example:**
```python
A = g.adjacency_matrix()
linf = A.norm_inf()
```

---

### Activation Functions (for GNNs)

#### `relu()`

ReLU activation: max(0, x).

**Returns:**
- `GraphMatrix`: Activated matrix

**Example:**
```python
H = A.relu()  # All negative values → 0
```

---

#### `leaky_relu()`

Leaky ReLU: max(αx, x) where α is small.

**Returns:**
- `GraphMatrix`: Activated matrix

**Example:**
```python
H = A.leaky_relu()
```

---

#### `elu()`

Exponential Linear Unit.

**Returns:**
- `GraphMatrix`: Activated matrix

**Example:**
```python
H = A.elu()
```

---

#### `gelu()`

Gaussian Error Linear Unit.

**Returns:**
- `GraphMatrix`: Activated matrix

**Example:**
```python
H = A.gelu()
```

---

#### `sigmoid()`

Sigmoid: 1/(1 + e^(-x)).

**Returns:**
- `GraphMatrix`: Values in (0, 1)

**Example:**
```python
probs = A.sigmoid()
```

---

#### `tanh()`

Hyperbolic tangent.

**Returns:**
- `GraphMatrix`: Values in (-1, 1)

**Example:**
```python
H = A.tanh()
```

---

#### `softmax()`

Softmax normalization.

**Returns:**
- `GraphMatrix`: Rows sum to 1

**Example:**
```python
attention = A.softmax()
```

---

### Element-wise Operations

#### `abs()`

Absolute values.

**Returns:**
- `GraphMatrix`: |A_ij|

**Example:**
```python
abs_A = A.abs()
```

---

#### `exp()`

Exponential: e^A_ij.

**Returns:**
- `GraphMatrix`: Exponential of each element

**Example:**
```python
exp_A = A.exp()
```

---

#### `log()`

Natural logarithm.

**Returns:**
- `GraphMatrix`: ln(A_ij)

**Example:**
```python
log_A = A.log()  # Be careful with zeros!
```

---

#### `sqrt()`

Square root.

**Returns:**
- `GraphMatrix`: sqrt(A_ij)

**Example:**
```python
sqrt_A = A.sqrt()
```

---

#### `map(func)` / `apply(func)`

Apply function to each element.

**Parameters:**
- `func`: Function to apply

**Returns:**
- `GraphMatrix`: Transformed matrix

**Example:**
```python
# Square each element
squared = A.map(lambda x: x ** 2)

# Threshold
thresholded = A.map(lambda x: x if x > 0.5 else 0)
```

---

### Matrix Decompositions

#### `svd()`

Singular Value Decomposition.

**Returns:**
- `tuple`: (U, S, V) matrices

**Example:**
```python
A = g.adjacency_matrix()
U, S, V = A.svd()

# Use for dimensionality reduction
k = 10
U_k = U[:, :k]  # Conceptual - actual indexing may differ
```

---

#### `qr_decomposition()`

QR decomposition.

**Returns:**
- `tuple[GraphMatrix, GraphMatrix]`: (Q, R) matrices

**Example:**
```python
Q, R = A.qr_decomposition()
# A = Q @ R
```

---

#### `rank()`

Matrix rank.

**Returns:**
- `int`: Rank

**Example:**
```python
A = g.adjacency_matrix()
r = A.rank()
print(f"Matrix rank: {r}")
```

---

### Data Access

#### `data()`

Get raw data as list of lists.

**Returns:**
- `list[list]`: Row-major data

**Example:**
```python
A = g.adjacency_matrix()
data = A.data()
print(data[0])  # First row
```

---

#### `flatten()`

Flatten to 1D array.

**Returns:**
- `NumArray`: Flattened values

**Example:**
```python
A = g.adjacency_matrix()
flat = A.flatten()
print(len(flat))  # rows * cols
```

---

#### `to_list()`

Convert to nested list.

**Returns:**
- `list`: Same as `data()`

**Example:**
```python
data = A.to_list()
```

---

### Iteration

#### `iter_rows()`

Iterate over rows.

**Returns:**
- Iterator over rows

**Example:**
```python
for row in A.iter_rows():
    print(row)
```

---

#### `iter_columns()`

Iterate over columns.

**Returns:**
- Iterator over columns

**Example:**
```python
for col in A.iter_columns():
    print(col)
```

---

#### `columns()`

Get column names (if applicable).

**Returns:**
- `list[str]`: Column names

**Example:**
```python
cols = A.columns()
```

---

### Conversion

#### `dense()`

Convert to dense representation.

**Returns:**
- `GraphMatrix`: Dense matrix

**Example:**
```python
if A.is_sparse():
    A_dense = A.dense()
```

**Notes:** Only needed if you require dense storage

---

#### `to_numpy()`

Convert to NumPy array.

**Returns:**
- `numpy.ndarray`: NumPy array

**Example:**
```python
import numpy as np

A = g.adjacency_matrix()
np_array = A.to_numpy()
print(type(np_array))  # numpy.ndarray
```

---

#### `to_pandas()`

Convert to pandas DataFrame.

**Returns:**
- `pandas.DataFrame`: DataFrame

**Example:**
```python
A = g.adjacency_matrix()
df = A.to_pandas()
print(df.head())
```

---

#### `to_dict()`

Convert to dictionary.

**Returns:**
- `dict`: Matrix as dict

**Example:**
```python
A = g.adjacency_matrix()
d = A.to_dict()
```

---

### Transformations

#### `transpose()`

Matrix transpose.

**Returns:**
- `GraphMatrix`: A^T

**Example:**
```python
A = g.adjacency_matrix()
AT = A.transpose()

# Check symmetry
if (A.data() == AT.data()):
    print("Symmetric (undirected graph)")
```

---

### Display

#### `summary()`

Get text summary.

**Returns:**
- `str`: Summary string

**Example:**
```python
A = g.adjacency_matrix()
print(A.summary())
# "GraphMatrix: 34x34, sparse, 78 nonzeros"
```

---

#### `preview(n=5)`

Preview first n rows/cols.

**Returns:**
- Display output

**Example:**
```python
A = g.adjacency_matrix()
A.preview(10)
```

---

### Automatic Differentiation

#### `requires_grad()`

Check if gradients are tracked.

**Returns:**
- `bool`: True if tracking gradients

**Example:**
```python
if A.requires_grad():
    print("Gradients will be computed")
```

---

#### `grad()`

Get gradient (after backward pass).

**Returns:**
- `GraphMatrix` or `None`: Gradient

**Example:**
```python
# After loss.backward()
dL_dA = A.grad()
```

---

## Usage Patterns

### Pattern 1: Spectral Analysis

```python
# Get Laplacian
L = g.laplacian_matrix()

# Convert to NumPy for eigenvalue decomposition
import numpy as np
L_np = L.to_numpy()

# Compute eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(L_np)

# Use for spectral clustering
```

### Pattern 2: GNN Forward Pass

```python
# Adjacency matrix
A = g.adjacency_matrix()

# Feature matrix (from node attributes)
ages = g.nodes["age"].to_list()
scores = g.nodes["score"].to_list()
X = np.column_stack([ages, scores])

# GNN layer: H = σ(A @ X @ W)
# (Simplified - actual GNN more complex)
H = A.to_numpy() @ X  # Message passing
H_activated = gr.matrix(H.tolist()).relu()  # Activation
```

### Pattern 3: Sparse Matrix Operations

```python
A = g.adjacency_matrix()

if A.is_sparse():
    print(f"Sparse: {A.shape()} with {A.sum()} nonzeros")

    # Efficient operations on sparse
    stats = {
        'mean': A.mean(),
        'max': A.max(),
        'norm': A.norm()
    }
```

### Pattern 4: Matrix Normalization

```python
W = g.edges.weight_matrix()

# Min-max normalization
min_w = W.min()
max_w = W.max()
W_norm = W.map(lambda x: (x - min_w) / (max_w - min_w) if max_w > min_w else x)

# Or standardization
mean_w = W.mean()
# std_w = W.std()  # If available
```

### Pattern 5: Attention Mechanism

```python
# Compute attention scores
A = g.adjacency_matrix()

# Apply softmax for attention weights
attention = A.softmax()

# Weighted aggregation
# features_aggregated = attention @ node_features
```

---

## Quick Reference

### Properties

| Method | Returns | Description |
|--------|---------|-------------|
| `shape()` | `tuple` | (rows, cols) |
| `dtype()` | `str` | Data type |
| `is_sparse()` | `bool` | Sparse storage? |
| `is_square()` | `bool` | n×n matrix? |
| `is_symmetric()` | `bool` | A = A^T? |

### Statistics

| Method | Returns | Description |
|--------|---------|-------------|
| `sum()` | `float` | Sum of elements |
| `mean()` | `float` | Mean value |
| `min()` / `max()` | `float` | Min/max value |
| `norm()` | `float` | Frobenius norm |
| `norm_l1()` | `float` | L1 norm |
| `norm_inf()` | `float` | L∞ norm |

### Activations (GNN)

| Method | Description |
|--------|-------------|
| `relu()` | max(0, x) |
| `leaky_relu()` | Leaky ReLU |
| `elu()` | Exponential Linear Unit |
| `gelu()` | Gaussian Error Linear Unit |
| `sigmoid()` | 1/(1 + e^(-x)) |
| `tanh()` | Hyperbolic tangent |
| `softmax()` | Row normalization |

### Element-wise

| Method | Description |
|--------|-------------|
| `abs()` | Absolute values |
| `exp()` | e^x |
| `log()` | Natural log |
| `sqrt()` | Square root |
| `map(func)` | Apply function |

### Decompositions

| Method | Returns | Description |
|--------|---------|-------------|
| `svd()` | `tuple` | U, S, V matrices |
| `qr_decomposition()` | `tuple` | Q, R matrices |
| `rank()` | `int` | Matrix rank |

### Conversion

| Method | Returns | Description |
|--------|---------|-------------|
| `to_numpy()` | `ndarray` | NumPy array |
| `to_pandas()` | `DataFrame` | Pandas DataFrame |
| `data()` | `list` | List of lists |
| `flatten()` | `NumArray` | 1D array |

---

## Performance Considerations

**Sparse vs Dense:**
- Large graphs (>1000 nodes, sparse): Sparse saves memory
- Small/dense graphs: Dense may be faster
- Groggy chooses automatically based on sparsity

**Efficient Operations:**
- Element-wise: `abs()`, `exp()`, `relu()` - O(nnz) for sparse
- Statistics: `sum()`, `mean()`, `norm()` - Single pass
- Conversion: `to_numpy()` - O(nnz) or O(n²)

**Less Efficient:**
- `dense()` on large sparse - Forces materialization
- Repeated conversions - Cache the NumPy/pandas result
- Element access in loops - Use bulk operations

**Best Practices:**
```python
# ✅ Good: check sparsity first
if A.is_sparse():
    # Work with sparse methods
    nnz = A.sum()
else:
    # Dense operations
    A_np = A.to_numpy()

# ✅ Good: batch operations
activated = A.relu().sigmoid()

# ❌ Avoid: repeated conversions
for i in range(100):
    np_array = A.to_numpy()  # Reconverts each time

# ✅ Good: convert once
A_np = A.to_numpy()
for i in range(100):
    # Use A_np
```

---

## Integration with SciPy

For advanced matrix operations:

```python
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Convert to scipy sparse
A = g.adjacency_matrix()
A_scipy = csr_matrix(A.to_numpy())

# Use scipy algorithms
eigenvalues, eigenvectors = eigsh(A_scipy, k=10)
```

---

## See Also

- **[Matrices Guide](../guide/matrices.md)**: Comprehensive tutorial
- **[Neural Guide](../guide/neural.md)**: GNN usage patterns
- **[Graph API](graph.md)**: Creating matrices from graphs
- **[NumArray API](numarray.md)**: Working with flattened matrices

## Additional Methods

#### `apply()`

Apply.

**Returns:**
- `GraphMatrix`: Return value

**Example:**
```python
obj.apply()
```

---

#### `backward()`

Backward.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.backward()
```

---

#### `cholesky_decomposition()`

Cholesky Decomposition.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.cholesky_decomposition()
```

---

#### `concatenate(other, axis)`

Concatenate.

**Parameters:**
- `other`: other
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.concatenate(other=..., axis=...)
```

---

#### `dense_html_repr()`

Dense Html Repr.

**Returns:**
- `str`: Return value

**Example:**
```python
obj.dense_html_repr()
```

---

#### `determinant()`

Determinant.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.determinant()
```

---

#### `dropout(p)`

Dropout.

**Parameters:**
- `p`: p

**Returns:**
- `None`: Return value

**Example:**
```python
obj.dropout(p=...)
```

---

#### `eigenvalue_decomposition()`

Eigenvalue Decomposition.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.eigenvalue_decomposition()
```

---

#### `elementwise_multiply(other)`

Elementwise Multiply.

**Parameters:**
- `other`: other

**Returns:**
- `None`: Return value

**Example:**
```python
obj.elementwise_multiply(other=...)
```

---

#### `filter(condition)`

Filter.

**Parameters:**
- `condition`: condition

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter(condition=...)
```

---

#### `from_base_array(base_array, rows, cols)`

From Base Array.

**Parameters:**
- `base_array`: base array
- `rows`: rows
- `cols`: cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_base_array(base_array=..., rows=..., cols=...)
```

---

#### `from_data(data)`

From Data.

**Parameters:**
- `data`: data

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_data(data=...)
```

---

#### `from_flattened(num_array, rows, cols)`

From Flattened.

**Parameters:**
- `num_array`: num array
- `rows`: rows
- `cols`: cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_flattened(num_array=..., rows=..., cols=...)
```

---

#### `from_graph_attributes(_graph, _attrs, _entities)`

From Graph Attributes.

**Parameters:**
- `_graph`:  graph
- `_attrs`:  attrs
- `_entities`:  entities

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_graph_attributes(_graph=..., _attrs=..., _entities=...)
```

---

#### `get(col)`

Get.

**Parameters:**
- `col`: col

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get(col=...)
```

---

#### `get_cell(col)`

Get Cell.

**Parameters:**
- `col`: col

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_cell(col=...)
```

---

#### `get_column()`

Get Column.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_column()
```

---

#### `get_column_by_name()`

Get Column By Name.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_column_by_name()
```

---

#### `get_row(row)`

Get Row.

**Parameters:**
- `row`: row

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_row(row=...)
```

---

#### `identity()`

Identity.

**Returns:**
- `GraphMatrix`: Return value

**Example:**
```python
obj.identity()
```

---

#### `inverse()`

Inverse.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.inverse()
```

---

#### `lu_decomposition()`

Lu Decomposition.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.lu_decomposition()
```

---

#### `max()`

Max.

**Returns:**
- `float`: Return value

**Example:**
```python
obj.max()
```

---

#### `max_axis(axis)`

Max Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.max_axis(axis=...)
```

---

#### `mean_axis(axis)`

Mean Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.mean_axis(axis=...)
```

---

#### `min_axis(axis)`

Min Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.min_axis(axis=...)
```

---

#### `multiply(operand)`

Multiply.

**Parameters:**
- `operand`: operand

**Returns:**
- `None`: Return value

**Example:**
```python
obj.multiply(operand=...)
```

---

#### `ones(rows, cols)`

Ones.

**Parameters:**
- `rows`: rows
- `cols`: cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.ones(rows=..., cols=...)
```

---

#### `power()`

Power.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.power()
```

---

#### `repeat(repeats, axis)`

Repeat.

**Parameters:**
- `repeats`: repeats
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.repeat(repeats=..., axis=...)
```

---

#### `requires_grad_(requires_grad)`

Requires Grad .

**Parameters:**
- `requires_grad`: requires grad

**Returns:**
- `None`: Return value

**Example:**
```python
obj.requires_grad_(requires_grad=...)
```

---

#### `reshape(new_rows, new_cols)`

Reshape.

**Parameters:**
- `new_rows`: new rows
- `new_cols`: new cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.reshape(new_rows=..., new_cols=...)
```

---

#### `rich_display()`

Rich Display.

**Returns:**
- `str`: Return value

**Example:**
```python
obj.rich_display()
```

---

#### `scalar_multiply(scalar)`

Scalar Multiply.

**Parameters:**
- `scalar`: scalar

**Returns:**
- `None`: Return value

**Example:**
```python
obj.scalar_multiply(scalar=...)
```

---

#### `set(col, value)`

Set.

**Parameters:**
- `col`: col
- `value`: value

**Returns:**
- `None`: Return value

**Example:**
```python
obj.set(col=..., value=...)
```

---

#### `solve(b)`

Solve.

**Parameters:**
- `b`: b

**Returns:**
- `None`: Return value

**Example:**
```python
obj.solve(b=...)
```

---

#### `split(split_points, axis)`

Split.

**Parameters:**
- `split_points`: split points
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.split(split_points=..., axis=...)
```

---

#### `stack(other, axis)`

Stack.

**Parameters:**
- `other`: other
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.stack(other=..., axis=...)
```

---

#### `std_axis(axis)`

Std Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.std_axis(axis=...)
```

---

#### `sum_axis(axis)`

Sum Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.sum_axis(axis=...)
```

---

#### `tile(reps)`

Tile.

**Parameters:**
- `reps`: reps

**Returns:**
- `None`: Return value

**Example:**
```python
obj.tile(reps=...)
```

---

#### `to_base_array()`

To Base Array.

**Returns:**
- `BaseArray`: Return value

**Example:**
```python
obj.to_base_array()
```

---

#### `to_degree_matrix()`

To Degree Matrix.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_degree_matrix()
```

---

#### `to_laplacian()`

To Laplacian.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_laplacian()
```

---

#### `to_normalized_laplacian()`

To Normalized Laplacian.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_normalized_laplacian()
```

---

#### `to_table_for_streaming()`

To Table For Streaming.

**Returns:**
- `BaseTable`: Return value

**Example:**
```python
obj.to_table_for_streaming()
```

---

#### `trace()`

Trace.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.trace()
```

---

#### `var_axis(axis)`

Var Axis.

**Parameters:**
- `axis`: axis

**Returns:**
- `None`: Return value

**Example:**
```python
obj.var_axis(axis=...)
```

---

#### `zero_grad()`

Zero Grad.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.zero_grad()
```

---

#### `zeros(rows, cols)`

Zeros.

**Parameters:**
- `rows`: rows
- `cols`: cols

**Returns:**
- `None`: Return value

**Example:**
```python
obj.zeros(rows=..., cols=...)
```

---

