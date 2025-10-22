# Comprehensive Matrix Operations Planning Document

## Overview
This document provides a complete roadmap for matrix operations in Groggy, comparing against industry standards (NumPy, PyTorch, TensorFlow, JAX, Eigen, BLAS/LAPACK) to ensure comprehensive coverage.

## Current Implementation Status

### ✅ IMPLEMENTED (Core)

#### **Basic Operations**
- [x] `zeros(rows, cols)` - Zero matrix creation
  ```python
  # Create 3x3 zero matrix
  m = groggy.GraphMatrix.zeros(3, 3)
  print(m.shape())  # (3, 3)
  ```

- [x] `ones(rows, cols)` - Ones matrix creation  
  ```python
  # Create 2x4 ones matrix
  m = groggy.GraphMatrix.ones(2, 4)
  # All elements are 1.0
  ```

- [x] `eye(size)` / `identity(size)` - Identity matrix
  ```python
  # Create 4x4 identity matrix
  I = groggy.GraphMatrix.identity(4)
  # Diagonal elements are 1.0, off-diagonal are 0.0
  ```

- [x] `shape()` - Matrix dimensions
  ```python
  m = groggy.GraphMatrix.zeros(5, 3)
  rows, cols = m.shape()  # (5, 3)
  ```

- [x] `get(row, col)` / `set(row, col, value)` - Element access
  ```python
  m = groggy.GraphMatrix.zeros(3, 3)
  m.set(1, 2, 5.0)  # Set element at row 1, col 2
  value = m.get(1, 2)  # Returns 5.0
  ```

- [x] `get_row(row)` / `get_column(col)` - Row/column extraction
  ```python
  m = groggy.GraphMatrix.ones(4, 3)
  row_data = m.get_row(1)     # Returns [1.0, 1.0, 1.0]
  col_data = m.get_column(0)  # Returns [1.0, 1.0, 1.0, 1.0]
  ```

- [x] `transpose()` - Matrix transpose
  ```python
  m = groggy.GraphMatrix.from_data([[1, 2, 3], [4, 5, 6]])  # 2x3
  mt = m.transpose()  # 3x2 matrix
  print(mt.shape())   # (3, 2)
  ```

#### **Arithmetic Operations**
- [x] `matmul(other)` - Matrix multiplication (A @ B)
  ```python
  A = groggy.GraphMatrix.from_data([[1, 2], [3, 4]])  # 2x2
  B = groggy.GraphMatrix.from_data([[5, 6], [7, 8]])  # 2x2
  C = A.matmul(B)  # [[19, 22], [43, 50]]
  ```

- [x] `multiply(other)` - Matrix multiplication (alias)
  ```python
  # Same as matmul - for compatibility
  C = A.multiply(B)
  ```

- [x] `elementwise_multiply(other)` - Element-wise multiplication (A * B)
  ```python
  A = groggy.GraphMatrix.from_data([[1, 2], [3, 4]])
  B = groggy.GraphMatrix.from_data([[2, 3], [4, 5]])
  C = A.elementwise_multiply(B)  # [[2, 6], [12, 20]]
  ```

- [x] `power(n)` - Matrix power (A^n)
  ```python
  A = groggy.GraphMatrix.from_data([[2, 1], [0, 2]])
  A_squared = A.power(2)  # A @ A
  A_cubed = A.power(3)    # A @ A @ A
  ```

- [x] `add()` / `mul()` / `scale()` - Basic arithmetic (via UnifiedMatrix)
  ```python
  A = groggy.GraphMatrix.ones(3, 3)
  B = A.scale(2.0)     # Multiply all elements by 2.0
  C = A.add(B)         # Element-wise addition
  ```

#### **Statistical Operations**
- [x] `sum_axis(axis)` - Sum along axis (rows/columns)
  ```python
  m = groggy.GraphMatrix.from_data([[1, 2, 3], [4, 5, 6]])
  row_sums = m.sum_axis(1)  # [6, 15] - sum along columns
  col_sums = m.sum_axis(0)  # [5, 7, 9] - sum along rows
  ```

- [x] `mean_axis(axis)` - Mean along axis
  ```python
  m = groggy.GraphMatrix.from_data([[2, 4, 6], [8, 10, 12]])
  row_means = m.mean_axis(1)  # [4.0, 10.0]
  col_means = m.mean_axis(0)  # [5.0, 7.0, 9.0]
  ```

- [x] `std_axis(axis)` - Standard deviation along axis
  ```python
  m = groggy.GraphMatrix.from_data([[1, 3, 5], [2, 4, 6]])
  row_stds = m.std_axis(1)   # Standard deviation for each row
  col_stds = m.std_axis(0)   # Standard deviation for each column
  ```

- [x] `sum()` / `max()` / `min()` - Global statistics (via UnifiedMatrix)
  ```python
  m = groggy.GraphMatrix.from_data([[1, 5, 3], [9, 2, 7]])
  total = m.sum()     # 27.0 - sum of all elements
  maximum = m.max()   # 9.0 - largest element
  minimum = m.min()   # 1.0 - smallest element
  ```

#### **Graph-Specific Operations**
- [x] `adjacency_from_edges()` - Create adjacency matrix from edges
  ```python
  # From graph with edges
  g = groggy.Graph()
  n1, n2, n3 = g.add_node(), g.add_node(), g.add_node()
  g.add_edge(n1, n2)
  g.add_edge(n2, n3)
  adj = g.view().adjacency_matrix()  # Binary adjacency matrix
  ```

- [x] `weighted_adjacency_from_edges()` - Weighted adjacency matrix
  ```python
  g = groggy.Graph()
  n1, n2 = g.add_node(), g.add_node()
  g.add_edge(n1, n2, weight=2.5)
  weighted_adj = g.view().adjacency_matrix()  # Uses edge weights
  ```

- [x] `to_laplacian()` - Graph Laplacian matrix
  ```python
  adj = g.view().adjacency_matrix()
  laplacian = adj.to_laplacian()  # L = D - A (degree - adjacency)
  # Useful for spectral graph analysis
  ```

- [x] `to_normalized_laplacian()` - Normalized Laplacian
  ```python
  adj = g.view().adjacency_matrix()
  norm_laplacian = adj.to_normalized_laplacian()  # L = I - D^(-1/2) A D^(-1/2)
  # Better numerical properties for eigenvalue analysis
  ```

- [x] `to_degree_matrix()` - Degree matrix
  ```python
  adj = g.view().adjacency_matrix()
  degree = adj.to_degree_matrix()  # Diagonal matrix with node degrees
  # D[i,i] = sum of row i in adjacency matrix
  ```

#### **Neural Network Operations**
- [x] `relu()` - ReLU activation
  ```python
  m = groggy.GraphMatrix.from_data([[-1, 2, -3], [4, -5, 6]])
  activated = m.relu()  # [[0, 2, 0], [4, 0, 6]]
  # Negative values become 0, positive values unchanged
  ```

- [x] `gelu()` - GELU activation
  ```python
  m = groggy.GraphMatrix.from_data([[-1, 0, 1], [2, -2, 3]])
  activated = m.gelu()  # Smooth activation function
  # GELU(x) = x * Φ(x) where Φ is standard normal CDF
  ```

- [x] `sigmoid()` / `tanh()` - Activation functions (via ActivationOps)
  ```python
  # Note: Currently via advanced_matrix backend
  from groggy.advanced_matrix import ActivationOps
  m = groggy.GraphMatrix.from_data([[-2, 0, 2]])
  sigmoid_result = ActivationOps.sigmoid(m.storage)
  tanh_result = ActivationOps.tanh(m.storage)
  ```

- [x] `conv2d()` - 2D convolution (placeholder)
  ```python
  # Currently placeholder - integration incomplete
  input_matrix = groggy.GraphMatrix.zeros(28, 28)  # Image-like data
  kernel = groggy.GraphMatrix.ones(3, 3)           # 3x3 filter
  # result = input_matrix.conv2d(kernel, config)   # TODO: Complete implementation
  ```

#### **Automatic Differentiation**
- [x] `requires_grad(bool)` - Enable gradient computation
  ```python
  # Enable gradient tracking
  A = groggy.GraphMatrix.ones(2, 2).requires_grad(True)
  B = groggy.GraphMatrix.zeros(2, 2).requires_grad(True)
  # Operations on A and B will be tracked in computation graph
  ```

- [x] `backward()` - Compute gradients via backpropagation
  ```python
  A = groggy.GraphMatrix.from_data([[1, 2], [3, 4]]).requires_grad(True)
  B = groggy.GraphMatrix.from_data([[2, 0], [1, 3]]).requires_grad(True)
  C = A.matmul(B)  # Forward pass
  loss = C.sum()   # Scalar output
  loss.backward()  # Compute gradients
  ```

- [x] `grad()` - Access gradient matrix
  ```python
  # After backward() call
  grad_A = A.grad()  # Gradient w.r.t A
  grad_B = B.grad()  # Gradient w.r.t B
  # Gradients have same shape as original matrices
  ```

- [x] Computation graph with forward/backward passes
  ```python
  # Automatic graph construction
  x = groggy.GraphMatrix.from_data([[1, 2]]).requires_grad(True)
  y = x.relu()           # Node in computation graph
  z = y.matmul(x.transpose())  # Another node
  loss = z.sum()         # Final node
  # Graph: x -> relu -> matmul -> sum
  loss.backward()        # Traverses graph in reverse
  ```

- [x] Gradient accumulation for multiple outputs
  ```python
  A = groggy.GraphMatrix.ones(2, 2).requires_grad(True)
  y1 = A.sum()
  y2 = A.matmul(A).sum()
  # Multiple backward passes accumulate gradients
  y1.backward()  # A.grad() = ones(2, 2)
  y2.backward()  # A.grad() += additional gradients
  ```

#### **Type System & Conversion**
- [x] `cast<T>()` - Type casting between numeric types
- [x] `dtype()` - Data type introspection
- [x] `FromAttrValue` trait - Convert from Groggy's attribute system
- [x] Backend selection (BLAS, NumPy, Rust native)

#### **Metadata & Display**
- [x] `column_names()` / `set_column_names()` - Column labeling
- [x] `row_labels()` / `set_row_labels()` - Row labeling
- [x] `preview(row_limit, col_limit)` - Matrix preview
- [x] `to_dense_html()` - Rich HTML representation with ellipses truncation
- [x] `summary_info()` - Matrix summary statistics

### 🚧 PARTIALLY IMPLEMENTED

#### **Slicing & Indexing**
- [x] Basic slicing infrastructure (`MatrixIndex`, `MatrixSlice`)
- [ ] Advanced indexing (boolean masks, fancy indexing)
- [ ] Slice assignment
- [ ] View vs copy semantics

#### **Advanced Backend Integration**
- [x] Backend selection framework
- [x] BLAS/LAPACK integration (partial)
- [ ] GPU acceleration (CUDA/ROCm)
- [ ] Distributed computing support

### ❌ MISSING (Critical Gaps)

## Missing Operations Analysis

### **Core Linear Algebra (HIGH PRIORITY)**

#### **Matrix Decompositions**
- [ ] `svd()` - Singular Value Decomposition
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[3, 2, 2], [2, 3, -2]])
  U, S, Vt = A.svd()
  # U: left singular vectors, S: singular values, Vt: right singular vectors^T
  # Verify: A ≈ U @ diag(S) @ Vt
  ```
  - Returns: (U, Σ, V^T) where A = UΣV^T
  - Use cases: Dimensionality reduction, pseudoinverse, rank computation
  - NumPy: `np.linalg.svd()`
  - PyTorch: `torch.svd()`

- [ ] `qr()` - QR Decomposition  
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[1, 2], [3, 4], [5, 6]])
  Q, R = A.qr()
  # Q: orthogonal matrix, R: upper triangular
  # Verify: A = Q @ R, Q^T @ Q = I
  ```
  - Returns: (Q, R) where A = QR, Q orthogonal, R upper triangular
  - Use cases: Solving linear systems, least squares
  - NumPy: `np.linalg.qr()`

- [ ] `lu()` - LU Decomposition
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
  P, L, U = A.lu()
  # P: permutation, L: lower triangular, U: upper triangular  
  # Verify: P @ A = L @ U
  ```
  - Returns: (P, L, U) where PA = LU
  - Use cases: Solving linear systems, matrix inversion
  - SciPy: `scipy.linalg.lu()`

- [ ] `cholesky()` - Cholesky Decomposition
  ```python
  # PLANNED API - for positive definite matrices
  A = groggy.GraphMatrix.from_data([[4, 2], [2, 3]])  # Must be pos. def.
  L = A.cholesky()
  # L: lower triangular, A = L @ L^T
  # Much faster than LU for symmetric positive definite systems
  ```
  - For positive definite matrices: A = L L^T
  - Use cases: Solving symmetric positive definite systems
  - NumPy: `np.linalg.cholesky()`

- [ ] `eigenvalues()` / `eigenvectors()` - Eigendecomposition
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[3, 1], [0, 2]])
  eigenvals = A.eigenvalues()          # [3.0, 2.0]
  eigenvals, eigenvecs = A.eigenvectors()  # Returns both
  # AV = λV where V are eigenvectors, λ are eigenvalues
  ```
  - Returns: (λ, V) where AV = λV  
  - Use cases: Principal Component Analysis, graph analysis
  - NumPy: `np.linalg.eig()`, `np.linalg.eigvals()`

#### **Matrix Properties & Solutions**
- [ ] `inverse()` / `pinv()` - Matrix inverse and pseudoinverse
  - Current: Placeholder in FFI, not implemented
  - NumPy: `np.linalg.inv()`, `np.linalg.pinv()`

- [ ] `determinant()` - Matrix determinant
  - Current: Placeholder returning None
  - NumPy: `np.linalg.det()`

- [ ] `rank()` - Matrix rank
  - NumPy: `np.linalg.matrix_rank()`

- [ ] `condition()` - Condition number
  - NumPy: `np.linalg.cond()`

- [ ] `solve(B)` - Solve linear system Ax = B
  - NumPy: `np.linalg.solve()`

#### **Shape Operations (CRITICAL)**
- [ ] `reshape(new_shape)` - **USER IDENTIFIED AS MISSING**
  ```python
  # PLANNED API - CRITICAL MISSING OPERATION
  A = groggy.GraphMatrix.from_data([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4
  B = A.reshape((4, 2))  # Reshape to 4x2: [[1, 2], [3, 4], [5, 6], [7, 8]]
  C = A.reshape((8, 1))  # Column vector: [[1], [2], ..., [8]]
  D = A.reshape((1, 8))  # Row vector: [[1, 2, 3, 4, 5, 6, 7, 8]]
  # Total elements must remain the same: 2*4 = 4*2 = 8*1 = 1*8
  ```
  - Fundamental for tensor operations
  - NumPy: `np.reshape()`, PyTorch: `tensor.reshape()`

- [ ] `flatten()` / `ravel()` - Convert to 1D
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[1, 2], [3, 4]])
  flat = A.flatten()    # 1D: [1, 2, 3, 4]
  ravel = A.ravel()     # Same but potentially view (not copy)
  ```
  - NumPy: `np.flatten()`, `np.ravel()`

- [ ] `squeeze()` / `unsqueeze()` - Remove/add dimensions
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[1], [2], [3]])  # 3x1
  squeezed = A.squeeze()     # Remove size-1 dims: 1D [1, 2, 3]
  unsqueezed = squeezed.unsqueeze(0)  # Add dim at pos 0: 1x3
  unsqueezed = squeezed.unsqueeze(1)  # Add dim at pos 1: 3x1
  ```
  - PyTorch: `tensor.squeeze()`, `tensor.unsqueeze()`

- [ ] `broadcast_to(shape)` - Broadcasting to target shape
  ```python
  # PLANNED API
  A = groggy.GraphMatrix.from_data([[1, 2, 3]])  # 1x3
  B = A.broadcast_to((4, 3))  # 4x3: [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
  # Broadcasting rules: dims of size 1 can be expanded
  ```
  - NumPy: `np.broadcast_to()`

#### **Advanced Slicing & Indexing**
- [ ] Boolean indexing: `matrix[mask]`
- [ ] Fancy indexing: `matrix[row_indices, col_indices]`  
- [ ] Slice assignment: `matrix[1:3, 2:5] = values`
- [ ] Multi-dimensional indexing
- [ ] Broadcasting in assignment operations

#### **Element-wise Math Functions**
- [ ] `exp()` / `log()` / `sqrt()` - Mathematical functions
  - Current: Implemented in advanced_matrix but not exposed
- [ ] `abs()` - Absolute value
- [ ] `sin()` / `cos()` / `tan()` - Trigonometric functions  
- [ ] `ceil()` / `floor()` / `round()` - Rounding functions
- [ ] `clamp(min, max)` - Value clamping
- [ ] `nan_to_num()` - NaN/Inf handling

#### **Advanced Statistical Operations**
- [ ] `median()` / `percentile()` - Robust statistics
- [ ] `var()` - Variance (has std, but not var)
- [ ] `corrcoef()` - Correlation coefficient matrix
- [ ] `cov()` - Covariance matrix
- [ ] `histogram()` - Histogram computation

#### **Sorting & Selection**
- [ ] `sort()` / `argsort()` - Sorting operations
- [ ] `topk()` - Top-k selection
- [ ] `searchsorted()` - Binary search

#### **Concatenation & Stacking**
- [ ] `concat()` / `stack()` - Join matrices
- [ ] `split()` / `chunk()` - Split matrices
- [ ] `tile()` / `repeat()` - Repetition operations

### **Neural Network Extensions (MEDIUM PRIORITY)**

#### **Activation Functions (Expand Current)**
- [x] ReLU, GELU (implemented)
- [ ] `leaky_relu()` / `elu()` / `selu()` - Advanced activations
- [ ] `softmax()` / `log_softmax()` - Softmax variants
- [ ] `swish()` / `mish()` - Modern activations

#### **Loss Functions**
- [ ] `mse_loss()` - Mean Squared Error
- [ ] `cross_entropy_loss()` - Cross Entropy Loss
- [ ] `binary_cross_entropy()` - Binary Cross Entropy
- [ ] `huber_loss()` - Huber Loss

#### **Convolution Extensions**
- [x] Basic conv2d (placeholder)
- [ ] `conv1d()` / `conv3d()` - 1D/3D convolutions
- [ ] `transpose_conv2d()` - Transposed convolution
- [ ] `max_pool2d()` / `avg_pool2d()` - Pooling operations
- [ ] `batch_norm()` / `layer_norm()` - Normalization

### **Performance & Memory (MEDIUM PRIORITY)**

#### **Sparse Matrix Support**
- [ ] Sparse matrix formats (COO, CSR, CSC)
- [ ] Sparse-dense operations  
- [ ] Sparse linear algebra
- Current: Only basic `is_sparse()` check

#### **Memory Management**
- [ ] In-place operations (`add_()`, `mul_()` suffixed variants)
- [ ] Memory-mapped matrices for large datasets
- [ ] Lazy evaluation and operation fusion
- [ ] Memory usage profiling

#### **Parallel & GPU Acceleration**
- [ ] Multi-threading for CPU operations
- [ ] GPU kernels (CUDA/ROCm)
- [ ] Distributed matrix operations
- [ ] Backend-specific optimizations

### **Graph-Specific Extensions (LOW PRIORITY)**

#### **Advanced Graph Matrices**
- [ ] `transition_matrix()` - Random walk transition matrix
- [ ] `modularity_matrix()` - Community detection
- [ ] `distance_matrix()` - All-pairs shortest paths
- [ ] `centrality_matrices()` - Various centrality measures

#### **Spectral Graph Theory**
- [ ] `spectral_embedding()` - Graph embedding via eigendecomposition
- [ ] `graph_fourier_transform()` - Graph signal processing
- [ ] `chebyshev_polynomials()` - For graph convolutions

## Implementation Priority Matrix

### **Phase 1: Core Linear Algebra (Next 2-4 weeks)**
1. **reshape()** - Critical missing operation identified by user
2. **SVD/QR/LU decompositions** - Foundation for advanced operations
3. **inverse()** / **determinant()** - Complete existing placeholders
4. **solve()** - Linear system solving
5. **eigenvalues()** / **eigenvectors()** - Graph analysis needs

### **Phase 2: Tensor Operations (4-6 weeks)**  
1. **Advanced indexing** - Boolean and fancy indexing
2. **Broadcasting** - NumPy-style broadcasting system
3. **Concatenation/stacking** - Data manipulation primitives
4. **Element-wise math** - Complete math function coverage
5. **In-place operations** - Memory efficiency

### **Phase 3: Performance & GPU (6-8 weeks)**
1. **Sparse matrix support** - Large graph efficiency
2. **GPU acceleration** - CUDA/ROCm backends
3. **Operation fusion** - Compiler optimizations
4. **Memory mapping** - Large dataset support

### **Phase 4: Domain-Specific (8+ weeks)**
1. **Advanced neural operations** - Complete ML toolkit
2. **Graph-specific matrices** - Specialized graph algorithms
3. **Statistical extensions** - Data science operations

## FFI Integration Strategy

### **Current FFI Status**
- Basic operations: ✅ Fully exposed
- Statistical operations: ✅ Mostly exposed  
- Neural operations: 🚧 Partially exposed
- Linear algebra: ❌ Major gaps (inverse, determinant placeholders)

### **FFI Priority Mapping**
1. **High Priority FFI**: reshape, SVD, QR, inverse, determinant, solve
2. **Medium Priority FFI**: Broadcasting, advanced indexing, element-wise math
3. **Low Priority FFI**: GPU operations, sparse matrices, domain-specific

## Testing Strategy

### **Current Test Coverage**
- Basic operations: ✅ Well tested
- Statistical operations: ✅ Good coverage
- Neural operations: 🚧 Basic tests
- Advanced operations: ❌ Missing tests

### **Required Test Categories**
1. **Numerical accuracy** - Compare against NumPy/SciPy
2. **Performance benchmarks** - BLAS/backend comparisons  
3. **Memory usage** - Large matrix handling
4. **Edge cases** - Singular matrices, NaN/Inf handling
5. **Gradient correctness** - Automatic differentiation validation

## Benchmark Targets

### **Performance Goals**
- **Basic operations**: Within 10% of NumPy performance
- **BLAS operations**: Within 5% of OpenBLAS/MKL
- **GPU operations**: Within 20% of CuBLAS (when implemented)
- **Memory usage**: No more than 2x overhead vs. raw arrays

### **Accuracy Goals**
- **Decompositions**: Machine precision accuracy vs. LAPACK
- **Gradients**: Numerical gradient checking with 1e-6 tolerance
- **Stability**: Robust handling of ill-conditioned matrices

## Documentation Requirements

### **API Documentation**
- [ ] Complete docstrings for all operations
- [ ] Performance characteristics documentation
- [ ] Memory usage guidelines
- [ ] Backend selection guide

### **Examples & Tutorials**
- [ ] Linear algebra cookbook
- [ ] Neural network operations guide  
- [ ] Graph analysis examples
- [ ] Performance optimization guide

## Dependencies & Ecosystem

### **Core Dependencies**
- **BLAS/LAPACK**: Linear algebra backend
- **PyO3**: Python integration
- **Unified matrix backend**: Type system foundation

### **Optional Dependencies**
- **CUDA/ROCm**: GPU acceleration
- **Intel MKL**: Optimized BLAS
- **SuiteSparse**: Sparse matrix operations

### **Python Ecosystem Integration**
- **NumPy compatibility**: Array protocol support
- **SciPy integration**: Advanced scientific computing
- **PyTorch interop**: Neural network ecosystem
- **JAX compatibility**: Automatic differentiation ecosystem

## Success Metrics

### **Completion Criteria**
1. **Feature parity**: 90% coverage vs. NumPy linear algebra module
2. **Performance**: Within benchmarked targets
3. **API consistency**: Pythonic, well-documented interface
4. **Test coverage**: >95% line coverage with numerical validation
5. **User adoption**: Successfully handles real-world graph analysis workloads

### **Quality Gates**
- All decompositions pass numerical accuracy tests
- Gradient computation validates against finite differences
- Memory usage stays within acceptable bounds
- Performance meets or exceeds benchmarks
- Documentation is complete and accurate

---

## Immediate Next Steps

1. **Implement reshape()** - User's immediate need
2. **Complete inverse()/determinant()** - Fix existing placeholders
3. **Add SVD implementation** - Foundation for many operations
4. **Create comprehensive test suite** - Validate existing operations
5. **Document current API** - Establish baseline before expansion

## API Organization & FFI Integration Examples

### **Python API Structure**
```python
# Proposed organization for clear API discoverability

# Core GraphMatrix class with method categories
class GraphMatrix:
    # === CREATION METHODS ===
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'GraphMatrix': ...
    @classmethod
    def ones(cls, rows: int, cols: int) -> 'GraphMatrix': ...
    @classmethod
    def identity(cls, size: int) -> 'GraphMatrix': ...
    @classmethod
    def from_data(cls, data: List[List[float]]) -> 'GraphMatrix': ...
    
    # === SHAPE & ACCESS ===
    def shape(self) -> Tuple[int, int]: ...
    def reshape(self, new_shape: Tuple[int, int]) -> 'GraphMatrix': ...  # MISSING
    def get(self, row: int, col: int) -> float: ...
    def set(self, row: int, col: int, value: float) -> None: ...
    def transpose(self) -> 'GraphMatrix': ...
    
    # === ARITHMETIC ===
    def matmul(self, other: 'GraphMatrix') -> 'GraphMatrix': ...
    def elementwise_multiply(self, other: 'GraphMatrix') -> 'GraphMatrix': ...
    def add(self, other: 'GraphMatrix') -> 'GraphMatrix': ...
    def scale(self, scalar: float) -> 'GraphMatrix': ...
    
    # === LINEAR ALGEBRA ===
    def inverse(self) -> 'GraphMatrix': ...  # NEEDS IMPLEMENTATION
    def determinant(self) -> float: ...      # NEEDS IMPLEMENTATION
    def svd(self) -> Tuple['GraphMatrix', List[float], 'GraphMatrix']: ...  # MISSING
    def qr(self) -> Tuple['GraphMatrix', 'GraphMatrix']: ...               # MISSING
    def eigenvalues(self) -> List[float]: ...                              # MISSING
    def solve(self, b: 'GraphMatrix') -> 'GraphMatrix': ...                # MISSING
    
    # === STATISTICS ===
    def sum(self) -> float: ...
    def mean(self) -> float: ...
    def sum_axis(self, axis: int) -> List[float]: ...
    def mean_axis(self, axis: int) -> List[float]: ...
    def std_axis(self, axis: int) -> List[float]: ...
    
    # === NEURAL NETWORK ===
    def relu(self) -> 'GraphMatrix': ...
    def gelu(self) -> 'GraphMatrix': ...
    def sigmoid(self) -> 'GraphMatrix': ...  # NEEDS FFI EXPOSURE
    def tanh(self) -> 'GraphMatrix': ...     # NEEDS FFI EXPOSURE
    
    # === AUTOMATIC DIFFERENTIATION ===
    def requires_grad(self, req: bool) -> 'GraphMatrix': ...
    def backward(self) -> None: ...
    def grad(self) -> Optional['GraphMatrix']: ...
    
    # === GRAPH-SPECIFIC ===
    def to_laplacian(self) -> 'GraphMatrix': ...
    def to_degree_matrix(self) -> 'GraphMatrix': ...
    def to_normalized_laplacian(self) -> 'GraphMatrix': ...
```

### **FFI Implementation Priority with Examples**

#### **Phase 1: Critical Missing Operations**
```rust
// src/ffi/storage/matrix.rs additions needed

#[pymethods]
impl PyGraphMatrix {
    fn reshape(&self, py: Python, new_shape: (usize, usize)) -> PyResult<Py<Self>> {
        let (new_rows, new_cols) = new_shape;
        let current_elements = self.inner.shape().0 * self.inner.shape().1;
        let new_elements = new_rows * new_cols;
        
        if current_elements != new_elements {
            return Err(PyValueError::new_err(
                format!("Cannot reshape matrix with {} elements to shape ({}, {})", 
                        current_elements, new_rows, new_cols)
            ));
        }
        
        // Implementation: extract all elements and create new matrix
        // This is the critical missing operation user identified
        todo!("Implement reshape functionality")
    }
    
    fn svd(&self, py: Python) -> PyResult<(Py<Self>, PyObject, Py<Self>)> {
        // Singular Value Decomposition
        // Returns (U, S, Vt) where A = U @ diag(S) @ Vt
        py.allow_threads(|| {
            // Call LAPACK dgesvd or equivalent
            self.inner.svd()
                .map(|(u, s, vt)| {
                    let u_py = Py::new(py, PyGraphMatrix { inner: u })?;
                    let s_py = s.to_object(py);  // Vector of singular values
                    let vt_py = Py::new(py, PyGraphMatrix { inner: vt })?;
                    Ok((u_py, s_py, vt_py))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
    
    fn inverse(&self, py: Python) -> PyResult<Py<Self>> {
        // Complete the existing placeholder
        py.allow_threads(|| {
            self.inner.inverse()
                .map(|result| Py::new(py, PyGraphMatrix { inner: result }))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }
}
```

#### **Phase 2: Advanced Indexing Examples**
```python
# Planned advanced indexing API
A = groggy.GraphMatrix.from_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Boolean indexing
mask = A > 5  # Returns boolean matrix
filtered = A[mask]  # Returns 1D array of values > 5

# Fancy indexing  
row_indices = [0, 2]    # Select rows 0 and 2
col_indices = [1, 2]    # Select columns 1 and 2
submatrix = A[row_indices, col_indices]  # 2x2 submatrix

# Slice assignment
A[1:3, 0:2] = groggy.GraphMatrix.zeros(2, 2)  # Set subregion to zeros
```

#### **Phase 3: Performance Integration Examples**
```python
# Backend selection for performance
A = groggy.GraphMatrix.zeros(1000, 1000)

# Use BLAS for large matrix multiplication
A.set_backend('blas')  # OpenBLAS/MKL acceleration
B = A.matmul(A)        # Fast BLAS gemm

# Use GPU backend (future)
A.set_backend('cuda')  # GPU acceleration
C = A.matmul(B)        # cuBLAS acceleration

# Sparse operations (future)
sparse_A = A.to_sparse('csr')  # Convert to sparse format
result = sparse_A.matmul(dense_B)  # Sparse-dense multiplication
```

### **Testing Examples for Each Operation**
```python
# Each operation needs comprehensive testing
def test_reshape():
    A = groggy.GraphMatrix.from_data([[1, 2, 3, 4]])  # 1x4
    
    # Test valid reshapes
    B = A.reshape((2, 2))  # Should work: [[1, 2], [3, 4]]
    assert B.shape() == (2, 2)
    assert B.get(0, 0) == 1
    assert B.get(1, 1) == 4
    
    # Test invalid reshape
    with pytest.raises(ValueError):
        A.reshape((2, 3))  # 4 elements can't fit in 2x3
    
    # Test numerical consistency 
    original_sum = A.sum()
    reshaped_sum = B.sum()
    assert abs(original_sum - reshaped_sum) < 1e-10

def test_svd_accuracy():
    # Test against NumPy for numerical accuracy
    A_data = [[3, 2, 2], [2, 3, -2]]
    A = groggy.GraphMatrix.from_data(A_data)
    
    # Groggy SVD
    U, S, Vt = A.svd()
    
    # NumPy reference
    import numpy as np
    U_np, S_np, Vt_np = np.linalg.svd(A_data)
    
    # Check reconstruction: A ≈ U @ diag(S) @ Vt
    reconstructed = U.matmul(groggy.GraphMatrix.diag(S)).matmul(Vt)
    reconstruction_error = (A - reconstructed).abs().max()
    assert reconstruction_error < 1e-12  # Machine precision accuracy
```

This plan provides a roadmap for making Groggy's matrix operations comprehensive and competitive with established libraries while maintaining focus on graph-specific use cases.