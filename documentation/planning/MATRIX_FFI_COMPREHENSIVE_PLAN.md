# 🚀 Matrix FFI Comprehensive Implementation Plan
**Intuitive, Chainable Matrix API for Groggy**

---

## 🎯 **Current State Analysis & Strategic Issues**

### Critical FFI Gaps Identified:
```python
# ❌ CURRENT ISSUES:
g.to_matrix()                    # Should be g.nodes.matrix() or g.matrix()
table.missing_matrix()           # No .matrix() method on tables
matrix_ops_in_core_not_ffi      # sum_axis, mean_axis not exposed
no_chaining_delegation          # Can't do matrix().power(3).eig()
no_functional_backend           # Neural ops (activations, conv) orphaned
```

### 🔍 **Root Cause Analysis:**
1. **Missing Accessor Pattern**: Matrix should be accessible via graph.nodes.matrix(), graph.edges.matrix()
2. **Incomplete FFI Coverage**: Core matrix operations not exposed to Python
3. **Broken Delegation Chain**: No chainable operations like `matrix().power(3).eig()`
4. **Neural Network Orphans**: Advanced operations in core but no Python access
5. **Table Integration Missing**: Tables should expose `.matrix()` for structured data

---

## 🏗️ **Comprehensive Matrix FFI Architecture**

### **1. Intuitive Access Patterns**
```python
# ✅ PROPOSED INTUITIVE API:

# Node-based matrices (most common)
g.nodes.matrix()                 # Node attribute matrix (current to_matrix)
g.nodes.matrix(['age', 'score']) # Specific attributes only

# Edge-based matrices  
g.edges.matrix()                 # Edge attribute matrix
g.edges.weight_matrix()          # Edge weights as matrix (default 'weight' attr)
g.edges.weight_matrix('strength') # Custom weight attribute

# Table-based matrices
table.matrix()                   # Table → matrix conversion
table.matrix(columns=['x', 'y']) # Specific columns

# Subgraph adjacency (primary location)
sg.adj()                         # Adjacency matrix (alias for adjacency_matrix)
sg.adjacency_matrix()            # Full adjacency matrix
sg.adjacency_list()              # Current adjacency format (renamed)

# Graph delegates to subgraph via __getattr__
g.adj()                          # Delegates to g.subgraph().adj()
g.adjacency_matrix()             # Delegates to g.subgraph().adjacency_matrix()
g.laplacian_matrix()             # Graph Laplacian  
g.incidence_matrix()             # Node-edge incidence
```

### **2. Chainable Operations (Delegation Pattern)**
```python
# ✅ MATRIX OPERATION CHAINING:

# Mathematical operations
g.nodes.matrix().power(3)                    # A³
g.adjacency_matrix().power(3).eig()          # Eigenvalues of A³
matrix @ other_matrix                        # Dot product semantics
matrix.transpose().multiply(other)           # Chain transpose

# Statistical operations  
matrix.sum_axis(0)                          # Sum rows (exposed from core)
matrix.mean_axis(1)                         # Mean columns (exposed from core)
matrix.std_axis(0).describe()               # Chain statistics

# Advanced linear algebra
matrix.svd()                                # Singular value decomposition
matrix.qr()                                 # QR decomposition
matrix.eig()                                # Eigendecomposition
matrix.solve(b)                             # Linear system solving

# Neural network operations
matrix.relu()                               # Element-wise ReLU
matrix.conv2d(kernel)                       # 2D convolution
matrix.softmax(axis=1)                      # Softmax activation
```

### **3. Functional Backend Integration**
```python
# ✅ NEURAL NETWORK API (groggy.neural as gnn):

import groggy.neural as gnn

# Neural network functions
gnn.relu(matrix)                              # Neural ReLU
gnn.conv2d(input, kernel, stride=1)           # 2D convolution
gnn.attention(query, key, value)              # Attention mechanism
gnn.batch_norm(x, gamma, beta)                # Batch normalization

# Advanced operations
gnn.softmax(logits, dim=-1)                   # Neural softmax
gnn.cross_entropy(preds, targets)             # Loss functions
gnn.adam_optimizer(params, gradients)         # Optimization

# Or method-based neural style
matrix.gnn.relu()                             # Method-based access
matrix.gnn.conv2d(kernel)                     # Chained neural ops
```

---

## 📐 **Detailed Implementation Plan**

### **Phase 1: Core Matrix Access API (Week 1)**

#### **1.1: Enhanced Matrix Accessors**
```rust
// python-groggy/src/ffi/storage/accessors.rs - EXPAND

#[pymethods]
impl PyNodesAccessor {
    /// Convert node attributes to matrix
    fn matrix(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let graph_ref = self.graph.borrow(py);
        let matrix = graph_ref.inner.borrow().to_matrix_f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Py::new(py, PyGraphMatrix { inner: matrix })
    }
    
    /// Matrix with specific attributes only
    fn matrix_attrs(&self, py: Python, attrs: Vec<String>) -> PyResult<Py<PyGraphMatrix>> {
        let graph_ref = self.graph.borrow(py);
        let matrix = graph_ref.inner.borrow().to_matrix_with_attrs_f64(&attrs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Py::new(py, PyGraphMatrix { inner: matrix })
    }
    
}

#[pymethods]
impl PyEdgesAccessor {
    /// Convert edge attributes to matrix  
    fn matrix(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let graph_ref = self.graph.borrow(py);
        let matrix = graph_ref.inner.borrow().edges_to_matrix_f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Py::new(py, PyGraphMatrix { inner: matrix })
    }
    
    /// Edge weight matrix (source × target with weights)
    /// Default to 'weight' attribute, but allow custom attribute selection
    fn weight_matrix(&self, py: Python, attr_name: Option<String>) -> PyResult<Py<PyGraphMatrix>> {
        let weight_attr = attr_name.unwrap_or_else(|| "weight".to_string());
        let graph_ref = self.graph.borrow(py);
        let matrix = graph_ref.inner.borrow().edge_weight_matrix(&weight_attr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Py::new(py, PyGraphMatrix { inner: matrix })
    }
}
```

#### **1.2: Table Matrix Integration**
```rust
// python-groggy/src/ffi/storage/graph_table.rs - ADD

#[pymethods]
impl PyGraphTable {
    /// Convert table to matrix format
    fn matrix(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let matrix = self.inner.borrow().to_matrix()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            
            Py::new(py, PyGraphMatrix { inner: matrix })
        })
    }
    
    /// Matrix with specific columns only
    fn matrix_columns(&self, py: Python, columns: Vec<String>) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let matrix = self.inner.borrow().to_matrix_with_columns(&columns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            
            Py::new(py, PyGraphMatrix { inner: matrix })
        })
    }
}
```

### **Phase 2: Chainable Matrix Operations (Week 2)**

#### **2.1: Enhanced PyGraphMatrix with Full Core API**
```rust
// python-groggy/src/ffi/storage/matrix.rs - EXPAND DRAMATICALLY

#[pymethods]
impl PyGraphMatrix {
    // ============= MATHEMATICAL OPERATIONS =============
    
    /// Matrix power (A^n)
    fn power(&self, py: Python, n: usize) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.power(n)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    /// Matrix multiplication (dot product)
    fn dot(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.multiply(&other.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    /// Element-wise multiplication
    fn multiply(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.elementwise_multiply(&other.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    /// Matrix addition
    fn add(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.add(&other.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    // ============= STATISTICAL OPERATIONS =============
    
    /// Sum along axis (EXPOSE FROM CORE)
    fn sum_axis(&self, py: Python, axis: usize) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            let axis_enum = if axis == 0 { 
                crate::storage::matrix::Axis::Rows 
            } else { 
                crate::storage::matrix::Axis::Columns 
            };
            
            self.inner.sum_axis(axis_enum)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        })
    }
    
    /// Mean along axis (EXPOSE FROM CORE)
    fn mean_axis(&self, py: Python, axis: usize) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            let axis_enum = if axis == 0 { 
                crate::storage::matrix::Axis::Rows 
            } else { 
                crate::storage::matrix::Axis::Columns 
            };
            
            self.inner.mean_axis(axis_enum)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        })
    }
    
    /// Standard deviation along axis (EXPOSE FROM CORE)
    fn std_axis(&self, py: Python, axis: usize) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            let axis_enum = if axis == 0 { 
                crate::storage::matrix::Axis::Rows 
            } else { 
                crate::storage::matrix::Axis::Columns 
            };
            
            self.inner.std_axis(axis_enum)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        })
    }
    
    // ============= LINEAR ALGEBRA =============
    
    /// Singular Value Decomposition
    fn svd(&self, py: Python) -> PyResult<(Py<PyGraphMatrix>, Vec<f64>, Py<PyGraphMatrix>)> {
        py.allow_threads(|| {
            let (u, s, vt) = self.inner.svd()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            
            Ok((
                Py::new(py, PyGraphMatrix { inner: u })?,
                s,
                Py::new(py, PyGraphMatrix { inner: vt })?
            ))
        })
    }
    
    /// QR Decomposition
    fn qr(&self, py: Python) -> PyResult<(Py<PyGraphMatrix>, Py<PyGraphMatrix>)> {
        py.allow_threads(|| {
            let (q, r) = self.inner.qr()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            
            Ok((
                Py::new(py, PyGraphMatrix { inner: q })?,
                Py::new(py, PyGraphMatrix { inner: r })?
            ))
        })
    }
    
    /// Eigenvalue decomposition  
    fn eig(&self, py: Python) -> PyResult<(Vec<f64>, Py<PyGraphMatrix>)> {
        py.allow_threads(|| {
            let (eigenvalues, eigenvectors) = self.inner.eig()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            
            Ok((
                eigenvalues,
                Py::new(py, PyGraphMatrix { inner: eigenvectors })?
            ))
        })
    }
    
    /// Solve linear system Ax = b
    fn solve(&self, py: Python, b: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.solve(&b.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    // ============= NEURAL NETWORK OPERATIONS =============
    
    /// ReLU activation function
    fn relu(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.relu()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    /// Softmax activation
    fn softmax(&self, py: Python, axis: usize) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.softmax(axis)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    /// 2D Convolution
    fn conv2d(&self, py: Python, kernel: &PyGraphMatrix, stride: usize, padding: usize) -> PyResult<Py<PyGraphMatrix>> {
        py.allow_threads(|| {
            let result = self.inner.conv2d(&kernel.inner, stride, padding)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Py::new(py, PyGraphMatrix { inner: result })
        })
    }
    
    // ============= PYTHON SPECIAL METHODS =============
    
    /// Matrix multiplication operator (@)
    fn __matmul__(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        self.dot(py, other)
    }
    
    /// Element-wise multiplication operator (*)
    fn __mul__(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        self.multiply(py, other)
    }
    
    /// Addition operator (+)
    fn __add__(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        self.add(py, other)
    }
}
```

### **Phase 3: Functional Backend (Week 3)**

#### **3.1: Functional Module Structure**
```rust
// python-groggy/src/ffi/functional/mod.rs - NEW MODULE

pub mod activations;
pub mod convolution;
pub mod optimization;
pub mod attention;
pub mod losses;

use pyo3::prelude::*;

#[pymodule]
pub fn functional(_py: Python, m: &PyModule) -> PyResult<()> {
    // Activation functions
    m.add_function(wrap_pyfunction!(activations::relu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(activations::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::tanh, m)?)?;
    
    // Convolution operations
    m.add_function(wrap_pyfunction!(convolution::conv2d, m)?)?;
    m.add_function(wrap_pyfunction!(convolution::conv1d, m)?)?;
    m.add_function(wrap_pyfunction!(convolution::max_pool2d, m)?)?;
    m.add_function(wrap_pyfunction!(convolution::avg_pool2d, m)?)?;
    
    // Attention mechanisms
    m.add_function(wrap_pyfunction!(attention::scaled_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(attention::multi_head, m)?)?;
    
    // Optimization functions
    m.add_function(wrap_pyfunction!(optimization::adam_step, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::sgd_step, m)?)?;
    
    // Loss functions
    m.add_function(wrap_pyfunction!(losses::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(losses::mse_loss, m)?)?;
    
    Ok(())
}
```

#### **3.2: Activation Functions Implementation**
```rust
// python-groggy/src/ffi/functional/activations.rs - NEW

use pyo3::prelude::*;
use crate::ffi::storage::matrix::PyGraphMatrix;

/// Functional ReLU activation
#[pyfunction]
pub fn relu(py: Python, input: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    py.allow_threads(|| {
        let result = input.inner.relu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Py::new(py, PyGraphMatrix { inner: result })
    })
}

/// Functional GELU activation  
#[pyfunction]
pub fn gelu(py: Python, input: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    py.allow_threads(|| {
        let result = input.inner.gelu()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Py::new(py, PyGraphMatrix { inner: result })
    })
}

/// Functional Softmax activation
#[pyfunction]
pub fn softmax(py: Python, input: &PyGraphMatrix, dim: Option<usize>) -> PyResult<Py<PyGraphMatrix>> {
    py.allow_threads(|| {
        let axis = dim.unwrap_or(1);
        let result = input.inner.softmax(axis)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Py::new(py, PyGraphMatrix { inner: result })
    })
}

/// Functional Sigmoid activation
#[pyfunction]
pub fn sigmoid(py: Python, input: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    py.allow_threads(|| {
        let result = input.inner.sigmoid()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Py::new(py, PyGraphMatrix { inner: result })
    })
}

/// Functional Tanh activation
#[pyfunction]
pub fn tanh(py: Python, input: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
    py.allow_threads(|| {
        let result = input.inner.tanh()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Py::new(py, PyGraphMatrix { inner: result })
    })
}
```

### **Phase 4: Core Matrix Operations Implementation (Week 4)**

#### **4.1: Missing Core Operations** 
```rust
// src/storage/matrix/matrix_core.rs - ADD MISSING OPERATIONS

impl<T: NumericType> GraphMatrix<T> 
where 
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + PartialOrd
{
    /// Matrix power (A^n)
    pub fn power(&self, n: usize) -> GraphResult<GraphMatrix<T>> {
        if n == 0 {
            let (size, _) = self.shape();
            return Self::identity(size);
        }
        
        let mut result = self.clone();
        for _ in 1..n {
            result = result.multiply(self)?;
        }
        
        Ok(result)
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        let (rows1, cols1) = self.shape();
        let (rows2, cols2) = other.shape();
        
        if rows1 != rows2 || cols1 != cols2 {
            return Err(GraphError::InvalidInput("Matrix dimensions must match for addition".into()));
        }
        
        let mut result = GraphMatrix::zeros(rows1, cols1);
        for row in 0..rows1 {
            for col in 0..cols1 {
                let a = self.get(row, col).unwrap_or(T::zero());
                let b = other.get(row, col).unwrap_or(T::zero());
                result.set(row, col, a + b)?;
            }
        }
        
        Ok(result)
    }
    
    /// ReLU activation (element-wise max(0, x))
    pub fn relu(&self) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        let mut result = GraphMatrix::zeros(rows, cols);
        
        for row in 0..rows {
            for col in 0..cols {
                let value = self.get(row, col).unwrap_or(T::zero());
                let relu_value = if value > T::zero() { value } else { T::zero() };
                result.set(row, col, relu_value)?;
            }
        }
        
        Ok(result)
    }
    
    /// Softmax activation along axis  
    pub fn softmax(&self, axis: usize) -> GraphResult<GraphMatrix<T>> 
    where 
        T: Into<f64> + From<f64>
    {
        let (rows, cols) = self.shape();
        let mut result = self.clone();
        
        if axis == 0 {
            // Softmax along rows (each column sums to 1)
            for col in 0..cols {
                let mut max_val = f64::NEG_INFINITY;
                for row in 0..rows {
                    let val: f64 = self.get(row, col).unwrap_or(T::zero()).into();
                    max_val = max_val.max(val);
                }
                
                let mut sum = 0.0;
                for row in 0..rows {
                    let val: f64 = self.get(row, col).unwrap_or(T::zero()).into();
                    let exp_val = (val - max_val).exp();
                    result.set(row, col, T::from(exp_val))?;
                    sum += exp_val;
                }
                
                for row in 0..rows {
                    let current: f64 = result.get(row, col).unwrap_or(T::zero()).into();
                    result.set(row, col, T::from(current / sum))?;
                }
            }
        } else {
            // Softmax along columns (each row sums to 1)
            for row in 0..rows {
                let mut max_val = f64::NEG_INFINITY;
                for col in 0..cols {
                    let val: f64 = self.get(row, col).unwrap_or(T::zero()).into();
                    max_val = max_val.max(val);
                }
                
                let mut sum = 0.0;
                for col in 0..cols {
                    let val: f64 = self.get(row, col).unwrap_or(T::zero()).into();
                    let exp_val = (val - max_val).exp();
                    result.set(row, col, T::from(exp_val))?;
                    sum += exp_val;
                }
                
                for col in 0..cols {
                    let current: f64 = result.get(row, col).unwrap_or(T::zero()).into();
                    result.set(row, col, T::from(current / sum))?;
                }
            }
        }
        
        Ok(result)
    }
}
```

---

## 🎯 **Implementation Priorities & Timeline**

### **🔥 IMMEDIATE (Week 1): Core Access Patterns**
```bash
Priority: CRITICAL - Foundation for all other work

Tasks:
1. ✅ Fix g.nodes.matrix() accessor pattern 
2. ✅ Add g.edges.matrix() and table.matrix()
3. ✅ Expose sum_axis(), mean_axis(), std_axis() to Python
4. ✅ Basic chainable operations (power, multiply, add)
5. ✅ Python operator overloading (@ for matmul, * for elementwise)

Success Criteria:
- g.nodes.matrix().power(3) works
- matrix @ other_matrix works  
- matrix.sum_axis(0) returns real data
```

### **⚡ HIGH (Week 2): Advanced Operations**
```bash
Priority: HIGH - Completes linear algebra functionality

Tasks:
1. SVD, QR, Eigenvalue decomposition (implement or delegate)
2. Linear system solving (Ax = b)
3. Matrix decompositions for ML/graph algorithms
4. Advanced statistical operations
5. Performance optimization for chainable ops

Success Criteria:
- sg.adj().power(3).eig() works
- g.adj() delegates to subgraph properly
- matrix.svd() returns proper decomposition
- Performance acceptable for 1000×1000 matrices
```

### **🧠 MEDIUM (Week 3): Neural Network Integration**  
```bash
Priority: MEDIUM - Enables AI/ML workflows

Tasks:
1. Neural network module (groggy.neural as nn)
2. Activation functions (ReLU, GELU, Softmax, etc.)
3. Basic convolution operations  
4. Loss functions and optimization primitives
5. Integration with existing neural network core

Success Criteria:
- nn.relu(matrix) works functionally
- matrix.relu() works as method
- nn.conv2d(input, kernel) basic implementation
- sg.adj() returns adjacency matrix
- g.adj() delegates to g.subgraph().adj()
- Neural network layers can be built
```

### **🔬 LOW (Week 4): Polish & Optimization**
```bash
Priority: LOW - Production readiness

Tasks:
1. Performance optimization and benchmarking
2. Memory management improvements
3. Error handling and edge cases
4. Documentation and examples
5. Integration testing with existing workflows

Success Criteria:
- Performance comparable to NumPy for basic ops
- Comprehensive error handling
- Production-ready stability
- Complete API documentation
```

---

## 🎪 **Smart Implementation Strategy**

### **1. Leverage Existing Core Work**
```rust
// ✅ ALREADY IMPLEMENTED in core:
sum_axis(), mean_axis(), std_axis()    // Just need FFI exposure
transpose(), multiply(), shape()       // Working in core
GraphMatrix<T> infrastructure         // Solid foundation

// 🚧 NEED TO IMPLEMENT in core:
power(), add(), relu(), softmax()     // Core mathematical ops
svd(), qr(), eig()                    // Advanced linear algebra
conv2d(), attention()                 // Neural network primitives
```

### **2. Incremental FFI Exposure**
```python
# Phase 1: Basic accessor patterns
g.nodes.matrix()           # ✅ Implement first
g.edges.matrix()           # ✅ Easy addition 
table.matrix()             # ✅ Table integration

# Phase 2: Core operation exposure  
matrix.sum_axis(0)         # ✅ Already in core
matrix.mean_axis(1)        # ✅ Already in core
matrix.power(3)            # 🚧 Need core implementation

# Phase 3: Advanced operations
matrix.svd()               # 🚧 Major implementation 
matrix @ other             # ✅ Python operator overloading
nn.relu(matrix)            # 🚧 Neural interface
```

### **3. Strategic Delegation Design**
```python
# ✅ FOLLOW GROGGY PATTERNS:

# Accessor-based matrix access
g.nodes.matrix()                    # Consistent with g.nodes.filter()
g.edges.matrix()                    # Consistent with g.edges.where()
table.matrix()                      # Consistent with table.agg()

# Chainable operations
matrix.power(3).transpose().eig()   # Delegation chain like arrays
matrix @ other_matrix               # Python semantics
nn.relu(matrix).softmax(dim=1)      # Neural chaining

# Performance-first implementation
matrix.sum_axis(0)                  # Delegate to optimized core
matrix.mean_axis(1)                 # Use SIMD where possible
```

---

## 📋 **Concrete Next Steps**

### **🎯 IMMEDIATE ACTION PLAN (This Week):**

1. **Fix Matrix Access Patterns** (Day 1-2)
   ```rust
   // Update python-groggy/src/ffi/storage/accessors.rs
   impl PyNodesAccessor {
       fn matrix(&self) -> PyResult<Py<PyGraphMatrix>> { ... }
   }
   ```

2. **Expose Core Statistical Operations** (Day 3-4)
   ```rust
   // Update python-groggy/src/ffi/storage/matrix.rs  
   impl PyGraphMatrix {
       fn sum_axis(&self, axis: usize) -> PyResult<Vec<f64>> { ... }
   }
   ```

3. **Basic Chainable Operations** (Day 5-6)
   ```rust
   // Add to matrix_core.rs and expose via FFI
   fn power(&self, n: usize) -> GraphResult<GraphMatrix<T>> { ... }
   fn add(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> { ... }
   ```

4. **Python Operator Overloading** (Day 7)
   ```rust
   // Add to PyGraphMatrix
   fn __matmul__(&self, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> { ... }
   ```

### **🔧 TESTING STRATEGY:**
```python
# Create comprehensive test to validate new API
def test_matrix_api_comprehensive():
    g = groggy.Graph()
    # ... setup graph ...
    
    # Test accessor patterns
    node_matrix = g.nodes.matrix()
    edge_matrix = g.edges.matrix()
    
    # Test chainable operations  
    result = node_matrix.power(2).transpose()
    
    # Test statistical operations
    row_sums = node_matrix.sum_axis(0)
    col_means = node_matrix.mean_axis(1)
    
    # Test Python operators
    combined = node_matrix @ node_matrix.transpose()
    
    # Test functional interface
    activated = nn.relu(node_matrix)
    
    assert all operations work with real data
```

This comprehensive plan addresses all the blind spots you identified and creates a cohesive, intuitive matrix API that follows Groggy's delegation patterns while exposing the powerful core functionality to Python users.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create comprehensive matrix API demonstration script", "status": "completed", "activeForm": "Creating comprehensive matrix API demonstration script"}, {"content": "Analyze current matrix FFI gaps and create comprehensive implementation plan", "status": "completed", "activeForm": "Analyzing current matrix FFI gaps and creating comprehensive implementation plan"}]