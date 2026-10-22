//! GraphMatrix - High-performance matrix with neural network capabilities
//!
//! This module provides GraphMatrix with advanced optimization backend using
//! the NumericType trait system for mixed-precision computations.
//!
//! # Design Principles
//! - GraphMatrix<T> with generic NumericType support (f64 default)
//! - Built on UnifiedMatrix backend with BLAS/NumPy delegation
//! - Neural network operations: matmul, conv2d, activations, autodiff
//! - Memory-efficient storage with fusion optimization
//! - Intelligent backend selection for optimal performance

#[path = "factor_autodiff.rs"]
mod factor_autodiff;
#[path = "numerical_data.rs"]
mod numerical_data;
use crate::storage::advanced_matrix::neural::autodiff::{FactorOutput, Operation};
pub use numerical_data::NumericalMatrixData;

use crate::errors::{GraphError, GraphResult};
use crate::storage::advanced_matrix::{
    AdvancedMemoryPool, ComputationGraph, NumericType, UnifiedMatrix,
};
use crate::storage::array::NumArray;
use crate::storage::table::BaseTable;
use crate::types::{AttrValue, NodeId};
use crate::viz::display::DisplayConfig;
use std::collections::HashMap;
use std::sync::Arc;

/// Owned solution plus a rounded normwise backward-error estimate.
/// A small backward error alone does not imply a small forward error.
#[derive(Debug)]
pub struct SolveReport<T: NumericType> {
    pub solution: GraphMatrix<T>,
    pub backward_error: f64,
    pub accepted_refinements: usize,
}

/// Dedicated complex eigen result for real input; eigenvectors are columns.
#[derive(Debug)]
pub struct ComplexEigenResult<T: NumericType> {
    pub eigenvalues_real: Vec<f64>,
    pub eigenvalues_imag: Vec<f64>,
    pub eigenvectors_real: GraphMatrix<T>,
    pub eigenvectors_imag: GraphMatrix<T>,
}

/// Matrix properties that can be computed and cached
#[derive(Debug, Clone)]
pub struct MatrixProperties {
    pub is_square: bool,
    pub is_symmetric: bool,
    pub is_numeric: bool,
    pub is_sparse: bool,
    pub sparsity: Option<f64>, // Ratio of zero elements
}

impl MatrixProperties {
    pub fn analyze<T: NumericType>(matrix: &GraphMatrix<T>) -> Self
    where
        f64: From<T>,
    {
        let (rows, cols) = matrix.shape();
        let is_square = rows == cols;
        let is_numeric = true; // NumArray is always numeric

        // Check symmetry (only for square numeric matrices)
        let is_symmetric = if is_square && is_numeric && rows > 0 {
            matrix.is_symmetric_internal()
        } else {
            false
        };

        // Numerical zero density is distinct from actual sparse storage.
        let sparsity = matrix.zero_count().ok().and_then(|zeros| {
            rows.checked_mul(cols).map(|total| {
                if total == 0 {
                    0.0
                } else {
                    zeros as f64 / total as f64
                }
            })
        });
        let is_sparse = matrix.is_sparse();

        Self {
            is_square,
            is_symmetric,
            is_numeric,
            is_sparse,
            sparsity,
        }
    }
}

/// Axis enumeration for operations
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    Rows = 0,
    Columns = 1,
}

/// Join types for matrix operations
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
}

/// High-performance matrix with neural network capabilities
#[derive(Debug, Clone)]
pub struct GraphMatrix<T: NumericType = f64> {
    /// Core storage using advanced matrix system
    storage: UnifiedMatrix<T>,
    /// Column names/labels
    column_names: Vec<String>,
    /// Row labels (optional)
    row_labels: Option<Vec<String>>,

    /// Neural network state
    requires_grad: bool,
    computation_graph: Option<ComputationGraph<T>>,
    current_node_id: Option<NodeId>,

    #[allow(dead_code)]
    memory_pool: Arc<AdvancedMemoryPool<T>>,

    /// Cached matrix properties
    #[allow(dead_code)]
    properties: Option<MatrixProperties>,
    /// Reference to the source graph (optional)
    #[allow(dead_code)]
    graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
}

impl<T: NumericType> GraphMatrix<T> {
    /// Construct a canonical CSR matrix with owned validated storage.
    pub fn from_csr(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> GraphResult<Self> {
        UnifiedMatrix::from_csr(rows, cols, values, col_indices, row_pointers)
            .map(Self::from_storage)
            .map_err(|e| GraphError::InvalidInput(format!("CSR construction failed: {e:?}")))
    }

    /// Create GraphMatrix from UnifiedMatrix storage
    pub fn from_storage(storage: UnifiedMatrix<T>) -> Self {
        let shape = storage.shape();
        let (_rows, cols) = (shape.rows, shape.cols);
        let column_names = (0..cols).map(|i| format!("col_{}", i)).collect();

        Self {
            storage,
            column_names,
            row_labels: None,
            requires_grad: false,
            computation_graph: None,
            current_node_id: None,
            memory_pool: Arc::new(AdvancedMemoryPool::new()),
            properties: None,
            graph: None,
        }
    }

    /// Create zeros matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let storage = UnifiedMatrix::zeros(rows, cols).unwrap_or_else(|_| {
            // Fallback to minimal matrix on error
            UnifiedMatrix::new(1, 1).expect("Failed to create minimal matrix")
        });
        Self::from_storage(storage)
    }

    /// Create a zero matrix with specified dimensions and type (FFI compatibility)
    pub fn zeros_with_type(
        rows: usize,
        cols: usize,
        _attr_type: crate::types::AttrValueType,
    ) -> Self {
        // Ignore attr_type since T is already specified by the type parameter
        Self::zeros(rows, cols)
    }

    /// Create ones matrix
    pub fn ones(rows: usize, cols: usize) -> Self {
        let storage = UnifiedMatrix::ones(rows, cols).unwrap_or_else(|_| {
            // Fallback to minimal matrix on error
            UnifiedMatrix::new(1, 1).expect("Failed to create minimal matrix")
        });
        Self::from_storage(storage)
    }

    /// Check whether the matrix contains any data
    pub fn is_empty(&self) -> bool {
        let shape = self.storage.shape();
        shape.rows == 0 || shape.cols == 0
    }

    /// Create a square identity matrix.
    pub fn eye(size: usize) -> Self {
        let storage = UnifiedMatrix::identity(size)
            .unwrap_or_else(|_| UnifiedMatrix::new(1, 1).expect("Failed to create minimal matrix"));
        Self::from_storage(storage)
    }

    /// Get matrix shape
    pub fn shape(&self) -> (usize, usize) {
        let shape = self.storage.shape();
        (shape.rows, shape.cols)
    }

    /// Create GraphMatrix from row-major data
    pub fn from_row_major_data(
        data: Vec<T>,
        rows: usize,
        cols: usize,
        nodes: Option<&[NodeId]>,
    ) -> GraphResult<Self> {
        let storage = UnifiedMatrix::from_data(data, rows, cols)
            .map_err(|e| GraphError::InvalidInput(format!("Matrix creation failed: {:?}", e)))?;

        let (column_names, row_labels) = if let Some(node_ids) = nodes {
            let col_names = node_ids.iter().map(|id| format!("node_{}", id)).collect();
            let row_labels = node_ids.iter().map(|id| format!("node_{}", id)).collect();
            (col_names, Some(row_labels))
        } else {
            let col_names = (0..cols).map(|i| format!("col_{}", i)).collect();
            (col_names, None)
        };

        Ok(Self {
            storage,
            column_names,
            row_labels,
            requires_grad: false,
            computation_graph: None,
            current_node_id: None,
            memory_pool: Arc::new(AdvancedMemoryPool::new()),
            properties: None,
            graph: None,
        })
    }

    /// Create adjacency matrix from edge data
    pub fn adjacency_from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> GraphResult<Self> {
        Self::adjacency_from_edges_with_direction(nodes, edges, false)
    }

    /// Create adjacency matrix while preserving directed edge orientation.
    pub fn adjacency_from_edges_with_direction(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
        directed: bool,
    ) -> GraphResult<Self> {
        let size = nodes.len();
        if size == 0 {
            return Ok(Self::zeros(0, 0));
        }

        let node_to_index: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        let mut matrix_data = vec![T::zero(); size * size];

        for &(source, target) in edges {
            if let (Some(&src_idx), Some(&tgt_idx)) =
                (node_to_index.get(&source), node_to_index.get(&target))
            {
                matrix_data[src_idx * size + tgt_idx] = T::one();
                if !directed {
                    matrix_data[tgt_idx * size + src_idx] = T::one();
                }
            }
        }

        Self::from_row_major_data(matrix_data, size, size, Some(nodes))
    }

    /// Create weighted adjacency matrix from weighted edge data  
    pub fn weighted_adjacency_from_edges(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, T)],
    ) -> GraphResult<Self> {
        Self::weighted_adjacency_from_edges_with_direction(nodes, weighted_edges, false)
    }

    /// Create weighted adjacency while preserving directed edge orientation.
    pub fn weighted_adjacency_from_edges_with_direction(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, T)],
        directed: bool,
    ) -> GraphResult<Self> {
        let size = nodes.len();
        if size == 0 {
            return Ok(Self::zeros(0, 0));
        }

        let node_to_index: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        let mut matrix_data = vec![T::zero(); size * size];

        for &(source, target, weight) in weighted_edges {
            if let (Some(&src_idx), Some(&tgt_idx)) =
                (node_to_index.get(&source), node_to_index.get(&target))
            {
                matrix_data[src_idx * size + tgt_idx] = weight;
                if !directed {
                    matrix_data[tgt_idx * size + src_idx] = weight;
                }
            }
        }

        Self::from_row_major_data(matrix_data, size, size, Some(nodes))
    }

    /// Internal method to get column by index - needed by slicing module
    pub(crate) fn get_column_internal(&self, col_idx: usize) -> GraphResult<Vec<T>> {
        let (rows, cols) = self.shape();
        if col_idx >= cols {
            return Err(GraphError::InvalidInput(format!(
                "Column {} not found",
                col_idx
            )));
        }

        // Extract column using individual element access
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            let element = UnifiedMatrix::get(&self.storage, row, col_idx).map_err(|e| {
                GraphError::InvalidInput(format!(
                    "Cannot access element ({}, {}): {:?}",
                    row, col_idx, e
                ))
            })?;
            column.push(element);
        }
        Ok(column)
    }

    /// Set column names
    pub fn set_column_names(&mut self, names: Vec<String>) {
        let (_, cols) = self.shape();
        if names.len() == cols {
            self.column_names = names;
        } else {
            // Adjust names to match column count
            let names_len = names.len();
            self.column_names = names
                .into_iter()
                .take(cols)
                .chain((names_len..cols).map(|i| format!("col_{}", i)))
                .collect();
        }
    }

    /// Get column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Get row labels
    pub fn row_labels(&self) -> Option<&[String]> {
        self.row_labels.as_deref()
    }

    /// Set row labels
    pub fn set_row_labels(&mut self, labels: Option<Vec<String>>) {
        self.row_labels = labels;
    }

    /// Check if matrix is symmetric (internal helper)
    pub(crate) fn is_symmetric_internal(&self) -> bool
    where
        f64: From<T>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return false;
        }

        // Check if A[i,j] == A[j,i] for all i,j
        let tolerance = 1e-10;
        for i in 0..rows {
            for j in 0..i {
                // Only check lower triangle (j < i)
                // Get elements at (i,j) and (j,i)
                let a_ij_opt = self.get(i, j);
                let a_ji_opt = self.get(j, i);

                match (a_ij_opt, a_ji_opt) {
                    (Some(a_ij), Some(a_ji)) => {
                        // Both elements exist, compare them
                        if !self.values_are_equal_within_tolerance(a_ij, a_ji, tolerance) {
                            return false;
                        }
                    }
                    (None, None) => {
                        // Both elements are missing, this is symmetric for sparse matrices
                        continue;
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        // One element exists but the other doesn't, not symmetric
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Helper function to compare two values within tolerance
    fn values_are_equal_within_tolerance(&self, a: T, b: T, tolerance: f64) -> bool
    where
        f64: From<T>,
    {
        let a_f64 = f64::from(a);
        let b_f64 = f64::from(b);
        (a_f64 - b_f64).abs() < tolerance
    }

    /// Count actual zero values, including implicit CSR entries.
    pub fn zero_count(&self) -> GraphResult<usize> {
        let (rows, cols) = self.shape();
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| GraphError::InvalidInput("Matrix element count overflow".into()))?;
        if let crate::storage::advanced_matrix::unified_matrix::MatrixStorage::Sparse(sparse) =
            &self.storage.storage
        {
            if (sparse.shape.rows, sparse.shape.cols) != (rows, cols) {
                return Err(GraphError::InvalidInput(
                    "CSR shape metadata mismatch".into(),
                ));
            }
            sparse
                .validate()
                .map_err(|e| GraphError::InvalidInput(format!("Invalid CSR: {e:?}")))?;
            return Ok(count - sparse.values.len());
        }
        let mut zeros = 0;
        for row in 0..rows {
            for col in 0..cols {
                zeros += usize::from(self.get_checked(row, col)? == T::zero());
            }
        }
        Ok(zeros)
    }

    /// Generate dense HTML representation for matrices with truncation
    ///
    /// Creates a clean HTML table representation of the matrix for display in
    /// Jupyter notebooks and other HTML contexts. Always uses dense format but
    /// truncates large matrices with ellipses, showing first/last rows and columns.
    /// Now delegates to the core table display system for consistency.
    pub fn to_dense_html(&self) -> GraphResult<String> {
        // Convert matrix to table format for display
        let table = self.to_table()?;

        // Use dense matrix display configuration (no headers, truncated)
        let config = DisplayConfig::dense_matrix();

        // Delegate to table's rich display with the dense matrix configuration
        let html = table.rich_display(Some(config));

        Ok(html)
    }

    /// Convert matrix to table format for display purposes
    fn to_table(&self) -> GraphResult<BaseTable> {
        let (rows, cols) = self.shape();
        let mut columns = std::collections::HashMap::new();

        // Create columns with generic names (no headers will be shown anyway)
        for col_idx in 0..cols {
            let mut column_data = Vec::new();
            for row_idx in 0..rows {
                let value = self.get(row_idx, col_idx).unwrap_or(T::zero());
                // Convert to AttrValue for table storage
                let attr_value = AttrValue::Float(value.to_f64() as f32);
                column_data.push(attr_value);
            }
            let col_name = format!("col_{}", col_idx);
            let array = crate::storage::array::BaseArray::from_attr_values(column_data);
            columns.insert(col_name, array);
        }

        BaseTable::from_columns(columns)
    }
}

// Backward compatibility for f64 GraphMatrix
impl GraphMatrix<f64> {
    /// Broadcast a scalar addition without constructing a second matrix.
    pub fn scalar_add(&self, scalar: f64) -> GraphResult<Self> {
        self.map_arithmetic(|value| value + scalar)
    }

    /// Broadcast a scalar subtraction without constructing a second matrix.
    pub fn scalar_subtract(&self, scalar: f64) -> GraphResult<Self> {
        self.map_arithmetic(|value| value - scalar)
    }

    /// Divide values directly; reject either signed zero, allow other IEEE values.
    pub fn scalar_divide(&self, scalar: f64) -> GraphResult<Self> {
        if scalar == 0.0 {
            return Err(GraphError::InvalidInput("Division by zero".into()));
        }
        self.map_arithmetic(|value| value / scalar)
    }

    /// Create zeros matrix with f64 type (backward compatibility)
    pub fn zeros_f64(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }

    /// Create adjacency matrix from edges (backward compatibility)
    pub fn adjacency_from_edges_f64(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
    ) -> GraphResult<Self> {
        Self::adjacency_from_edges(nodes, edges)
    }

    /// Create weighted adjacency matrix (backward compatibility)
    pub fn weighted_adjacency_from_edges_f64(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, f64)],
    ) -> GraphResult<Self> {
        Self::weighted_adjacency_from_edges(nodes, weighted_edges)
    }

    /// Helper for row-major conversion (backward compatibility)
    pub fn from_row_major_data_f64(
        data: Vec<f64>,
        rows: usize,
        cols: usize,
        nodes: Option<&[NodeId]>,
    ) -> GraphResult<Self> {
        Self::from_row_major_data(data, rows, cols, nodes)
    }
}

// Neural Network Operations for GraphMatrix
impl<T: NumericType> GraphMatrix<T> {
    /// Matrix multiplication with backend optimization
    pub fn matmul(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        self.multiply(other)
    }

    /// ReLU activation function
    pub fn relu(&self) -> GraphResult<GraphMatrix<T>> {
        use crate::storage::advanced_matrix::neural::activations::ActivationOps;
        let result_storage = ActivationOps::relu(&self.storage)
            .map_err(|e| GraphError::InvalidInput(format!("ReLU activation failed: {:?}", e)))?;
        Ok(Self::from_storage(result_storage))
    }

    /// GELU activation function
    pub fn gelu(&self) -> GraphResult<GraphMatrix<T>> {
        use crate::storage::advanced_matrix::neural::activations::ActivationOps;
        let result_storage = ActivationOps::gelu(&self.storage)
            .map_err(|e| GraphError::InvalidInput(format!("GELU activation failed: {:?}", e)))?;
        Ok(Self::from_storage(result_storage))
    }

    /// Extract data as a Vec<T> in row-major order
    pub fn to_vec(&self) -> GraphResult<Vec<T>> {
        self.storage
            .to_vec()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }

    /// Leaky ReLU activation function
    /// Applies f(x) = max(alpha * x, x) where alpha is the negative slope
    pub fn leaky_relu(&self, alpha: Option<f64>) -> GraphResult<GraphMatrix<T>> {
        let alpha_val = alpha.unwrap_or(0.01); // Default negative slope
        let (rows, cols) = self.shape();
        let mut result = GraphMatrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                if let Some(value) = self.get(i, j) {
                    let x = value.to_f64();
                    let activated = if x > 0.0 { x } else { alpha_val * x };
                    let result_val = T::from_f64(activated).unwrap_or_else(|| T::zero());
                    let _ = result.set(i, j, result_val);
                }
            }
        }

        Ok(result)
    }

    /// ELU (Exponential Linear Unit) activation function
    /// Applies f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    pub fn elu(&self, alpha: Option<f64>) -> GraphResult<GraphMatrix<T>> {
        let alpha_val = alpha.unwrap_or(1.0); // Default alpha
        let (rows, cols) = self.shape();
        let mut result = GraphMatrix::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                if let Some(value) = self.get(i, j) {
                    let x = value.to_f64();
                    let activated = if x > 0.0 {
                        x
                    } else {
                        alpha_val * (x.exp() - 1.0)
                    };
                    let result_val = T::from_f64(activated).unwrap_or_else(|| T::zero());
                    let _ = result.set(i, j, result_val);
                }
            }
        }

        Ok(result)
    }

    /// Dropout operation for regularization
    /// Randomly sets elements to zero with probability p
    pub fn dropout(&self, p: f64, training: bool) -> GraphResult<GraphMatrix<T>> {
        if !training {
            // In evaluation mode, return original matrix scaled by (1 - p)
            let scale = 1.0 - p;
            let (rows, cols) = self.shape();
            let mut result = GraphMatrix::zeros(rows, cols);

            for i in 0..rows {
                for j in 0..cols {
                    if let Some(value) = self.get(i, j) {
                        let scaled_val = value.to_f64() * scale;
                        let result_val = T::from_f64(scaled_val).unwrap_or_else(|| T::zero());
                        let _ = result.set(i, j, result_val);
                    }
                }
            }

            return Ok(result);
        }

        if !(0.0..=1.0).contains(&p) {
            return Err(GraphError::InvalidInput(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        let (rows, cols) = self.shape();
        let mut result = GraphMatrix::zeros(rows, cols);
        let scale = 1.0 / (1.0 - p); // Scale factor to maintain expected value

        // Simple pseudo-random number generation for demonstration
        // In production, this should use a proper RNG
        let mut seed = (rows * cols) as u64;

        for i in 0..rows {
            for j in 0..cols {
                // Simple LCG for demonstration
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let rand_val = (seed as f64) / (u64::MAX as f64);

                if rand_val >= p {
                    // Keep the element, scaled up
                    if let Some(value) = self.get(i, j) {
                        let scaled_val = value.to_f64() * scale;
                        let result_val = T::from_f64(scaled_val).unwrap_or_else(|| T::zero());
                        let _ = result.set(i, j, result_val);
                    }
                }
                // Otherwise element stays zero (dropped out)
            }
        }

        Ok(result)
    }

    /// 2D Convolution operation (placeholder - integration incomplete)
    pub fn conv2d(
        &self,
        _kernel: &GraphMatrix<T>,
        _config: crate::storage::advanced_matrix::neural::convolution::ConvolutionConfig,
    ) -> GraphResult<GraphMatrix<T>> {
        // TODO: Complete Conv2D integration - API mismatch between Conv2D (expects ConvTensor) and GraphMatrix (UnifiedMatrix)
        // Conv2D::new requires (in_channels, out_channels, config) not (kernel, config)
        // Conv2D::forward expects ConvTensor<T> not UnifiedMatrix<T>
        Err(GraphError::InvalidInput(
            "Conv2D integration incomplete - API requires ConvTensor conversion".into(),
        ))
    }

    /// Enable automatic differentiation
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.computation_graph.is_none() {
            let mut graph = ComputationGraph::new();
            // Create a leaf node for this matrix
            let node_id = graph.create_leaf(self.storage.clone(), true);
            self.current_node_id = Some(node_id);
            self.computation_graph = Some(graph);
        } else if !requires_grad {
            self.computation_graph = None;
            self.current_node_id = None;
        }
        self
    }

    /// Compute gradients via backpropagation
    pub fn backward(&mut self) -> GraphResult<()> {
        if let Some(ref mut graph) = self.computation_graph {
            if let Some(node_id) = self.current_node_id {
                graph
                    .backward(node_id)
                    .map_err(|e| GraphError::InvalidInput(e.to_string()))
            } else {
                Err(GraphError::InvalidInput(
                    "No current node ID for gradient computation".into(),
                ))
            }
        } else {
            Err(GraphError::InvalidInput(
                "No computation graph available for gradient computation".into(),
            ))
        }
    }

    /// Pull back an explicit output cotangent through the saved computation tape.
    pub fn backward_with_gradient(&mut self, gradient: &Self) -> GraphResult<()> {
        let input = gradient
            .storage
            .to_dense()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let id = self
            .current_node_id
            .ok_or_else(|| GraphError::InvalidInput("No current gradient node".into()))?;
        let graph = self
            .computation_graph
            .as_mut()
            .ok_or_else(|| GraphError::InvalidInput("No computation graph".into()))?;
        graph
            .backward_with_gradient(id, input)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }

    /// Get gradient matrix
    pub fn grad(&self) -> Option<GraphMatrix<T>> {
        if let Some(ref graph) = self.computation_graph {
            if let Some(node_id) = self.current_node_id {
                if let Some(node_arc) = graph.get_node(node_id) {
                    let node = node_arc.lock().unwrap();
                    if let Some(ref gradient_matrix) = node.gradient {
                        // Convert UnifiedMatrix gradient to GraphMatrix
                        let mut gradient = Self::from_storage(gradient_matrix.clone());
                        gradient.column_names = self.column_names.clone();
                        gradient.row_labels = self.row_labels.clone();
                        return Some(gradient);
                    }
                }
            }
        }
        None
    }

    /// Zero out gradients in the computation graph
    pub fn zero_grad(&mut self) -> GraphResult<()> {
        if let Some(ref graph) = self.computation_graph {
            // Clear gradients for all nodes in the computation graph
            for (_, node_arc) in graph.nodes() {
                let mut node = node_arc.lock().unwrap();
                node.gradient = None;
            }
            Ok(())
        } else {
            Err(GraphError::InvalidInput(
                "No computation graph available for gradient clearing".into(),
            ))
        }
    }

    /// Check if gradients are enabled for this matrix
    pub fn requires_grad_enabled(&self) -> bool {
        self.requires_grad
    }

    /// Cast matrix to different numeric type
    pub fn cast<U: NumericType>(&self) -> GraphResult<GraphMatrix<U>> {
        self.cast_with_policy(crate::storage::array::numeric_arithmetic::ConversionPolicy::Exact)
    }

    /// Owned matrix cast with an explicit conversion policy, retaining coordinate identity.
    pub fn cast_with_policy<U: NumericType>(
        &self,
        policy: crate::storage::array::numeric_arithmetic::ConversionPolicy,
    ) -> GraphResult<GraphMatrix<U>> {
        use crate::storage::array::numeric_arithmetic::{convert, from_scalar};
        let (rows, cols) = self.shape();
        let mut result = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            let mut values = Vec::new();
            let mut indices = Vec::new();
            let mut pointers = vec![0];
            for row in 0..rows {
                for entry in sparse.row_pointers[row]..sparse.row_pointers[row + 1] {
                    let value = convert::<U>(from_scalar(sparse.values[entry]), policy)?;
                    if value != U::zero() {
                        values.push(value);
                        indices.push(sparse.col_indices[entry]);
                    }
                }
                pointers.push(values.len());
            }
            GraphMatrix::<U>::from_csr(rows, cols, values, indices, pointers)?
        } else {
            let values = NumArray::new(self.to_vec()?)
                .cast_with_policy::<U>(policy)?
                .into_vec();
            GraphMatrix::<U>::from_row_major_data(values, rows, cols, None)?
        };
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }

    /// Check if the matrix is sparse (has many zero elements)
    pub fn is_sparse(&self) -> bool {
        matches!(
            &self.storage.storage,
            crate::storage::advanced_matrix::unified_matrix::MatrixStorage::Sparse(_)
        )
    }

    /// Check if the matrix is square (same number of rows and columns)
    pub fn is_square(&self) -> bool {
        let (rows, cols) = self.shape();
        rows == cols
    }

    /// Check if the matrix contains numeric data (always true for GraphMatrix<T: NumericType>)
    pub fn is_numeric(&self) -> bool {
        true
    }

    /// Check if the matrix is symmetric (A[i,j] == A[j,i] for all i,j)
    pub fn is_symmetric(&self) -> bool
    where
        f64: From<T>,
    {
        self.is_symmetric_internal()
    }

    /// Get the data type of the matrix elements
    pub fn dtype(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    /// Create an identity matrix of given size
    pub fn identity(size: usize) -> GraphResult<Self> {
        UnifiedMatrix::identity(size)
            .map(Self::from_storage)
            .map_err(|error| {
                GraphError::InvalidInput(format!("Identity matrix creation failed: {error:?}"))
            })
    }

    /// Get element at specific row and column
    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        self.storage.get(row, col).ok()
    }

    /// Set element at specific row and column  
    pub fn set(&mut self, row: usize, col: usize, value: T) -> GraphResult<()> {
        self.storage
            .set(row, col, value)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }

    /// Get element at specific row and column with error details
    pub fn get_checked(&self, row: usize, col: usize) -> GraphResult<T> {
        self.storage
            .get(row, col)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }

    /// Set column names with result return (for FFI map_err chaining)
    pub fn set_column_names_result(&mut self, names: Vec<String>) -> GraphResult<()> {
        self.set_column_names(names);
        Ok(())
    }

    /// Get a full row as a vector
    pub fn get_row(&self, row: usize) -> Option<Vec<T>> {
        let (rows, cols) = self.shape();
        if row >= rows {
            return None;
        }

        let mut row_data = Vec::with_capacity(cols);
        for col in 0..cols {
            if let Some(value) = self.get(row, col) {
                row_data.push(value);
            } else {
                return None; // If any cell is invalid, return None
            }
        }
        Some(row_data)
    }

    /// Get a full column as a vector
    pub fn get_column(&self, col: usize) -> Option<Vec<T>> {
        let (rows, cols) = self.shape();
        if col >= cols {
            return None;
        }

        let mut col_data = Vec::with_capacity(rows);
        for row in 0..rows {
            if let Some(value) = self.get(row, col) {
                col_data.push(value);
            } else {
                return None; // If any cell is invalid, return None
            }
        }
        Some(col_data)
    }

    /// Get a column by name
    pub fn get_column_by_name(&self, name: &str) -> Option<Vec<T>> {
        if let Some(col_idx) = self.column_names.iter().position(|n| n == name) {
            self.get_column(col_idx)
        } else {
            None // Column not found
        }
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        if self.is_sparse() {
            if self.row_labels.as_ref().is_some_and(|v| v.len() != rows) {
                return Err(GraphError::InvalidInput(
                    "Row label count does not match shape".into(),
                ));
            }
            let mut result = Self::from_storage(
                self.storage
                    .transpose()
                    .map_err(|e| GraphError::InvalidInput(e.to_string()))?,
            );
            result.row_labels = Some(self.column_names.clone());
            if let Some(labels) = &self.row_labels {
                result.column_names = labels.clone();
            }
            return Ok(result);
        }
        let mut result = Self::arithmetic_zeros(cols, rows)?;
        for row in 0..rows {
            for col in 0..cols {
                result.set(col, row, self.get_checked(row, col)?)?;
            }
        }
        result.row_labels = Some(self.column_names.clone());
        if let Some(labels) = &self.row_labels {
            if labels.len() != rows {
                return Err(GraphError::InvalidInput(
                    "Row label count does not match shape".into(),
                ));
            }
            result.column_names = labels.clone();
        }
        Ok(result)
    }

    /// Explicit owned conversion to canonical CSR storage.
    pub fn to_csr(&self) -> GraphResult<Self> {
        self.convert_storage(true)
    }
    /// Explicit owned dense materialization.
    pub fn to_dense(&self) -> GraphResult<Self> {
        self.convert_storage(false)
    }
    fn convert_storage(&self, sparse: bool) -> GraphResult<Self> {
        let storage = if sparse {
            self.storage.to_csr()
        } else {
            self.storage.to_dense()
        }
        .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut result = Self::from_storage(storage);
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }

    /// Matrix multiplication
    pub fn multiply(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        if self.shape().1 != other.shape().0 {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
                self.shape().0,
                self.shape().1,
                other.shape().0,
                other.shape().1
            )));
        }

        let storage = self
            .storage
            .matmul(&other.storage)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut result = Self::from_storage(storage);
        result.row_labels = self.row_labels.clone();
        result.column_names = other.column_names.clone();
        Ok(result)
    }

    /// Positional matrix subtraction with checked shape and cell access.
    pub fn subtract(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        self.zip_arithmetic(other, T::checked_sub)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn elementwise_multiply(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        self.zip_arithmetic(other, T::checked_mul)
    }

    /// Matrix power operation (repeated multiplication)
    pub fn power(&self, n: u32) -> GraphResult<GraphMatrix<T>> {
        if !self.is_square() {
            return Err(GraphError::InvalidInput(
                "Matrix must be square for power operation".into(),
            ));
        }

        // Handle computation graph if gradients are required
        if self.requires_grad {
            if let Some(ref graph) = self.computation_graph {
                if let Some(input_node_id) = self.current_node_id {
                    // Create power operation node in computation graph
                    let mut graph_mut = graph.clone(); // We need mutable access
                    let result_shape = (self.shape().0, self.shape().1);

                    // Create the operation node
                    let output_node_id = graph_mut
                        .create_operation(
                            crate::storage::advanced_matrix::neural::autodiff::Operation::Power {
                                exponent: n,
                            },
                            vec![input_node_id],
                            crate::storage::advanced_matrix::unified_matrix::Shape::new(
                                result_shape.0,
                                result_shape.1,
                            ),
                            true, // requires_grad
                        )
                        .map_err(|e| GraphError::InvalidInput(e.to_string()))?;

                    // Compute the actual result
                    let result_storage = if n == 0 {
                        let (size, _) = self.shape();
                        Self::identity(size)?.storage
                    } else {
                        let mut result_storage = self.storage.clone();
                        for _ in 1..n {
                            result_storage = result_storage
                                .matmul(&self.storage)
                                .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
                        }
                        result_storage
                    };

                    // Set the computed value in the graph node
                    if let Some(node_arc) = graph_mut.get_node(output_node_id) {
                        let mut node = node_arc.lock().unwrap();
                        node.value = Some(result_storage.clone());
                    }

                    // Create result matrix with computation graph
                    let mut result = Self::from_storage(result_storage);
                    result.requires_grad = true;
                    result.computation_graph = Some(graph_mut);
                    result.current_node_id = Some(output_node_id);

                    return Ok(result);
                }
            }
        }

        // Fallback: regular computation without gradients
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

    /// Mean per logical axis; empty reductions are zero and outputs retain T.
    pub fn mean_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, super::reductions::typed_mean)
    }
    /// Sample standard deviation, independently scaled before output conversion.
    pub fn std_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, |values, count| {
            super::reductions::statistic_value(super::reductions::extended_std(values, count, true))
        })
    }
    /// Sample variance; nonfinite observations propagate NaN.
    pub fn var_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, |values, count| {
            super::reductions::statistic_value(super::reductions::extended_variance(
                values, count, true,
            ))
        })
    }
    fn reduce_axis<R>(
        &self,
        axis: Axis,
        operation: impl Fn(&[T], usize) -> GraphResult<R>,
    ) -> GraphResult<Vec<R>> {
        self.storage
            .reduce_axis(matches!(axis, Axis::Columns), |values, count| {
                operation(values, count).map_err(|e| {
                    crate::storage::advanced_matrix::unified_matrix::MatrixError::InvalidInput(
                        e.to_string(),
                    )
                })
            })
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }
    fn logical_values(&self) -> GraphResult<Vec<T>> {
        self.storage
            .to_vec()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))
    }

    /// Minimum per axis; empty reductions retain legacy zero, NaNs propagate.
    pub fn min_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, |values, count| {
            super::reductions::extended_extremum(values, count, true)
        })
    }
    /// Maximum per axis; empty reductions retain legacy zero, NaNs propagate.
    pub fn max_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, |values, count| {
            super::reductions::extended_extremum(values, count, false)
        })
    }

    /// Materialize the matrix (convert lazy operations to concrete data)
    pub fn materialize(&self) -> GraphResult<GraphMatrix<T>> {
        let storage = if self.is_sparse() {
            self.storage.to_csr()
        } else {
            self.storage.to_dense()
        }
        .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut result = Self::from_storage(storage);
        result.column_names = self.column_names.clone();
        result.row_labels = self.row_labels.clone();
        Ok(result)
    }

    /// Get a preview of the matrix data (first few rows/columns)
    pub fn preview(&self, row_limit: usize, col_limit: usize) -> (Vec<Vec<T>>, Vec<String>) {
        let (rows, cols) = self.shape();
        let actual_rows = rows.min(row_limit);
        let actual_cols = cols.min(col_limit);

        let mut preview_data = Vec::new();
        for i in 0..actual_rows {
            let mut row = Vec::new();
            for j in 0..actual_cols {
                let value = self.get(i, j).unwrap_or_else(|| T::zero());
                row.push(value);
            }
            preview_data.push(row);
        }

        let col_names = if self.column_names.len() >= actual_cols {
            self.column_names[0..actual_cols].to_vec()
        } else {
            (0..actual_cols).map(|i| format!("col_{}", i)).collect()
        };

        (preview_data, col_names)
    }

    /// Get summary information about the matrix
    pub fn summary_info(&self) -> String {
        let (rows, cols) = self.shape();
        format!(
            "GraphMatrix<{}>: {}x{} matrix",
            std::any::type_name::<T>(),
            rows,
            cols
        )
    }

    /// Explicit owned dense representation.
    pub fn dense(&self) -> GraphResult<GraphMatrix<T>> {
        self.to_dense()
    }

    /// Create GraphMatrix from arrays (legacy compatibility method)
    pub fn from_arrays(arrays: Vec<crate::storage::array::NumArray<T>>) -> GraphResult<Self> {
        if arrays.is_empty() {
            return Ok(Self::zeros(0, 0));
        }

        let cols = arrays.len();
        let rows = arrays[0].len();

        // Ensure all arrays have the same length
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != rows {
                return Err(GraphError::InvalidInput(format!(
                    "Array {} has length {} but expected {}",
                    i,
                    array.len(),
                    rows
                )));
            }
        }

        // Create the matrix and populate it column by column
        let mut matrix = Self::zeros(rows, cols);
        for (col_idx, array) in arrays.iter().enumerate() {
            for (row_idx, value) in array.iter().enumerate() {
                matrix.set(row_idx, col_idx, *value)?;
            }
        }

        Ok(matrix)
    }

    /// Checked typed sum per axis, with exact floating accumulation.
    pub fn sum_axis(&self, axis: Axis) -> GraphResult<Vec<T>> {
        self.reduce_axis(axis, |values, _| super::reductions::sum(values))
    }

    /// Reshape matrix to new dimensions while preserving total element count
    ///
    /// # Arguments
    /// * `new_rows` - New number of rows
    /// * `new_cols` - New number of columns
    ///
    /// # Returns
    /// New GraphMatrix with reshaped dimensions
    ///
    /// # Errors
    /// Returns error if new_rows * new_cols != current total elements
    pub fn reshape(&self, new_rows: usize, new_cols: usize) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| GraphError::InvalidInput("Input shape overflow".into()))?;
        if new_rows.checked_mul(new_cols) != Some(count) {
            return Err(GraphError::InvalidInput(
                "Reshape element counts must match without overflow".into(),
            ));
        }
        if self.is_sparse() {
            if let Some(storage) = self
                .storage
                .sparse_reshape(new_rows, new_cols)
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?
            {
                let mut result = Self::from_storage(storage);
                if (rows, cols) == (new_rows, new_cols) {
                    result.row_labels = self.row_labels.clone();
                    result.column_names = self.column_names.clone();
                }
                return Ok(result);
            }
        }
        let mut result = Self::arithmetic_zeros(new_rows, new_cols)?;
        for row in 0..rows {
            for col in 0..cols {
                let index = row * cols + col;
                result.set(
                    index / new_cols,
                    index % new_cols,
                    self.get_checked(row, col)?,
                )?;
            }
        }
        if (rows, cols) == (new_rows, new_cols) {
            result.column_names = self.column_names.clone();
            result.row_labels = self.row_labels.clone();
        }
        Ok(result)
    }

    /// Checked typed global sum; callers must handle GraphResult.
    pub fn sum(&self) -> GraphResult<T> {
        if let crate::storage::advanced_matrix::unified_matrix::MatrixStorage::Sparse(s) =
            &self.storage.storage
        {
            if s.shape.as_tuple() != self.shape() {
                return Err(GraphError::InvalidInput(
                    "CSR shape metadata mismatch".into(),
                ));
            }
            s.validate()
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
            return super::reductions::sum(&s.values);
        }
        super::reductions::sum(&self.logical_values()?)
    }
    /// Global f64 mean, avoiding intermediate typed-sum overflow; empty=0.
    pub fn mean(&self) -> GraphResult<f64> {
        let (rows, cols) = self.shape();
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| GraphError::InvalidInput("Matrix element count overflow".into()))?;
        let values = self
            .storage
            .reduction_values()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        Ok(super::reductions::extended_mean(&values, count))
    }
    /// Global minimum; empty=None and any NaN propagates.
    pub fn min(&self) -> GraphResult<Option<T>> {
        if self.is_sparse() {
            let (rows, cols) = self.shape();
            if rows == 0 || cols == 0 {
                return Ok(None);
            }
            return self
                .storage
                .min()
                .map(Some)
                .map_err(|e| GraphError::InvalidInput(e.to_string()));
        }
        super::reductions::extremum(&self.logical_values()?, true)
    }
    /// Global maximum; empty=None and any NaN propagates.
    pub fn max(&self) -> GraphResult<Option<T>> {
        if self.is_sparse() {
            let (rows, cols) = self.shape();
            if rows == 0 || cols == 0 {
                return Ok(None);
            }
            return self
                .storage
                .max()
                .map(Some)
                .map_err(|e| GraphError::InvalidInput(e.to_string()));
        }
        super::reductions::extremum(&self.logical_values()?, false)
    }

    /// Calculate the trace (sum of diagonal elements) of the matrix
    /// Only valid for square matrices
    pub fn trace(&self) -> GraphResult<T>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Trace is only defined for square matrices, got {}x{}",
                rows, cols
            )));
        }

        let diagonal = (0..rows)
            .map(|i| self.get_checked(i, i))
            .collect::<GraphResult<Vec<_>>>()?;
        super::reductions::sum(&diagonal)
    }

    /// Calculate the Frobenius norm (Euclidean norm) of the matrix
    /// ||A||_F = sqrt(sum(a_ij^2)) for all i,j
    pub fn norm(&self) -> GraphResult<f64> {
        let values = self
            .storage
            .reduction_values()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        Ok(crate::storage::array::numeric_arithmetic::norm(&values))
    }

    /// Entrywise L1 norm: sum of absolute values (not an induced matrix norm).
    pub fn norm_l1(&self) -> GraphResult<f64> {
        let values = self
            .storage
            .reduction_values()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut total = super::dot_product::Dot::new();
        for value in values.iter() {
            total.add(value.to_f64().abs(), 1.0);
        }
        Ok(total.total())
    }

    /// Entrywise infinity norm: maximum absolute value; NaNs propagate.
    pub fn norm_inf(&self) -> GraphResult<f64> {
        let values = self
            .storage
            .reduction_values()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        if values.iter().any(|x| x.to_f64().is_nan()) {
            return Ok(f64::NAN);
        }
        Ok(values
            .iter()
            .map(|x| x.to_f64().abs())
            .fold(0.0_f64, f64::max))
    }
}

// Graph-specific matrix operations
impl<T: NumericType> GraphMatrix<T> {
    /// Convert adjacency matrix to Laplacian matrix
    pub fn to_laplacian(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Laplacian matrix requires square adjacency matrix, got {}x{}",
                rows, cols
            )));
        }

        // Calculate degree matrix D
        let degree_matrix = self.to_degree_matrix()?;

        // Calculate L = D - A
        let laplacian = degree_matrix.subtract(self)?;

        Ok(laplacian)
    }

    /// Convert adjacency matrix to normalized Laplacian matrix
    /// Enhanced version: (D^eps @ A @ D^eps)^k
    ///
    /// # Arguments
    /// * `eps` - Exponent for degree matrix (default: -0.5 for standard normalization)
    /// * `k` - Power to raise the result to (default: 1)
    ///
    /// # Standard Usage
    /// - `eps = -0.5, k = 1`: Standard normalized Laplacian L = D^(-1/2) @ A @ D^(-1/2)
    /// - `eps = -1.0, k = 1`: Random walk Laplacian L = D^(-1) @ A
    /// - `eps = 0.5, k = 1`: Symmetric scaling L = D^(1/2) @ A @ D^(1/2)
    pub fn to_normalized_laplacian(&self, eps: f64, k: u32) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Normalized Laplacian requires square matrix, got {}x{}",
                rows, cols
            )));
        }

        // Step 1: Calculate degree matrix D
        let degree_matrix = self.to_degree_matrix()?;

        // Step 2: Calculate D^eps
        let mut d_eps = GraphMatrix::zeros(rows, cols);
        for i in 0..rows {
            if let Some(degree) = degree_matrix.get(i, i) {
                let degree_f64 = degree.to_f64();
                if degree_f64 > 0.0 {
                    let powered_degree = degree_f64.powf(eps);
                    let powered_t = T::from_f64(powered_degree).unwrap_or_else(|| T::zero());
                    let _ = d_eps.set(i, i, powered_t);
                }
            }
        }

        // Step 3: Calculate D^eps @ A @ D^eps
        let temp_result = d_eps.multiply(self)?; // D^eps @ A
        let mut normalized_result = temp_result.multiply(&d_eps)?; // @ D^eps

        // Step 4: Raise to power k if k != 1
        if k > 1 {
            for _ in 1..k {
                normalized_result = normalized_result.multiply(&normalized_result)?;
            }
        }

        Ok(normalized_result)
    }

    /// Convert adjacency matrix to standard normalized Laplacian
    /// Convenience method for standard L = D^(-1/2) @ A @ D^(-1/2)
    pub fn to_normalized_laplacian_standard(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        self.to_normalized_laplacian(-0.5, 1)
    }

    /// Check if this matrix represents a valid adjacency matrix
    pub fn is_adjacency_matrix(&self) -> bool {
        // Check if matrix is square
        if !self.is_square() {
            return false;
        }

        // Check for non-negative values
        let (rows, cols) = self.shape();
        for i in 0..rows {
            for j in 0..cols {
                if let Some(val) = self.get(i, j) {
                    if val.to_f64() < 0.0 {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    /// Get degree matrix from adjacency matrix
    pub fn to_degree_matrix(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Degree matrix requires square adjacency matrix, got {}x{}",
                rows, cols
            )));
        }

        let mut degree_matrix = GraphMatrix::zeros(rows, cols);

        // Calculate degree for each node (sum of row in adjacency matrix)
        for i in 0..rows {
            let mut degree_sum = T::zero();
            for j in 0..cols {
                if let Some(value) = self.get(i, j) {
                    degree_sum = degree_sum + value;
                }
            }
            // Set degree on diagonal
            let _ = degree_matrix.set(i, i, degree_sum);
        }

        Ok(degree_matrix)
    }

    // === ADVANCED RESHAPING OPERATIONS ===

    /// Concatenate matrices along specified axis
    ///
    /// Args:
    ///     other: Matrix to concatenate with
    ///     axis: 0 for row-wise (vertical), 1 for column-wise (horizontal)
    pub fn concatenate(&self, other: &GraphMatrix<T>, axis: usize) -> GraphResult<GraphMatrix<T>> {
        for source in [self, other] {
            if source.column_names.len() != source.shape().1
                || source
                    .row_labels
                    .as_ref()
                    .is_some_and(|labels| labels.len() != source.shape().0)
            {
                return Err(GraphError::InvalidInput(
                    "Concatenation label count does not match shape".into(),
                ));
            }
        }
        let storage = self
            .storage
            .concatenate(&other.storage, axis)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut result = Self::from_storage(storage);
        result.column_names = self.column_names.clone();
        result.row_labels = self.row_labels.clone();
        if axis == 0 {
            result.row_labels = self
                .row_labels
                .as_ref()
                .zip(other.row_labels.as_ref())
                .map(|(a, b)| a.iter().chain(b).cloned().collect());
        } else {
            result
                .column_names
                .extend(other.column_names.iter().cloned());
        }
        Ok(result)
    }

    /// Stack matrices along a new axis (creates a new dimension)
    ///
    /// Args:
    ///     other: Matrix to stack with
    ///     axis: 0 for depth stacking (creates 3D-like structure as flattened 2D)
    pub fn stack(&self, other: &GraphMatrix<T>, axis: usize) -> GraphResult<GraphMatrix<T>> {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        if self_rows != other_rows || self_cols != other_cols {
            return Err(GraphError::InvalidInput(format!(
                "Matrix shapes must match for stacking: {}x{} vs {}x{}",
                self_rows, self_cols, other_rows, other_cols
            )));
        }

        match axis {
            0 => {
                // Stack along depth (treat as row concatenation for 2D representation)
                self.concatenate(other, 0)
            }
            1 => {
                // Stack along new column dimension
                self.concatenate(other, 1)
            }
            _ => Err(GraphError::InvalidInput(format!(
                "Invalid axis for stacking: {}. Must be 0 or 1",
                axis
            ))),
        }
    }

    /// Split matrix along specified axis
    ///
    /// Args:
    ///     split_points: Indices where to split (exclusive)
    ///     axis: 0 for row-wise split, 1 for column-wise split
    pub fn split(&self, split_points: &[usize], axis: usize) -> GraphResult<Vec<GraphMatrix<T>>> {
        let (rows, cols) = self.shape();
        let limit = match axis {
            0 => rows,
            1 => cols,
            _ => {
                return Err(GraphError::InvalidInput(
                    "Split axis must be 0 (rows) or 1 (columns)".into(),
                ))
            }
        };
        let mut previous = 0;
        for &point in split_points {
            if point <= previous || point > limit {
                return Err(GraphError::InvalidInput(
                    "Split points must increase from zero and stay within the selected dimension"
                        .into(),
                ));
            }
            previous = point;
        }
        let mut results = Vec::new();
        previous = 0;
        for point in split_points.iter().copied().chain(std::iter::once(limit)) {
            if point > previous {
                let selected_rows: Vec<_> = if axis == 0 {
                    (previous..point).collect()
                } else {
                    (0..rows).collect()
                };
                let selected_cols: Vec<_> = if axis == 1 {
                    (previous..point).collect()
                } else {
                    (0..cols).collect()
                };
                results.push(self.select_indices(&selected_rows, &selected_cols)?);
            }
            previous = point;
        }
        Ok(results)
    }

    // === BATCH 5: ADVANCED LINEAR ALGEBRA OPERATIONS ===

    /// Floating determinant; empty order is one, singular input is zero.
    pub fn determinant(&self) -> GraphResult<f64> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Determinant requires square input".into(),
            ));
        }
        if rows > 2 {
            if let Some(sparse) = self
                .storage
                .checked_sparse()
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?
            {
                let (sign, log) = super::sparse_factorization::slogdet(sparse)?;
                return Ok(if sign == 0. { 0. } else { sign * log.exp() });
            }
        }
        let values = self.decomposition_input()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GraphError::InvalidInput(
                "Determinant requires finite input".into(),
            ));
        }
        if rows == 0 {
            return Ok(1.);
        }
        if rows == 1 {
            return Ok(values[0]);
        }
        if rows == 2 {
            let mut dot = super::dot_product::Dot::new();
            dot.add(values[0], values[3]);
            dot.add(-values[1], values[2]);
            return Ok(dot.total());
        }
        let factor = super::factorization::lu(values, rows)?;
        let (sign, log) = super::factorization::slogdet(&factor, rows);
        Ok(if sign == 0. { 0. } else { sign * log.exp() })
    }
    /// Sign and log absolute determinant, avoiding final product range loss.
    pub fn slogdet(&self) -> GraphResult<(f64, f64)> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Log determinant requires square input".into(),
            ));
        }
        if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            return super::sparse_factorization::slogdet(sparse);
        }
        let factor = super::factorization::lu(self.decomposition_input()?, rows)?;
        Ok(super::factorization::slogdet(&factor, rows))
    }

    /// Floating dense inverse using a direct solve against identity.
    pub fn inverse(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Matrix inverse requires square input".into(),
            ));
        }
        let mut identity = Self::arithmetic_zeros(rows, rows)?;
        for i in 0..rows {
            identity.set(i, i, T::one())?;
        }
        identity.column_names = self
            .row_labels
            .clone()
            .unwrap_or_else(|| identity.column_names.clone());
        self.solve(&identity)
    }

    /// Floating dense direct solve with row equilibration and partial pivoting.
    /// Reject singular/nonfinite/unrepresentable input rather than guessing output.
    pub fn solve(&self, b: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    {
        let (rows, cols) = self.shape();
        let (b_rows, b_cols) = b.shape();
        if rows != cols || b_rows != rows {
            return Err(GraphError::InvalidInput(
                "Linear solve requires square coefficients and matching RHS rows".into(),
            ));
        }
        let values = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            super::sparse_factorization::solve(sparse, b.decomposition_input()?, b_cols)?
        } else {
            super::linear_system::solve(
                self.decomposition_input()?,
                b.decomposition_input()?,
                rows,
                b_cols,
            )?
        };
        let mut result = Self::arithmetic_zeros(rows, b_cols)?;
        for row in 0..rows {
            for col in 0..b_cols {
                let value = values[row * b_cols + col];
                let converted = T::from_f64(value)
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        GraphError::InvalidInput(
                            "Linear solve output exceeds its element dtype".into(),
                        )
                    })?;
                result.set(row, col, converted)?;
            }
        }
        result.row_labels = Some(self.column_names.clone());
        result.column_names = b.column_names.clone();
        Ok(result)
    }

    /// Solve with residual reporting and at most 16 optional refinement attempts.
    /// Accept a correction only when the returned-dtype backward error decreases.
    pub fn solve_refined(
        &self,
        b: &GraphMatrix<T>,
        max_refinements: usize,
    ) -> GraphResult<SolveReport<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    {
        if max_refinements > 16 {
            return Err(GraphError::InvalidInput(
                "At most 16 refinement attempts are supported".into(),
            ));
        }
        let mut solution = self.solve(b)?;
        let (n, rhs) = b.shape();
        let a = self.decomposition_input()?;
        let b = b.decomposition_input()?;
        let mut x = solution.decomposition_input()?;
        let mut residual = super::linear_system::residual(&a, &b, &x, n, rhs)?;
        let mut error = super::linear_system::backward_error(&a, &b, &x, &residual, n, rhs);
        let mut accepted = 0;
        for _ in 0..max_refinements {
            if error == 0. {
                break;
            }
            let correction = super::linear_system::solve(a.clone(), residual, n, rhs)?;
            let candidate: Vec<_> = x
                .iter()
                .zip(correction)
                .map(|(v, delta)| v + delta)
                .collect();
            let mut candidate = Self::factor_matrix(candidate, n, rhs)?;
            let candidate_x = candidate.decomposition_input()?;
            let candidate_residual = super::linear_system::residual(&a, &b, &candidate_x, n, rhs)?;
            let candidate_error = super::linear_system::backward_error(
                &a,
                &b,
                &candidate_x,
                &candidate_residual,
                n,
                rhs,
            );
            if candidate_error >= error {
                break;
            }
            candidate.row_labels = solution.row_labels.clone();
            candidate.column_names = solution.column_names.clone();
            solution = candidate;
            x = candidate_x;
            residual = candidate_residual;
            error = candidate_error;
            accepted += 1;
        }
        Ok(SolveReport {
            solution,
            backward_error: error,
            accepted_refinements: accepted,
        })
    }

    /// Minimum-norm real least squares using a relative truncated SVD.
    /// Supports rectangular/rank-deficient input and multiple RHS columns.
    pub fn least_squares(&self, b: &GraphMatrix<T>, relative: Option<f64>) -> GraphResult<Self> {
        let (rows, cols) = self.shape();
        let (b_rows, rhs) = b.shape();
        if b_rows != rows {
            return Err(GraphError::InvalidInput(
                "Least squares requires matching RHS rows".into(),
            ));
        }
        let relative = relative.unwrap_or_else(|| self.default_rank_tolerance());
        if !relative.is_finite() || relative < 0. {
            return Err(GraphError::InvalidInput(
                "Least-squares tolerance must be finite and nonnegative".into(),
            ));
        }
        let values = super::least_squares::solve(
            self.decomposition_input()?,
            b.decomposition_input()?,
            rows,
            cols,
            rhs,
            relative,
        )?;
        let mut result = Self::factor_matrix(values, cols, rhs)?;
        result.row_labels = Some(self.column_names.clone());
        result.column_names = b.column_names.clone();
        Ok(result)
    }

    /// SVD decomposition: A = U * Σ * V^T
    /// Returns (U, singular_values, V_transpose)
    pub fn svd(&self) -> GraphResult<(GraphMatrix<T>, Vec<f64>, GraphMatrix<T>)>
    where
        T: Clone
            + Copy
            + Default
            + std::fmt::Debug
            + PartialEq
            + PartialOrd
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        let (rows, cols) = self.shape();
        if self.requires_grad && rows != cols {
            return Err(GraphError::InvalidInput("Tracked rectangular SVD requires svd_tracked() thin outputs; a full null basis has no supported derivative".into()));
        }
        let result = super::decomposition::svd(self.decomposition_input()?, rows, cols)?;
        Ok((
            self.track_factor(
                Self::decomposition_matrix(result.u, rows, rows)?,
                Operation::FactorOutput {
                    part: FactorOutput::SvdU,
                },
            )?,
            result.singular,
            self.track_factor(
                Self::decomposition_matrix(result.vt, cols, cols)?,
                Operation::FactorOutput {
                    part: FactorOutput::SvdVt,
                },
            )?,
        ))
    }

    /// Economy Householder QR: Q is m x min(m,n), R is min(m,n) x n.
    /// Zero/rank-deficient inputs retain an orthonormal Q; finite floating input only.
    pub fn qr_decomposition(&self) -> GraphResult<(GraphMatrix<T>, GraphMatrix<T>)> {
        let (rows, cols) = self.shape();
        if self
            .row_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "QR row labels do not match shape".into(),
            ));
        }
        let k = rows.min(cols);
        let (q, r) = super::factorization::qr(self.decomposition_input()?, rows, cols)?;
        let mut q = Self::factor_matrix(q, rows, k)?;
        let mut r = Self::factor_matrix(r, k, cols)?;
        let basis: Vec<_> = (0..k).map(|i| format!("qr_{i}")).collect();
        q.row_labels = self.row_labels.clone();
        q.column_names = basis.clone();
        r.row_labels = Some(basis);
        r.column_names = self.column_names.clone();
        Ok((
            self.track_factor(
                q,
                Operation::FactorOutput {
                    part: FactorOutput::QrQ,
                },
            )?,
            self.track_factor(
                r,
                Operation::FactorOutput {
                    part: FactorOutput::QrR,
                },
            )?,
        ))
    }
    fn factor_matrix(values: Vec<f64>, rows: usize, cols: usize) -> GraphResult<Self> {
        let values = values
            .into_iter()
            .map(|value| {
                if !value.is_finite() {
                    return Err(GraphError::InvalidInput(
                        "Nonfinite factorization output".into(),
                    ));
                }
                T::from_f64(value).filter(|v| v.is_finite()).ok_or_else(|| {
                    GraphError::InvalidInput("Factorization output exceeds dtype range".into())
                })
            })
            .collect::<GraphResult<Vec<_>>>()?;
        Self::from_row_major_data(values, rows, cols, None)
    }

    fn sparse_factor_matrix(
        factor: crate::storage::advanced_matrix::unified_matrix::SparseMatrix<f64>,
    ) -> GraphResult<Self> {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        let mut pointers = vec![0];
        for row in 0..factor.shape.rows {
            for entry in factor.row_pointers[row]..factor.row_pointers[row + 1] {
                let value = T::from_f64(factor.values[entry])
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        GraphError::InvalidInput("Sparse factor output exceeds dtype range".into())
                    })?;
                if value != T::zero() {
                    values.push(value);
                    indices.push(factor.col_indices[entry]);
                }
            }
            pointers.push(values.len());
        }
        Self::from_csr(
            factor.shape.rows,
            factor.shape.cols,
            values,
            indices,
            pointers,
        )
    }

    /// Floating partial-pivot factorization PA=LU; singular pivots remain zero.
    pub fn lu_decomposition(
        &self,
    ) -> GraphResult<(GraphMatrix<T>, GraphMatrix<T>, GraphMatrix<T>)> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput("LU requires square input".into()));
        }
        if self
            .row_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "LU row labels do not match shape".into(),
            ));
        }
        let (permutation, mut p, mut lower, mut upper) = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            let factor = super::sparse_factorization::lu(sparse)?;
            let p = Self::from_csr(
                rows,
                rows,
                vec![T::one(); rows],
                factor.permutation.clone(),
                (0..=rows).collect(),
            )?;
            (
                factor.permutation,
                p,
                Self::sparse_factor_matrix(factor.lower)?,
                Self::sparse_factor_matrix(factor.upper)?,
            )
        } else {
            let factor = super::factorization::lu(self.decomposition_input()?, rows)?;
            let mut p = Self::arithmetic_zeros(rows, rows)?;
            for (row, &col) in factor.permutation.iter().enumerate() {
                p.set(row, col, T::one())?;
            }
            (
                factor.permutation,
                p,
                Self::factor_matrix(factor.lower, rows, rows)?,
                Self::factor_matrix(factor.upper, rows, rows)?,
            )
        };
        let permuted = self.row_labels.as_ref().map(|labels| {
            permutation
                .iter()
                .map(|&i| labels[i].clone())
                .collect::<Vec<_>>()
        });
        p.row_labels = permuted.clone();
        if let Some(labels) = &self.row_labels {
            p.column_names = labels.clone();
        }
        lower.row_labels = permuted;
        let basis: Vec<_> = (0..rows).map(|i| format!("lu_{i}")).collect();
        lower.column_names = basis.clone();
        upper.row_labels = Some(basis);
        upper.column_names = self.column_names.clone();
        Ok((
            p,
            self.track_factor(
                lower,
                Operation::FactorOutput {
                    part: FactorOutput::LuL,
                },
            )?,
            self.track_factor(
                upper,
                Operation::FactorOutput {
                    part: FactorOutput::LuU,
                },
            )?,
        ))
    }

    /// Checked real floating Cholesky factor A=L*L^T.
    pub fn cholesky_decomposition(&self) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Cholesky requires square input".into(),
            ));
        }
        if self
            .row_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "Cholesky row labels do not match shape".into(),
            ));
        }
        let mut lower = if let Some(sparse) = self
            .storage
            .checked_sparse()
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?
        {
            Self::sparse_factor_matrix(super::sparse_factorization::cholesky(sparse)?)?
        } else {
            Self::factor_matrix(
                super::factorization::cholesky(self.decomposition_input()?, rows)?,
                rows,
                rows,
            )?
        };
        lower.row_labels = self.row_labels.clone();
        lower.column_names = (0..rows).map(|i| format!("chol_{i}")).collect();
        self.track_cholesky(lower)
    }

    // Transfer owned storage across native numerical/tape layers without exposing it publicly.
    pub(crate) fn into_storage(self) -> UnifiedMatrix<T> {
        self.storage
    }
    fn track_cholesky(&self, result: Self) -> GraphResult<Self> {
        self.track_factor(result, Operation::Cholesky)
    }
    fn track_factor(&self, mut result: Self, operation: Operation) -> GraphResult<Self> {
        if !self.requires_grad {
            return Ok(result);
        }
        let invalid = |message: &str| GraphError::InvalidInput(message.into());
        let mut graph = self
            .computation_graph
            .clone()
            .ok_or_else(|| invalid("Missing Factor input tape"))?;
        let input_id = self
            .current_node_id
            .ok_or_else(|| invalid("Missing Factor input node"))?;
        let input = graph
            .get_node(input_id)
            .ok_or_else(|| invalid("Missing Factor input node"))?;
        {
            let input = input
                .lock()
                .map_err(|_| invalid("Factor input tape lock poisoned"))?;
            let saved = input
                .value
                .as_ref()
                .ok_or_else(|| invalid("Missing Factor input snapshot"))?;
            let saved = Self::from_storage(saved.clone());
            let same_values = if let (Some(a), Some(b)) = (
                saved
                    .storage
                    .checked_sparse()
                    .map_err(|e| invalid(&e.to_string()))?,
                self.storage
                    .checked_sparse()
                    .map_err(|e| invalid(&e.to_string()))?,
            ) {
                a.shape == b.shape
                    && a.values == b.values
                    && a.col_indices == b.col_indices
                    && a.row_pointers == b.row_pointers
            } else {
                saved.decomposition_input()? == self.decomposition_input()?
            };
            if saved.shape() != self.shape() || !same_values {
                return Err(invalid("Factor input changed since gradient binding; rebind with requires_grad(false).requires_grad(true)"));
            }
        }
        let (rows, cols) = result.shape();
        let id = graph
            .create_operation(
                operation,
                vec![input_id],
                crate::storage::advanced_matrix::unified_matrix::Shape::new(rows, cols),
                true,
            )
            .map_err(|e| invalid(&e.to_string()))?;
        let output = graph
            .get_node(id)
            .ok_or_else(|| invalid("Missing Factor output node"))?;
        output
            .lock()
            .map_err(|_| invalid("Factor output tape lock poisoned"))?
            .value = Some(result.storage.clone());
        result.requires_grad = true;
        result.current_node_id = Some(id);
        result.computation_graph = Some(graph);
        Ok(result)
    }

    /// Directional derivative of Cholesky for a symmetric input tangent.
    pub fn cholesky_jvp(&self, tangent: &Self) -> GraphResult<Self> {
        self.cholesky_derivative(tangent, false)
    }
    /// Pull back a factor cotangent to a symmetric Frobenius input gradient.
    /// Off-diagonal entries contribute twice when a symmetric pair is varied together.
    pub fn cholesky_vjp(&self, cotangent: &Self) -> GraphResult<Self> {
        self.cholesky_derivative(cotangent, true)
    }
    fn cholesky_derivative(&self, direction: &Self, reverse: bool) -> GraphResult<Self> {
        let (rows, cols) = self.shape();
        if rows != cols || direction.shape() != self.shape() {
            return Err(GraphError::InvalidInput(
                "Cholesky derivative requires square input and a matching direction".into(),
            ));
        }
        if self
            .row_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "Cholesky row labels do not match shape".into(),
            ));
        }
        let values = super::decomposition_derivatives::cholesky(
            self.decomposition_input()?,
            direction.decomposition_input()?,
            rows,
            reverse,
        )?;
        let mut result = Self::factor_matrix(values, rows, cols)?;
        result.row_labels = self.row_labels.clone();
        result.column_names = if reverse {
            self.column_names.clone()
        } else {
            (0..rows).map(|i| format!("chol_{i}")).collect()
        };
        Ok(result)
    }

    /// General real-input eigendecomposition with dedicated real/imaginary outputs.
    /// Reject nonconvergence, defective/unresolved bases and failed residual checks.
    pub fn complex_eigenvalue_decomposition(&self) -> GraphResult<ComplexEigenResult<T>> {
        if self.requires_grad {
            return Err(GraphError::InvalidInput("Automatic gradients for general complex eigenpairs are unsupported; detach explicitly to compute owned factors".into()));
        }
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Complex eigen requires square input".into(),
            ));
        }
        let result = super::complex_eigen::eigen(self.decomposition_input()?, rows)?;
        let mut real = Self::factor_matrix(result.vectors_real, rows, rows)?;
        let mut imag = Self::factor_matrix(result.vectors_imag, rows, rows)?;
        let basis: Vec<_> = (0..rows).map(|i| format!("eigen_{i}")).collect();
        real.row_labels = Some(self.column_names.clone());
        imag.row_labels = real.row_labels.clone();
        real.column_names = basis.clone();
        imag.column_names = basis;
        Ok(ComplexEigenResult {
            eigenvalues_real: result.values_real,
            eigenvalues_imag: result.values_imag,
            eigenvectors_real: real,
            eigenvectors_imag: imag,
        })
    }

    /// Economy QR directional derivative on a resolved locally fixed sign branch.
    pub fn qr_jvp(&self, tangent: &Self) -> GraphResult<(Self, Self)> {
        let (m, n) = self.shape();
        let k = m.min(n);
        if tangent.shape() != self.shape() {
            return Err(GraphError::InvalidInput("QR tangent shape mismatch".into()));
        }
        let (q, r) = super::triangular_derivatives::qr_jvp(
            self.decomposition_input()?,
            tangent.decomposition_input()?,
            m,
            n,
        )?;
        Ok((Self::factor_matrix(q, m, k)?, Self::factor_matrix(r, k, n)?))
    }
    /// Economy QR pullback for matching Q/R cotangents.
    pub fn qr_vjp(&self, q: &Self, r: &Self) -> GraphResult<Self> {
        let (m, n) = self.shape();
        let k = m.min(n);
        if q.shape() != (m, k) || r.shape() != (k, n) {
            return Err(GraphError::InvalidInput(
                "QR cotangent shape mismatch".into(),
            ));
        }
        let values = super::triangular_derivatives::qr_vjp(
            self.decomposition_input()?,
            q.decomposition_input()?,
            r.decomposition_input()?,
            m,
            n,
        )?;
        let mut result = Self::factor_matrix(values, m, n)?;
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }
    /// Fixed-pivot LU directional derivative; permutation derivative is zero.
    pub fn lu_jvp(&self, tangent: &Self) -> GraphResult<(Self, Self)> {
        let (n, cols) = self.shape();
        if n != cols || tangent.shape() != self.shape() {
            return Err(GraphError::InvalidInput(
                "LU derivative requires matching square inputs".into(),
            ));
        }
        let (l, u) = super::triangular_derivatives::lu_jvp(
            self.decomposition_input()?,
            tangent.decomposition_input()?,
            n,
        )?;
        Ok((Self::factor_matrix(l, n, n)?, Self::factor_matrix(u, n, n)?))
    }
    /// Fixed-pivot LU pullback for lower/upper factor cotangents.
    pub fn lu_vjp(&self, l: &Self, u: &Self) -> GraphResult<Self> {
        let (n, cols) = self.shape();
        if n != cols || l.shape() != self.shape() || u.shape() != self.shape() {
            return Err(GraphError::InvalidInput(
                "LU derivative requires matching square cotangents".into(),
            ));
        }
        let values = super::triangular_derivatives::lu_vjp(
            self.decomposition_input()?,
            l.decomposition_input()?,
            u.decomposition_input()?,
            n,
        )?;
        let mut result = Self::factor_matrix(values, n, n)?;
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }

    /// Symmetric eigenvalue/eigenvector directional derivative; distinct spectrum required.
    pub fn eigen_jvp(&self, tangent: &Self) -> GraphResult<(Vec<f64>, Self)> {
        let (n, cols) = self.shape();
        if n != cols || tangent.shape() != self.shape() {
            return Err(GraphError::InvalidInput(
                "Eigen JVP requires matching square matrices".into(),
            ));
        }
        let (values, vectors) = super::spectral_derivatives::eigen_jvp(
            self.decomposition_input()?,
            tangent.decomposition_input()?,
            n,
        )?;
        Ok((values, Self::factor_matrix(vectors, n, n)?))
    }
    /// Symmetric Frobenius pullback of real symmetric eigenpairs.
    pub fn eigen_vjp(&self, values: &[f64], vectors: &Self) -> GraphResult<Self> {
        let (n, cols) = self.shape();
        if n != cols || values.len() != n || vectors.shape() != (n, n) {
            return Err(GraphError::InvalidInput(
                "Eigen VJP requires matching eigenvalue/vector cotangents".into(),
            ));
        }
        let values = super::spectral_derivatives::eigen_vjp(
            self.decomposition_input()?,
            values.to_vec(),
            vectors.decomposition_input()?,
            n,
        )?;
        let mut result = Self::factor_matrix(values, n, n)?;
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }
    /// Thin SVD directional derivative: dU(m*k), dS(k), dVt(k*n).
    pub fn svd_jvp(&self, tangent: &Self) -> GraphResult<(Self, Vec<f64>, Self)> {
        let (m, n) = self.shape();
        let k = m.min(n);
        if tangent.shape() != self.shape() {
            return Err(GraphError::InvalidInput(
                "SVD JVP requires a matching input tangent".into(),
            ));
        }
        let (u, s, vt) = super::spectral_derivatives::svd_jvp(
            self.decomposition_input()?,
            tangent.decomposition_input()?,
            m,
            n,
        )?;
        Ok((
            Self::factor_matrix(u, m, k)?,
            s,
            Self::factor_matrix(vt, k, n)?,
        ))
    }
    /// Thin SVD pullback; positive separated singular values required.
    pub fn svd_vjp(&self, u: &Self, s: &[f64], vt: &Self) -> GraphResult<Self> {
        let (m, n) = self.shape();
        let k = m.min(n);
        if u.shape() != (m, k) || s.len() != k || vt.shape() != (k, n) {
            return Err(GraphError::InvalidInput(
                "SVD VJP requires thin factor cotangents".into(),
            ));
        }
        let values = super::spectral_derivatives::svd_vjp(
            self.decomposition_input()?,
            u.decomposition_input()?,
            s.to_vec(),
            vt.decomposition_input()?,
            m,
            n,
        )?;
        let mut result = Self::factor_matrix(values, m, n)?;
        result.row_labels = self.row_labels.clone();
        result.column_names = self.column_names.clone();
        Ok(result)
    }

    /// Real symmetric eigendecomposition with descending eigenvalues.
    /// Eigenvectors are orthonormal columns; other domains reject explicitly.
    pub fn eigenvalue_decomposition(&self) -> GraphResult<(Vec<f64>, GraphMatrix<T>)>
    where
        T: Clone
            + Copy
            + Default
            + std::fmt::Debug
            + PartialEq
            + PartialOrd
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Eigendecomposition requires square matrix".into(),
            ));
        }
        let (values, vectors) =
            super::decomposition::symmetric_eigen(self.decomposition_input()?, rows)?;
        Ok((
            values,
            self.track_factor(
                Self::decomposition_matrix(vectors, rows, rows)?,
                Operation::FactorOutput {
                    part: FactorOutput::EigenVectors,
                },
            )?,
        ))
    }

    fn decomposition_input(&self) -> GraphResult<Vec<f64>> {
        use crate::storage::advanced_matrix::DType;
        if !matches!(T::DTYPE, DType::Float32 | DType::Float64) {
            return Err(GraphError::InvalidInput(
                "Decomposition requires floating-point matrix elements".into(),
            ));
        }
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }

        let mut data = Vec::with_capacity(
            rows.checked_mul(cols)
                .ok_or_else(|| GraphError::InvalidInput("Matrix dimensions overflow".into()))?,
        );
        for row in 0..rows {
            for col in 0..cols {
                data.push(
                    self.get(row, col)
                        .ok_or_else(|| {
                            GraphError::InvalidInput("Missing decomposition input value".into())
                        })?
                        .to_f64(),
                );
            }
        }
        Ok(data)
    }

    fn decomposition_matrix(values: Vec<f64>, rows: usize, cols: usize) -> GraphResult<Self> {
        let values = values
            .into_iter()
            .map(|value| {
                T::from_f64(value).ok_or_else(|| {
                    GraphError::InvalidInput(
                        "Decomposition output is not representable in its element type".into(),
                    )
                })
            })
            .collect::<GraphResult<_>>()?;
        Self::from_row_major_data(values, rows, cols, None)
    }

    /// Numerical rank using singular values and the input dtype's relative precision.
    pub fn rank(&self) -> GraphResult<usize> {
        self.rank_with_tolerance(self.default_rank_tolerance())
    }
    fn default_rank_tolerance(&self) -> f64 {
        let (rows, cols) = self.shape();
        let epsilon = if T::DTYPE == crate::storage::advanced_matrix::DType::Float32 {
            f64::from(f32::EPSILON)
        } else {
            f64::EPSILON
        };
        epsilon * rows.max(cols).max(1) as f64
    }
    /// Numerical rank with an explicit nonnegative finite relative cutoff.
    pub fn rank_with_tolerance(&self, relative: f64) -> GraphResult<usize> {
        if !relative.is_finite() || relative < 0. {
            return Err(GraphError::InvalidInput(
                "Rank tolerance must be finite and nonnegative".into(),
            ));
        }
        let singular = self.normalized_spectrum()?;
        let largest = singular.first().copied().unwrap_or(0.);
        if largest == 0. {
            return Ok(0);
        }
        Ok(singular
            .iter()
            .filter(|&&value| value / largest > relative)
            .count())
    }
    /// Estimated spectral (2-norm) condition number from normalized singular values.
    /// Zero matrices/zero smallest singular value yield infinity; empty order errors.
    pub fn condition_number(&self) -> GraphResult<f64> {
        let singular = self.normalized_spectrum()?;
        let largest = singular.first().copied().ok_or_else(|| {
            GraphError::InvalidInput("Condition number requires nonempty dimensions".into())
        })?;
        let smallest = singular.last().copied().unwrap_or(0.);
        Ok(if smallest == 0. {
            f64::INFINITY
        } else {
            largest / smallest
        })
    }
    fn normalized_spectrum(&self) -> GraphResult<Vec<f64>> {
        let (rows, cols) = self.shape();
        let mut values = self.decomposition_input()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GraphError::InvalidInput(
                "Numerical spectrum requires finite input".into(),
            ));
        }
        if rows == 0 || cols == 0 {
            return Ok(Vec::new());
        }
        let scale = values.iter().fold(0.0_f64, |scale, v| scale.max(v.abs()));
        if scale != 0. {
            for value in &mut values {
                *value /= scale;
            }
        }
        Ok(super::decomposition::svd(values, rows, cols)?.singular)
    }

    /// Tile (repeat) the matrix a specified number of times along each axis
    /// reps: (rows_repeat, cols_repeat) - how many times to repeat in each dimension
    pub fn tile(&self, reps: (usize, usize)) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
    {
        let (rows, cols) = self.shape();
        let (row_reps, col_reps) = reps;

        let new_rows = rows * row_reps;
        let new_cols = cols * col_reps;

        // Create new arrays for the tiled matrix
        let mut new_arrays = Vec::new();

        for new_col in 0..new_cols {
            let orig_col = new_col % cols;
            let mut column_data = Vec::with_capacity(new_rows);

            // Repeat the original column row_reps times
            for _ in 0..row_reps {
                for row in 0..rows {
                    let value = self.get(row, orig_col).unwrap_or_else(|| T::default());
                    column_data.push(value);
                }
            }

            new_arrays.push(NumArray::new(column_data));
        }

        GraphMatrix::from_arrays(new_arrays)
    }

    /// Repeat elements of the matrix along a specified axis
    /// repeats: number of times to repeat each element
    /// axis: 0 for rows, 1 for columns
    pub fn repeat(&self, repeats: usize, axis: usize) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
    {
        let (rows, cols) = self.shape();

        match axis {
            0 => {
                // Repeat along rows (each row gets repeated)
                let new_rows = rows * repeats;
                let mut new_arrays = Vec::new();

                for col in 0..cols {
                    let mut column_data = Vec::with_capacity(new_rows);

                    for row in 0..rows {
                        let value = self.get(row, col).unwrap_or_else(|| T::default());
                        // Repeat this value 'repeats' times
                        for _ in 0..repeats {
                            column_data.push(value);
                        }
                    }

                    new_arrays.push(NumArray::new(column_data));
                }

                GraphMatrix::from_arrays(new_arrays)
            }
            1 => {
                // Repeat along columns (each column gets repeated)
                let _new_cols = cols * repeats;
                let mut new_arrays = Vec::new();

                for col in 0..cols {
                    let column_data: Vec<T> = (0..rows)
                        .map(|row| self.get(row, col).unwrap_or_else(|| T::default()))
                        .collect();

                    // Add this column 'repeats' times
                    for _ in 0..repeats {
                        new_arrays.push(NumArray::new(column_data.clone()));
                    }
                }

                GraphMatrix::from_arrays(new_arrays)
            }
            _ => Err(GraphError::InvalidInput(
                "Axis must be 0 (rows) or 1 (columns)".into(),
            )),
        }
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        let mut new_arrays = Vec::new();

        for col in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row in 0..rows {
                let value = self.get(row, col).unwrap_or_else(|| T::default());
                let abs_value = f64::from(value).abs();
                column_data.push(T::from(abs_value));
            }

            new_arrays.push(NumArray::new(column_data));
        }

        GraphMatrix::from_arrays(new_arrays)
    }

    /// Element-wise exponential (e^x)
    pub fn exp(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        let mut new_arrays = Vec::new();

        for col in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row in 0..rows {
                let value = self.get(row, col).unwrap_or_else(|| T::default());
                let exp_value = f64::from(value).exp();
                column_data.push(T::from(exp_value));
            }

            new_arrays.push(NumArray::new(column_data));
        }

        GraphMatrix::from_arrays(new_arrays)
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        let mut new_arrays = Vec::new();

        for col in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row in 0..rows {
                let value = self.get(row, col).unwrap_or_else(|| T::default());
                let f_value = f64::from(value);

                // Handle negative values and zero (ln is undefined)
                let log_value = if f_value > 0.0 {
                    f_value.ln()
                } else {
                    f64::NAN // Or could return error
                };

                column_data.push(T::from(log_value));
            }

            new_arrays.push(NumArray::new(column_data));
        }

        GraphMatrix::from_arrays(new_arrays)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: Clone + Copy + Default,
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        let mut new_arrays = Vec::new();

        for col in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row in 0..rows {
                let value = self.get(row, col).unwrap_or_else(|| T::default());
                let f_value = f64::from(value);

                // Handle negative values (sqrt is undefined for real numbers)
                let sqrt_value = if f_value >= 0.0 {
                    f_value.sqrt()
                } else {
                    f64::NAN // Or could return error
                };

                column_data.push(T::from(sqrt_value));
            }

            new_arrays.push(NumArray::new(column_data));
        }

        GraphMatrix::from_arrays(new_arrays)
    }
}

// === CHECKED OPERATOR OVERLOADING IMPLEMENTATIONS ===
// Failures return GraphResult instead of substituted zero data.
impl<T: NumericType> std::ops::Add<&GraphMatrix<T>> for &GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn add(self, rhs: &GraphMatrix<T>) -> Self::Output {
        GraphMatrix::add(self, rhs)
    }
}
impl<T: NumericType> std::ops::Add<GraphMatrix<T>> for GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn add(self, rhs: GraphMatrix<T>) -> Self::Output {
        GraphMatrix::add(&self, &rhs)
    }
}
impl<T: NumericType> std::ops::Sub<&GraphMatrix<T>> for &GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn sub(self, rhs: &GraphMatrix<T>) -> Self::Output {
        GraphMatrix::subtract(self, rhs)
    }
}
impl<T: NumericType> std::ops::Sub<GraphMatrix<T>> for GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn sub(self, rhs: GraphMatrix<T>) -> Self::Output {
        GraphMatrix::subtract(&self, &rhs)
    }
}
impl<T: NumericType> std::ops::Mul<&GraphMatrix<T>> for &GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn mul(self, rhs: &GraphMatrix<T>) -> Self::Output {
        GraphMatrix::multiply(self, rhs)
    }
}
impl<T: NumericType> std::ops::Mul<GraphMatrix<T>> for GraphMatrix<T> {
    type Output = GraphResult<GraphMatrix<T>>;
    fn mul(self, rhs: GraphMatrix<T>) -> Self::Output {
        GraphMatrix::multiply(&self, &rhs)
    }
}

impl<T: NumericType> Default for GraphMatrix<T> {
    fn default() -> Self {
        GraphMatrix::zeros(0, 0)
    }
}

impl<T: NumericType> GraphMatrix<T> {
    /// Owned positional selection, preserving duplicates/order and rejecting invalid indices.
    pub fn select_indices(&self, rows: &[usize], cols: &[usize]) -> GraphResult<Self> {
        self.gather_indices(Some(rows), cols)
    }
    /// Owned column selection with unchanged row identity.
    pub fn select_columns(&self, indices: &[usize]) -> GraphResult<Self> {
        self.gather_indices(None, indices)
    }
    fn gather_indices(
        &self,
        selected_rows: Option<&[usize]>,
        selected_cols: &[usize],
    ) -> GraphResult<Self> {
        let (rows, cols) = self.shape();
        if selected_rows.is_some_and(|indices| indices.iter().any(|&i| i >= rows))
            || selected_cols.iter().any(|&i| i >= cols)
        {
            return Err(GraphError::InvalidInput(
                "Matrix selection index out of bounds".into(),
            ));
        }
        if self.column_names.len() != cols {
            return Err(GraphError::InvalidInput(
                "Column label count does not match shape".into(),
            ));
        }
        if self
            .row_labels
            .as_ref()
            .is_some_and(|labels| labels.len() != rows)
        {
            return Err(GraphError::InvalidInput(
                "Row label count does not match shape".into(),
            ));
        }
        let result_rows = selected_rows.map_or(rows, |r| r.len());
        if self.is_sparse() {
            let all_rows: Vec<_>;
            let rows = if let Some(rows) = selected_rows {
                rows
            } else {
                all_rows = (0..self.shape().0).collect();
                &all_rows
            };
            if let Some(storage) = self
                .storage
                .sparse_select(rows, selected_cols)
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?
            {
                let mut result = Self::from_storage(storage);
                result.column_names = selected_cols
                    .iter()
                    .map(|&i| self.column_names[i].clone())
                    .collect();
                result.row_labels = self
                    .row_labels
                    .as_ref()
                    .map(|labels| rows.iter().map(|&i| labels[i].clone()).collect());
                return Ok(result);
            }
        }
        let mut result = Self::arithmetic_zeros(result_rows, selected_cols.len())?;
        for (col, &source_col) in selected_cols.iter().enumerate() {
            for row in 0..result_rows {
                let source_row = selected_rows.map_or(row, |r| r[row]);
                result.set(row, col, self.get_checked(source_row, source_col)?)?;
            }
        }
        result.column_names = selected_cols
            .iter()
            .map(|&i| self.column_names[i].clone())
            .collect();
        result.row_labels = self.row_labels.as_ref().map(|labels| match selected_rows {
            Some(indices) => indices.iter().map(|&i| labels[i].clone()).collect(),
            None => labels.clone(),
        });
        Ok(result)
    }

    fn arithmetic_zeros(rows: usize, cols: usize) -> GraphResult<Self> {
        rows.checked_mul(cols)
            .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| GraphError::InvalidInput("Arithmetic output size overflow".into()))?;
        let storage = UnifiedMatrix::zeros(rows, cols).map_err(|e| {
            GraphError::InvalidInput(format!("Arithmetic allocation failed: {e:?}"))
        })?;
        Ok(Self::from_storage(storage))
    }

    // Arithmetic results own storage and retain left positional row/column labels.
    fn map_arithmetic(&self, operation: impl Fn(T) -> T) -> GraphResult<Self> {
        let (rows, cols) = self.shape();
        if self.is_sparse() {
            if let Some(storage) = self
                .storage
                .sparse_map(&operation)
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?
            {
                let mut result = Self::from_storage(storage);
                result.row_labels = self.row_labels.clone();
                result.column_names = self.column_names.clone();
                return Ok(result);
            }
        }
        let mut result = Self::arithmetic_zeros(rows, cols)?;
        for row in 0..rows {
            for col in 0..cols {
                result.set(row, col, operation(self.get_checked(row, col)?))?;
            }
        }
        result.column_names = self.column_names.clone();
        result.row_labels = self.row_labels.clone();
        Ok(result)
    }

    fn zip_arithmetic(&self, other: &Self, operation: fn(T, T) -> Option<T>) -> GraphResult<Self> {
        if self.shape() != other.shape() {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions must match for element-wise arithmetic: {:?} vs {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let (rows, cols) = self.shape();
        if self.is_sparse() && other.is_sparse() {
            let storage = self
                .storage
                .checked_zip(&other.storage, operation)
                .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
            let mut result = Self::from_storage(storage);
            result.row_labels = self.row_labels.clone();
            result.column_names = self.column_names.clone();
            return Ok(result);
        }
        let mut result = Self::arithmetic_zeros(rows, cols)?;
        for row in 0..rows {
            for col in 0..cols {
                result.set(
                    row,
                    col,
                    operation(self.get_checked(row, col)?, other.get_checked(row, col)?)
                        .ok_or_else(|| {
                            GraphError::InvalidInput(
                                "Integer element-wise arithmetic overflow".into(),
                            )
                        })?,
                )?;
            }
        }
        result.column_names = self.column_names.clone();
        result.row_labels = self.row_labels.clone();
        Ok(result)
    }

    /// Multiply each value by a scalar, preserving shape and positional labels.
    pub fn scalar_multiply(&self, scalar: f64) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Mul<Output = T> + From<f64>,
    {
        let scalar = T::from(scalar);
        self.map_arithmetic(|value| value * scalar)
    }

    /// Add two matrices positionally; mismatched shapes return an error.
    pub fn add(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        self.zip_arithmetic(other, T::checked_add)
    }

    /// Population variance per column (ddof=0), preserving this legacy convention.
    pub fn column_variances(&self) -> GraphResult<Vec<f64>> {
        self.reduce_axis(Axis::Columns, |values, count| {
            Ok(super::reductions::extended_variance(values, count, false))
        })
    }

    /// Concatenate matrices horizontally (by columns)
    pub fn concatenate_columns(matrices: Vec<GraphMatrix<T>>) -> GraphResult<GraphMatrix<T>> {
        if matrices.is_empty() {
            return Ok(GraphMatrix::zeros(0, 0));
        }

        if matrices.len() == 1 {
            return matrices[0].materialize();
        }

        // Use existing concatenate method with axis=1 (columns)
        let mut result = matrices[0].clone();
        for matrix in matrices.into_iter().skip(1) {
            result = result.concatenate(&matrix, 1)?;
        }

        Ok(result)
    }

    /// f64 mean per column; empty reductions are zero.
    pub fn column_means(&self) -> GraphResult<Vec<f64>> {
        self.reduce_axis(Axis::Columns, |values, count| {
            Ok(super::reductions::extended_mean(values, count))
        })
    }
}
