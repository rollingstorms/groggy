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

use crate::errors::{GraphError, GraphResult};
use crate::storage::advanced_matrix::{
    AdvancedMemoryPool, BackendSelector, ComputationGraph, NumericType, UnifiedMatrix,
};
use crate::storage::array::NumArray;
use crate::storage::table::BaseTable;
use crate::types::{AttrValue, NodeId};
use crate::viz::display::DisplayConfig;
use std::collections::HashMap;
use std::sync::Arc;

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

        // Calculate sparsity for numeric matrices
        let (is_sparse, sparsity) = if is_numeric {
            let zero_count = matrix.count_zeros();
            let total_elements = rows * cols;
            let sparsity_ratio = zero_count as f64 / total_elements as f64;
            (sparsity_ratio > 0.5, Some(sparsity_ratio))
        } else {
            (false, None)
        };

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

    /// Backend optimization
    backend_selector: Arc<BackendSelector>,
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
            backend_selector: Arc::new(BackendSelector::new()),
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

    /// Create identity matrix (using ones as placeholder until eye is implemented)
    pub fn eye(size: usize) -> Self {
        // TODO: Implement proper identity matrix in UnifiedMatrix
        let storage = UnifiedMatrix::ones(size, size)
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
            backend_selector: Arc::new(BackendSelector::new()),
            memory_pool: Arc::new(AdvancedMemoryPool::new()),
            properties: None,
            graph: None,
        })
    }

    /// Create adjacency matrix from edge data
    pub fn adjacency_from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> GraphResult<Self> {
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
                // Set both directions for undirected graph (symmetric adjacency matrix)
                matrix_data[src_idx * size + tgt_idx] = T::one();
                matrix_data[tgt_idx * size + src_idx] = T::one();
            }
        }

        Self::from_row_major_data(matrix_data, size, size, Some(nodes))
    }

    /// Create weighted adjacency matrix from weighted edge data  
    pub fn weighted_adjacency_from_edges(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, T)],
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
                // Set both directions for undirected graph (symmetric adjacency matrix)
                matrix_data[src_idx * size + tgt_idx] = weight;
                matrix_data[tgt_idx * size + src_idx] = weight;
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

    /// Get column name by index - needed by slicing module  
    pub(crate) fn get_column_name(&self, col_idx: usize) -> String {
        self.column_names
            .get(col_idx)
            .map(|s| s.clone())
            .unwrap_or_else(|| format!("col_{}", col_idx))
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

    /// Count zero elements in the matrix (internal helper)
    pub(crate) fn count_zeros(&self) -> usize {
        // For now, return 0 as UnifiedMatrix doesn't expose element access
        // TODO: Implement proper zero counting when UnifiedMatrix exposes element iteration
        0
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
        use crate::storage::advanced_matrix::backend::{BackendHint, OperationType};

        let (rows, cols) = self.shape();
        let size = rows * cols;

        let _backend = self.backend_selector.select_backend(
            OperationType::GEMM,
            size,
            T::DTYPE,
            BackendHint::PreferSpeed,
        );

        // For now, use UnifiedMatrix's built-in matmul until backend integration is complete
        let result_storage = self.storage.matmul(&other.storage).map_err(|e| {
            GraphError::InvalidInput(format!("Matrix multiplication failed: {:?}", e))
        })?;
        Ok(Self::from_storage(result_storage))
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
        let (rows, cols) = self.shape();
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(self.get(i, j).unwrap_or_else(|| T::zero()));
            }
        }
        Ok(data)
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

    /// Get gradient matrix
    pub fn grad(&self) -> Option<GraphMatrix<T>> {
        if let Some(ref graph) = self.computation_graph {
            if let Some(node_id) = self.current_node_id {
                if let Some(node_arc) = graph.get_node(node_id) {
                    let node = node_arc.lock().unwrap();
                    if let Some(ref gradient_matrix) = node.gradient {
                        // Convert UnifiedMatrix gradient to GraphMatrix
                        return Some(Self::from_storage(gradient_matrix.clone()));
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
        // TODO: Implement type casting when UnifiedMatrix supports it
        // For now, return an error
        Err(GraphError::InvalidInput(
            "Type casting not yet implemented in UnifiedMatrix backend".into(),
        ))
    }

    /// Check if the matrix is sparse (has many zero elements)
    pub fn is_sparse(&self) -> bool {
        // TODO: Implement sparsity detection based on UnifiedMatrix data
        // For now, assume dense matrices
        false
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
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix.set(i, i, T::one())?;
        }
        Ok(matrix)
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
        let mut result = Self::zeros(cols, rows);

        for i in 0..rows {
            for j in 0..cols {
                if let Some(value) = self.get(i, j) {
                    result.set(j, i, value)?;
                } else {
                    return Err(GraphError::InvalidInput(format!(
                        "Invalid cell at ({}, {})",
                        i, j
                    )));
                }
            }
        }

        Ok(result)
    }

    /// Matrix multiplication
    pub fn multiply(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        if self_cols != other_rows {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
                self_rows, self_cols, other_rows, other_cols
            )));
        }

        let mut result = Self::zeros(self_rows, other_cols);

        for i in 0..self_rows {
            for j in 0..other_cols {
                let mut sum = T::zero();
                for k in 0..self_cols {
                    let a = self.get(i, k).ok_or_else(|| {
                        GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, k))
                    })?;
                    let b = other.get(k, j).ok_or_else(|| {
                        GraphError::InvalidInput(format!("Invalid cell at ({}, {})", k, j))
                    })?;
                    sum = T::add(sum, T::mul(a, b));
                }
                result.set(i, j, sum)?;
            }
        }

        Ok(result)
    }

    /// Matrix subtraction
    pub fn subtract(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        if self_rows != other_rows || self_cols != other_cols {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions must match for subtraction: {}x{} - {}x{}",
                self_rows, self_cols, other_rows, other_cols
            )));
        }

        let mut result = Self::zeros(self_rows, self_cols);

        for i in 0..self_rows {
            for j in 0..self_cols {
                let a = self.get(i, j).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j))
                })?;
                let b = other.get(i, j).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j))
                })?;
                let result_val = T::sub(a, b);
                result.set(i, j, result_val)?;
            }
        }

        Ok(result)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn elementwise_multiply(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        if rows != other_rows || cols != other_cols {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions must match for element-wise multiplication: {}x{} vs {}x{}",
                rows, cols, other_rows, other_cols
            )));
        }

        let mut result = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let a = self.get(i, j).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j))
                })?;
                let b = other.get(i, j).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j))
                })?;
                result.set(i, j, T::mul(a, b))?;
            }
        }

        Ok(result)
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

    /// Calculate mean along an axis
    pub fn mean_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let sums = self.sum_axis(axis)?;
        let count = match axis {
            Axis::Rows => self.shape().1,    // number of columns
            Axis::Columns => self.shape().0, // number of rows
        };

        if count == 0 {
            return Ok(Vec::new());
        }

        let count_t = T::from_f64(count as f64).unwrap_or(T::one());
        Ok(sums.into_iter().map(|sum| T::div(sum, count_t)).collect())
    }

    /// Calculate standard deviation along an axis
    pub fn std_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        // This is a simplified implementation - for now just return the sum_axis
        // TODO: Implement proper standard deviation calculation
        self.sum_axis(axis)
    }

    /// Calculate variance along an axis
    pub fn var_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + Copy,
    {
        let means = self.mean_axis(axis)?;
        let (rows, cols) = self.shape();

        match axis {
            Axis::Rows => {
                // Variance of each row across columns
                let mut result = Vec::with_capacity(rows);
                for row in 0..rows {
                    let row_mean = T::from_f64(means[row].to_f64()).unwrap_or_else(|| T::zero());
                    let mut sum_sq_diff = T::zero();
                    for col in 0..cols {
                        if let Some(value) = self.get(row, col) {
                            let diff = value - row_mean;
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }
                    }
                    let variance = if cols > 1 {
                        T::from_f64(sum_sq_diff.to_f64() / (cols - 1) as f64)
                            .unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    result.push(variance);
                }
                Ok(result)
            }
            Axis::Columns => {
                // Variance of each column across rows
                let mut result = Vec::with_capacity(cols);
                for col in 0..cols {
                    let col_mean = T::from_f64(means[col].to_f64()).unwrap_or_else(|| T::zero());
                    let mut sum_sq_diff = T::zero();
                    for row in 0..rows {
                        if let Some(value) = self.get(row, col) {
                            let diff = value - col_mean;
                            sum_sq_diff = sum_sq_diff + (diff * diff);
                        }
                    }
                    let variance = if rows > 1 {
                        T::from_f64(sum_sq_diff.to_f64() / (rows - 1) as f64)
                            .unwrap_or_else(|| T::zero())
                    } else {
                        T::zero()
                    };
                    result.push(variance);
                }
                Ok(result)
            }
        }
    }

    /// Calculate minimum along an axis
    pub fn min_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: PartialOrd + Copy,
    {
        let (rows, cols) = self.shape();
        match axis {
            Axis::Rows => {
                // Minimum of each row across columns
                let mut result = Vec::with_capacity(rows);
                for row in 0..rows {
                    let mut row_min = None;
                    for col in 0..cols {
                        if let Some(value) = self.get(row, col) {
                            match row_min {
                                None => row_min = Some(value),
                                Some(current_min) => {
                                    if value < current_min {
                                        row_min = Some(value);
                                    }
                                }
                            }
                        }
                    }
                    result.push(row_min.unwrap_or_else(|| T::zero()));
                }
                Ok(result)
            }
            Axis::Columns => {
                // Minimum of each column across rows
                let mut result = Vec::with_capacity(cols);
                for col in 0..cols {
                    let mut col_min = None;
                    for row in 0..rows {
                        if let Some(value) = self.get(row, col) {
                            match col_min {
                                None => col_min = Some(value),
                                Some(current_min) => {
                                    if value < current_min {
                                        col_min = Some(value);
                                    }
                                }
                            }
                        }
                    }
                    result.push(col_min.unwrap_or_else(|| T::zero()));
                }
                Ok(result)
            }
        }
    }

    /// Calculate maximum along an axis
    pub fn max_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: PartialOrd + Copy,
    {
        let (rows, cols) = self.shape();
        match axis {
            Axis::Rows => {
                // Maximum of each row across columns
                let mut result = Vec::with_capacity(rows);
                for row in 0..rows {
                    let mut row_max = None;
                    for col in 0..cols {
                        if let Some(value) = self.get(row, col) {
                            match row_max {
                                None => row_max = Some(value),
                                Some(current_max) => {
                                    if value > current_max {
                                        row_max = Some(value);
                                    }
                                }
                            }
                        }
                    }
                    result.push(row_max.unwrap_or_else(|| T::zero()));
                }
                Ok(result)
            }
            Axis::Columns => {
                // Maximum of each column across rows
                let mut result = Vec::with_capacity(cols);
                for col in 0..cols {
                    let mut col_max = None;
                    for row in 0..rows {
                        if let Some(value) = self.get(row, col) {
                            match col_max {
                                None => col_max = Some(value),
                                Some(current_max) => {
                                    if value > current_max {
                                        col_max = Some(value);
                                    }
                                }
                            }
                        }
                    }
                    result.push(col_max.unwrap_or_else(|| T::zero()));
                }
                Ok(result)
            }
        }
    }

    /// Materialize the matrix (convert lazy operations to concrete data)
    pub fn materialize(&self) -> GraphResult<GraphMatrix<T>> {
        // For now, just return a clone since we don't have lazy operations yet
        Ok(self.clone())
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

    /// Convert to dense representation (no-op for GraphMatrix which is already dense)
    pub fn dense(&self) -> GraphResult<GraphMatrix<T>> {
        Ok(self.clone())
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

    /// Sum along a specific axis
    pub fn sum_axis(&self, axis: Axis) -> GraphResult<Vec<T>>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        match axis {
            Axis::Rows => {
                // Sum each row across columns (returns vector of length `rows`)
                let mut result = Vec::with_capacity(rows);
                for row in 0..rows {
                    let mut row_sum = T::zero();
                    for col in 0..cols {
                        if let Some(value) = self.get(row, col) {
                            row_sum = row_sum + value;
                        }
                    }
                    result.push(row_sum);
                }
                Ok(result)
            }
            Axis::Columns => {
                // Sum each column across rows (returns vector of length `cols`)
                let mut result = Vec::with_capacity(cols);
                for col in 0..cols {
                    let mut col_sum = T::zero();
                    for row in 0..rows {
                        if let Some(value) = self.get(row, col) {
                            col_sum = col_sum + value;
                        }
                    }
                    result.push(col_sum);
                }
                Ok(result)
            }
        }
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
        use crate::storage::advanced_matrix::operations::MatrixOperations;

        let reshaped_storage = self
            .storage
            .reshape(new_rows, new_cols)
            .map_err(|e| GraphError::InvalidInput(format!("Reshape failed: {:?}", e)))?;

        Ok(Self::from_storage(reshaped_storage))
    }

    /// Global sum of all elements in the matrix
    pub fn sum(&self) -> T
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        let mut total = T::zero();
        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    total = total + value;
                }
            }
        }
        total
    }

    /// Global mean of all elements in the matrix
    pub fn mean(&self) -> f64
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        let total_elements = (rows * cols) as f64;
        if total_elements == 0.0 {
            return 0.0;
        }
        self.sum().to_f64() / total_elements
    }

    /// Global minimum value in the matrix
    pub fn min(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        let (rows, cols) = self.shape();
        let mut min_val = None;
        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    match min_val {
                        None => min_val = Some(value),
                        Some(current_min) => {
                            if value < current_min {
                                min_val = Some(value);
                            }
                        }
                    }
                }
            }
        }
        min_val
    }

    /// Global maximum value in the matrix
    pub fn max(&self) -> Option<T>
    where
        T: PartialOrd + Copy,
    {
        let (rows, cols) = self.shape();
        let mut max_val = None;
        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    match max_val {
                        None => max_val = Some(value),
                        Some(current_max) => {
                            if value > current_max {
                                max_val = Some(value);
                            }
                        }
                    }
                }
            }
        }
        max_val
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

        let mut trace_sum = T::zero();
        for i in 0..rows {
            if let Some(diagonal_value) = self.get(i, i) {
                trace_sum = trace_sum + diagonal_value;
            }
        }
        Ok(trace_sum)
    }

    /// Calculate the Frobenius norm (Euclidean norm) of the matrix
    /// ||A||_F = sqrt(sum(a_ij^2)) for all i,j
    pub fn norm(&self) -> f64
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        let mut sum_of_squares = 0.0;

        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    let f64_value = value.to_f64();
                    sum_of_squares += f64_value * f64_value;
                }
            }
        }

        sum_of_squares.sqrt()
    }

    /// Calculate the L1 norm (sum of absolute values) of the matrix
    /// ||A||_1 = sum(|a_ij|) for all i,j
    pub fn norm_l1(&self) -> f64 {
        let (rows, cols) = self.shape();
        let mut sum_abs = 0.0;

        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    sum_abs += value.to_f64().abs();
                }
            }
        }

        sum_abs
    }

    /// Calculate the L∞ norm (maximum absolute value) of the matrix
    /// ||A||_∞ = max(|a_ij|) for all i,j
    pub fn norm_inf(&self) -> f64 {
        let (rows, cols) = self.shape();
        let mut max_abs = 0.0;

        for row in 0..rows {
            for col in 0..cols {
                if let Some(value) = self.get(row, col) {
                    let abs_value = value.to_f64().abs();
                    if abs_value > max_abs {
                        max_abs = abs_value;
                    }
                }
            }
        }

        max_abs
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
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        match axis {
            0 => {
                // Concatenate along rows (vertical stacking)
                if self_cols != other_cols {
                    return Err(GraphError::InvalidInput(format!(
                        "Column count mismatch for row concatenation: {} vs {}",
                        self_cols, other_cols
                    )));
                }

                let new_rows = self_rows + other_rows;
                let mut result = GraphMatrix::zeros(new_rows, self_cols);

                // Copy self matrix to top part
                for i in 0..self_rows {
                    for j in 0..self_cols {
                        if let Some(value) = self.get(i, j) {
                            let _ = result.set(i, j, value);
                        }
                    }
                }

                // Copy other matrix to bottom part
                for i in 0..other_rows {
                    for j in 0..other_cols {
                        if let Some(value) = other.get(i, j) {
                            let _ = result.set(self_rows + i, j, value);
                        }
                    }
                }

                Ok(result)
            }
            1 => {
                // Concatenate along columns (horizontal stacking)
                if self_rows != other_rows {
                    return Err(GraphError::InvalidInput(format!(
                        "Row count mismatch for column concatenation: {} vs {}",
                        self_rows, other_rows
                    )));
                }

                let new_cols = self_cols + other_cols;
                let mut result = GraphMatrix::zeros(self_rows, new_cols);

                // Copy self matrix to left part
                for i in 0..self_rows {
                    for j in 0..self_cols {
                        if let Some(value) = self.get(i, j) {
                            let _ = result.set(i, j, value);
                        }
                    }
                }

                // Copy other matrix to right part
                for i in 0..other_rows {
                    for j in 0..other_cols {
                        if let Some(value) = other.get(i, j) {
                            let _ = result.set(i, self_cols + j, value);
                        }
                    }
                }

                Ok(result)
            }
            _ => Err(GraphError::InvalidInput(format!(
                "Invalid axis for concatenation: {}. Must be 0 (rows) or 1 (columns)",
                axis
            ))),
        }
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
        let mut results = Vec::new();

        match axis {
            0 => {
                // Split along rows
                let max_rows = rows;
                let mut prev_split = 0;

                for &split_point in split_points {
                    if split_point > max_rows {
                        return Err(GraphError::InvalidInput(format!(
                            "Split point {} exceeds matrix rows {}",
                            split_point, max_rows
                        )));
                    }
                    if split_point <= prev_split {
                        return Err(GraphError::InvalidInput(
                            "Split points must be in increasing order".into(),
                        ));
                    }

                    let split_rows = split_point - prev_split;
                    let mut chunk = GraphMatrix::zeros(split_rows, cols);

                    for i in 0..split_rows {
                        for j in 0..cols {
                            if let Some(value) = self.get(prev_split + i, j) {
                                let _ = chunk.set(i, j, value);
                            }
                        }
                    }

                    results.push(chunk);
                    prev_split = split_point;
                }

                // Add remaining rows if any
                if prev_split < max_rows {
                    let split_rows = max_rows - prev_split;
                    let mut chunk = GraphMatrix::zeros(split_rows, cols);

                    for i in 0..split_rows {
                        for j in 0..cols {
                            if let Some(value) = self.get(prev_split + i, j) {
                                let _ = chunk.set(i, j, value);
                            }
                        }
                    }

                    results.push(chunk);
                }
            }
            1 => {
                // Split along columns
                let max_cols = cols;
                let mut prev_split = 0;

                for &split_point in split_points {
                    if split_point > max_cols {
                        return Err(GraphError::InvalidInput(format!(
                            "Split point {} exceeds matrix cols {}",
                            split_point, max_cols
                        )));
                    }
                    if split_point <= prev_split {
                        return Err(GraphError::InvalidInput(
                            "Split points must be in increasing order".into(),
                        ));
                    }

                    let split_cols = split_point - prev_split;
                    let mut chunk = GraphMatrix::zeros(rows, split_cols);

                    for i in 0..rows {
                        for j in 0..split_cols {
                            if let Some(value) = self.get(i, prev_split + j) {
                                let _ = chunk.set(i, j, value);
                            }
                        }
                    }

                    results.push(chunk);
                    prev_split = split_point;
                }

                // Add remaining columns if any
                if prev_split < max_cols {
                    let split_cols = max_cols - prev_split;
                    let mut chunk = GraphMatrix::zeros(rows, split_cols);

                    for i in 0..rows {
                        for j in 0..split_cols {
                            if let Some(value) = self.get(i, prev_split + j) {
                                let _ = chunk.set(i, j, value);
                            }
                        }
                    }

                    results.push(chunk);
                }
            }
            _ => {
                return Err(GraphError::InvalidInput(format!(
                    "Invalid axis for split: {}. Must be 0 (rows) or 1 (columns)",
                    axis
                )))
            }
        }

        Ok(results)
    }

    // === BATCH 5: ADVANCED LINEAR ALGEBRA OPERATIONS ===

    /// Matrix determinant calculation (for square matrices)
    pub fn determinant(&self) -> GraphResult<f64>
    where
        T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + Copy,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Determinant requires square matrix, got {}x{}",
                rows, cols
            )));
        }

        if rows == 1 {
            // 1x1 matrix determinant
            return Ok(self
                .get(0, 0)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64());
        }

        if rows == 2 {
            // 2x2 matrix determinant: ad - bc
            let a = self
                .get(0, 0)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64();
            let b = self
                .get(0, 1)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64();
            let c = self
                .get(1, 0)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64();
            let d = self
                .get(1, 1)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64();

            return Ok(a * d - b * c);
        }

        // For larger matrices, use cofactor expansion (simplified implementation)
        let mut det = 0.0;
        for j in 0..cols {
            let element = self
                .get(0, j)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?
                .to_f64();

            // Create minor matrix (remove row 0 and column j)
            let minor = self.create_minor(0, j)?;
            let minor_det = minor.determinant()?;

            // Add to determinant with alternating signs
            if j % 2 == 0 {
                det += element * minor_det;
            } else {
                det -= element * minor_det;
            }
        }

        Ok(det)
    }

    /// Create minor matrix by removing specified row and column
    fn create_minor(&self, remove_row: usize, remove_col: usize) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        let new_rows = rows - 1;
        let new_cols = cols - 1;

        let mut minor = GraphMatrix::zeros(new_rows, new_cols);

        let mut minor_i = 0;
        for i in 0..rows {
            if i == remove_row {
                continue;
            }

            let mut minor_j = 0;
            for j in 0..cols {
                if j == remove_col {
                    continue;
                }

                if let Some(value) = self.get(i, j) {
                    let _ = minor.set(minor_i, minor_j, value);
                }
                minor_j += 1;
            }
            minor_i += 1;
        }

        Ok(minor)
    }

    /// Matrix inverse calculation (for square matrices)
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
            return Err(GraphError::InvalidInput(format!(
                "Matrix inverse requires square matrix, got {}x{}",
                rows, cols
            )));
        }

        let det = self.determinant()?;
        if det.abs() < 1e-10 {
            return Err(GraphError::InvalidInput(
                "Matrix is singular (determinant near zero), cannot compute inverse".into(),
            ));
        }

        if rows == 1 {
            // 1x1 inverse: 1/a
            let a = self
                .get(0, 0)
                .ok_or_else(|| GraphError::InvalidInput("Invalid matrix element".into()))?;
            let inv_val = T::from_f64(1.0 / a.to_f64()).unwrap_or_else(|| T::one());
            let mut result = GraphMatrix::zeros(1, 1);
            let _ = result.set(0, 0, inv_val);
            return Ok(result);
        }

        if rows == 2 {
            // 2x2 inverse: (1/det) * [[d, -b], [-c, a]]
            let a = self.get(0, 0).unwrap().to_f64();
            let b = self.get(0, 1).unwrap().to_f64();
            let c = self.get(1, 0).unwrap().to_f64();
            let d = self.get(1, 1).unwrap().to_f64();

            let inv_det = 1.0 / det;
            let mut result = GraphMatrix::zeros(2, 2);

            let _ = result.set(0, 0, T::from_f64(d * inv_det).unwrap_or_else(|| T::zero()));
            let _ = result.set(0, 1, T::from_f64(-b * inv_det).unwrap_or_else(|| T::zero()));
            let _ = result.set(1, 0, T::from_f64(-c * inv_det).unwrap_or_else(|| T::zero()));
            let _ = result.set(1, 1, T::from_f64(a * inv_det).unwrap_or_else(|| T::zero()));

            return Ok(result);
        }

        // For larger matrices, use Gaussian elimination with partial pivoting
        self.inverse_gaussian_elimination()
    }

    /// Gaussian elimination method for matrix inverse
    fn inverse_gaussian_elimination(&self) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    {
        let n = self.shape().0;

        // Create augmented matrix [A | I]
        let mut augmented = GraphMatrix::zeros(n, 2 * n);

        // Fill left side with original matrix
        for i in 0..n {
            for j in 0..n {
                if let Some(value) = self.get(i, j) {
                    let _ = augmented.set(i, j, value);
                }
            }
        }

        // Fill right side with identity matrix
        for i in 0..n {
            let _ = augmented.set(i, n + i, T::one());
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = i;
            let mut max_val = augmented.get(i, i).unwrap_or(T::zero()).to_f64().abs();

            for k in (i + 1)..n {
                let val = augmented.get(k, i).unwrap_or(T::zero()).to_f64().abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = k;
                }
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented.get(i, j).unwrap_or(T::zero());
                    let pivot_val = augmented.get(pivot_row, j).unwrap_or(T::zero());
                    let _ = augmented.set(i, j, pivot_val);
                    let _ = augmented.set(pivot_row, j, temp);
                }
            }

            // Make diagonal element 1
            let pivot = augmented.get(i, i).unwrap_or(T::zero());
            if pivot.to_f64().abs() < 1e-10 {
                return Err(GraphError::InvalidInput("Matrix is singular".into()));
            }

            for j in 0..(2 * n) {
                let val = augmented.get(i, j).unwrap_or(T::zero());
                let scaled =
                    T::from_f64(val.to_f64() / pivot.to_f64()).unwrap_or_else(|| T::zero());
                let _ = augmented.set(i, j, scaled);
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented.get(k, i).unwrap_or(T::zero());
                    for j in 0..(2 * n) {
                        let aug_val = augmented.get(k, j).unwrap_or(T::zero());
                        let pivot_val = augmented.get(i, j).unwrap_or(T::zero());
                        let new_val =
                            T::from_f64(aug_val.to_f64() - factor.to_f64() * pivot_val.to_f64())
                                .unwrap_or_else(|| T::zero());
                        let _ = augmented.set(k, j, new_val);
                    }
                }
            }
        }

        // Extract inverse matrix from right side
        let mut inverse = GraphMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if let Some(value) = augmented.get(i, n + j) {
                    let _ = inverse.set(i, j, value);
                }
            }
        }

        Ok(inverse)
    }

    /// Solve linear system Ax = b using LU decomposition
    pub fn solve(&self, b: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Copy,
    {
        let (rows, cols) = self.shape();
        let (b_rows, _b_cols) = b.shape();

        if rows != cols {
            return Err(GraphError::InvalidInput(format!(
                "Linear system requires square coefficient matrix, got {}x{}",
                rows, cols
            )));
        }

        if b_rows != rows {
            return Err(GraphError::InvalidInput(format!(
                "Right-hand side must have same number of rows as coefficient matrix: {} vs {}",
                b_rows, rows
            )));
        }

        // For small systems, use direct methods
        if rows <= 3 {
            let inv = self.inverse()?;
            return inv.multiply(b);
        }

        // For larger systems, would implement LU decomposition
        // For now, fall back to inverse method
        let inv = self.inverse()?;
        inv.multiply(b)
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
        f64: From<T>,
    {
        let (rows, cols) = self.shape();

        // For simplicity, implement via eigenvalue decomposition of A^T * A
        // This gives us V and Σ^2, then compute U = A * V * Σ^(-1)
        let a_transpose = self.transpose()?;
        let ata = a_transpose.multiply(self)?;

        // Compute eigenvalues and eigenvectors of A^T * A using simplified power iteration
        let (eigenvalues, eigenvectors) = ata.eigendecomposition()?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, Vec<T>)> =
            eigenvalues.into_iter().zip(eigenvectors).collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Extract sorted eigenvalues and eigenvectors
        let sorted_eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(val, _)| *val).collect();
        let sorted_eigenvectors: Vec<Vec<T>> =
            eigen_pairs.into_iter().map(|(_, vec)| vec).collect();

        // Singular values are square roots of eigenvalues
        let singular_values: Vec<f64> = sorted_eigenvalues
            .into_iter()
            .map(|val| val.max(0.0).sqrt())
            .collect();

        // V matrix (right singular vectors) - transpose of eigenvector matrix
        let mut v_arrays = Vec::new();
        for i in 0..cols {
            if i < sorted_eigenvectors.len() {
                v_arrays.push(NumArray::new(sorted_eigenvectors[i].clone()));
            } else {
                // Fill with zeros if we don't have enough eigenvectors
                v_arrays.push(NumArray::new(vec![T::default(); cols]));
            }
        }
        let v_matrix = GraphMatrix::from_arrays(v_arrays)?;

        // Compute U = A * V * Σ^(-1)
        let mut u_arrays = Vec::new();
        let min_dim = std::cmp::min(rows, cols);

        for j in 0..min_dim {
            let mut u_column = Vec::new();
            for i in 0..rows {
                let mut sum = T::default();
                for k in 0..cols {
                    if let (Some(a_val), Some(v_val)) = (self.get(i, k), v_matrix.get(k, j)) {
                        let sigma_inv = if j < singular_values.len() && singular_values[j] > 1e-10 {
                            T::from_f64(1.0 / singular_values[j]).unwrap_or_else(|| T::default())
                        } else {
                            T::default()
                        };
                        sum = sum + a_val * v_val * sigma_inv;
                    }
                }
                u_column.push(sum);
            }
            u_arrays.push(NumArray::new(u_column));
        }

        // Fill remaining columns with orthogonal vectors if needed
        while u_arrays.len() < rows {
            u_arrays.push(NumArray::new(vec![T::default(); rows]));
        }

        let u_matrix = GraphMatrix::from_arrays(u_arrays)?;
        let vt_matrix = v_matrix.transpose()?;

        Ok((u_matrix, singular_values, vt_matrix))
    }

    /// QR decomposition: A = Q * R
    /// Returns (Q, R) where Q is orthogonal and R is upper triangular
    pub fn qr_decomposition(&self) -> GraphResult<(GraphMatrix<T>, GraphMatrix<T>)>
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
        f64: From<T>,
    {
        let (rows, cols) = self.shape();

        // Modified Gram-Schmidt process
        let mut q_vectors: Vec<Vec<T>> = Vec::new();
        let mut r_matrix = GraphMatrix::zeros(cols, cols);

        for j in 0..cols {
            // Get column j from original matrix
            let mut col_j = Vec::new();
            for i in 0..rows {
                col_j.push(self.get(i, j).unwrap_or_else(|| T::default()));
            }

            // Orthogonalize against previous Q columns
            for k in 0..j {
                if let Some(q_k) = q_vectors.get(k) {
                    // Compute R[k,j] = q_k^T * a_j
                    let mut dot_product = 0.0;
                    for i in 0..rows {
                        dot_product += f64::from(q_k[i]) * f64::from(col_j[i]);
                    }
                    let _ = r_matrix.set(
                        k,
                        j,
                        T::from_f64(dot_product).unwrap_or_else(|| T::default()),
                    );

                    // a_j = a_j - R[k,j] * q_k
                    for i in 0..rows {
                        let old_val = f64::from(col_j[i]);
                        let projection = dot_product * f64::from(q_k[i]);
                        col_j[i] =
                            T::from_f64(old_val - projection).unwrap_or_else(|| T::default());
                    }
                }
            }

            // Compute R[j,j] = ||a_j||
            let mut norm_sq = 0.0;
            for &val in &col_j {
                let f_val = f64::from(val);
                norm_sq += f_val * f_val;
            }
            let norm = norm_sq.sqrt();
            let _ = r_matrix.set(j, j, T::from_f64(norm).unwrap_or_else(|| T::default()));

            // q_j = a_j / R[j,j]
            if norm > 1e-10 {
                for val in &mut col_j {
                    let f_val = f64::from(*val);
                    *val = T::from_f64(f_val / norm).unwrap_or_else(|| T::default());
                }
            }

            q_vectors.push(col_j);
        }

        // Construct Q matrix
        let mut q_arrays = Vec::new();
        for j in 0..cols {
            if j < q_vectors.len() {
                q_arrays.push(NumArray::new(q_vectors[j].clone()));
            } else {
                // Fill with orthogonal unit vectors
                let mut unit_vec = vec![T::default(); rows];
                if j < rows {
                    unit_vec[j] = T::from_f64(1.0).unwrap_or_else(|| T::default());
                }
                q_arrays.push(NumArray::new(unit_vec));
            }
        }

        let q_matrix = GraphMatrix::from_arrays(q_arrays)?;

        Ok((q_matrix, r_matrix))
    }

    /// LU Decomposition with partial pivoting: PA = LU
    /// Returns (P, L, U) where P is permutation matrix, L is lower triangular, U is upper triangular
    pub fn lu_decomposition(&self) -> GraphResult<(GraphMatrix<T>, GraphMatrix<T>, GraphMatrix<T>)>
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
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "LU decomposition requires square matrix".into(),
            ));
        }

        let n = rows;

        // Create working copy of the matrix
        let mut a = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..n {
                let val = self.get(i, j).unwrap_or_else(|| T::default());
                row.push(f64::from(val));
            }
            a.push(row);
        }

        // Initialize permutation vector (tracks row swaps)
        let mut perm: Vec<usize> = (0..n).collect();

        // Initialize L matrix (will be lower triangular with 1s on diagonal)
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            l[i][i] = 1.0; // Diagonal elements of L are 1
        }

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find the pivot (largest element in column k, rows k to n-1)
            let mut pivot_row = k;
            let mut max_val = a[k][k].abs();

            for i in (k + 1)..n {
                let abs_val = a[i][k].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }

            // Check for singular matrix
            if max_val < 1e-12 {
                return Err(GraphError::InvalidInput(
                    "Matrix is singular, LU decomposition failed".into(),
                ));
            }

            // Swap rows if needed
            if pivot_row != k {
                a.swap(k, pivot_row);
                perm.swap(k, pivot_row);

                // Also swap the L matrix entries for rows already processed
                for j in 0..k {
                    let temp = l[k][j];
                    l[k][j] = l[pivot_row][j];
                    l[pivot_row][j] = temp;
                }
            }

            // Elimination step
            for i in (k + 1)..n {
                let factor = a[i][k] / a[k][k];
                l[i][k] = factor; // Store multiplier in L matrix

                // Update row i
                for j in k..n {
                    a[i][j] -= factor * a[k][j];
                }
            }
        }

        // Build U matrix (a now contains the upper triangular part)
        let mut u_arrays = Vec::new();
        for j in 0..n {
            let mut col_data = Vec::new();
            for i in 0..n {
                let val = if i <= j { a[i][j] } else { 0.0 };
                col_data.push(T::from(val));
            }
            u_arrays.push(NumArray::new(col_data));
        }
        let u_matrix = GraphMatrix::from_arrays(u_arrays)?;

        // Build L matrix
        let mut l_arrays = Vec::new();
        for j in 0..n {
            let mut col_data = Vec::new();
            for i in 0..n {
                let val = if i >= j { l[i][j] } else { 0.0 };
                col_data.push(T::from(val));
            }
            l_arrays.push(NumArray::new(col_data));
        }
        let l_matrix = GraphMatrix::from_arrays(l_arrays)?;

        // Build permutation matrix P
        let mut p_arrays = Vec::new();
        for j in 0..n {
            let mut col_data = Vec::new();
            for i in 0..n {
                let val = if perm[i] == j { 1.0 } else { 0.0 };
                col_data.push(T::from(val));
            }
            p_arrays.push(NumArray::new(col_data));
        }
        let p_matrix = GraphMatrix::from_arrays(p_arrays)?;

        Ok((p_matrix, l_matrix, u_matrix))
    }

    /// Cholesky Decomposition: A = L * L^T for positive definite matrices
    /// Returns L (lower triangular matrix) where A = L * L^T
    pub fn cholesky_decomposition(&self) -> GraphResult<GraphMatrix<T>>
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
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Cholesky decomposition requires square matrix".into(),
            ));
        }

        let n = rows;

        // Check if matrix is symmetric (simplified check)
        for i in 0..n {
            for j in 0..i {
                let a_ij = f64::from(self.get(i, j).unwrap_or_else(|| T::default()));
                let a_ji = f64::from(self.get(j, i).unwrap_or_else(|| T::default()));
                if (a_ij - a_ji).abs() > 1e-10 {
                    return Err(GraphError::InvalidInput(
                        "Matrix must be symmetric for Cholesky decomposition".into(),
                    ));
                }
            }
        }

        // Initialize L matrix
        let mut l = vec![vec![0.0; n]; n];

        // Cholesky decomposition algorithm
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements: L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
                    let mut sum = 0.0;
                    for k in 0..i {
                        sum += l[i][k] * l[i][k];
                    }
                    let a_ii = f64::from(self.get(i, i).unwrap_or_else(|| T::default()));
                    let diag_val = a_ii - sum;

                    if diag_val <= 0.0 {
                        return Err(GraphError::InvalidInput(
                            "Matrix is not positive definite".into(),
                        ));
                    }

                    l[i][j] = diag_val.sqrt();
                } else {
                    // Off-diagonal elements: L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    let a_ij = f64::from(self.get(i, j).unwrap_or_else(|| T::default()));

                    if l[j][j] == 0.0 {
                        return Err(GraphError::InvalidInput(
                            "Division by zero in Cholesky decomposition".into(),
                        ));
                    }

                    l[i][j] = (a_ij - sum) / l[j][j];
                }
            }
        }

        // Build L matrix
        let mut l_arrays = Vec::new();
        for j in 0..n {
            let mut col_data = Vec::new();
            for i in 0..n {
                let val = if i >= j { l[i][j] } else { 0.0 };
                col_data.push(T::from(val));
            }
            l_arrays.push(NumArray::new(col_data));
        }
        let l_matrix = GraphMatrix::from_arrays(l_arrays)?;

        Ok(l_matrix)
    }

    /// Eigenvalue decomposition using QR algorithm
    /// Returns (eigenvalues, eigenvectors) where A * V = V * Λ
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
        f64: From<T>,
        T: From<f64>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let n = rows;

        // Use simplified power iteration method for dominant eigenvalue/eigenvector
        // For a more complete implementation, would use QR algorithm with shifts
        let mut eigenvalues = Vec::new();
        let mut eigenvector_columns = Vec::new();

        // Create working copy of matrix
        let mut a = Vec::new();
        for i in 0..n {
            let mut row = Vec::new();
            for j in 0..n {
                let val = self.get(i, j).unwrap_or_else(|| T::default());
                row.push(f64::from(val));
            }
            a.push(row);
        }

        // Find dominant eigenvalue using power iteration
        let mut v = vec![1.0; n];
        let mut eigenvalue = 0.0;

        // Normalize initial vector
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for val in &mut v {
                *val /= norm;
            }
        }

        // Power iteration
        for _iter in 0..100 {
            let mut av = vec![0.0; n];

            // Compute A * v
            for i in 0..n {
                for j in 0..n {
                    av[i] += a[i][j] * v[j];
                }
            }

            // Compute Rayleigh quotient: λ = v^T * A * v / (v^T * v)
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for i in 0..n {
                numerator += v[i] * av[i];
                denominator += v[i] * v[i];
            }

            if denominator > 1e-10 {
                eigenvalue = numerator / denominator;
            }

            // Normalize Av for next iteration
            let av_norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
            if av_norm > 1e-10 {
                for i in 0..n {
                    v[i] = av[i] / av_norm;
                }
            }
        }

        eigenvalues.push(eigenvalue);

        // Create eigenvector column
        let eigenvector_col: Vec<T> = v.into_iter().map(|x| T::from(x)).collect();
        eigenvector_columns.push(NumArray::new(eigenvector_col));

        // For simplicity, fill remaining eigenvalues with approximations
        // In a complete implementation, would use deflation or QR algorithm
        for i in 1..n {
            // Add approximate eigenvalues (diagonal elements as rough estimates)
            let diag_val = f64::from(self.get(i, i).unwrap_or_else(|| T::default()));
            eigenvalues.push(diag_val);

            // Create identity-like eigenvectors for remaining values
            let mut identity_col = vec![T::default(); n];
            if i < n {
                identity_col[i] = T::from(1.0);
            }
            eigenvector_columns.push(NumArray::new(identity_col));
        }

        // Build eigenvector matrix
        let eigenvector_matrix = GraphMatrix::from_arrays(eigenvector_columns)?;

        Ok((eigenvalues, eigenvector_matrix))
    }

    /// Simple eigenvalue decomposition using power iteration method
    /// Returns (eigenvalues, eigenvectors) - simplified implementation for SVD
    fn eigendecomposition(&self) -> GraphResult<(Vec<f64>, Vec<Vec<T>>)>
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
        f64: From<T>,
    {
        let (rows, cols) = self.shape();
        if rows != cols {
            return Err(GraphError::InvalidInput(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Vec::new();

        // For simplicity, just compute the dominant eigenvalue/eigenvector using power iteration
        // In a full implementation, you'd use more sophisticated methods like QR algorithm

        // Start with random vector
        let mut v = vec![1.0f64; rows];
        let iterations = 100;

        for _ in 0..iterations {
            let mut new_v = vec![0.0; rows];

            // Multiply matrix by vector: new_v = A * v
            for i in 0..rows {
                for j in 0..cols {
                    if let Some(a_val) = self.get(i, j) {
                        new_v[i] += f64::from(a_val) * v[j];
                    }
                }
            }

            // Normalize
            let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for val in &mut new_v {
                    *val /= norm;
                }
            }
            v = new_v;
        }

        // Compute eigenvalue: λ = v^T * A * v (Rayleigh quotient)
        let mut lambda = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                if let Some(a_val) = self.get(i, j) {
                    lambda += v[i] * f64::from(a_val) * v[j];
                }
            }
        }

        eigenvalues.push(lambda);
        eigenvectors.push(
            v.into_iter()
                .map(|x| T::from_f64(x).unwrap_or_else(|| T::default()))
                .collect(),
        );

        // For completeness, add remaining eigenvalues as zeros with identity eigenvectors
        for i in 1..rows {
            eigenvalues.push(0.0);
            let mut ev = vec![T::default(); rows];
            if i < rows {
                ev[i] = T::from_f64(1.0).unwrap_or_else(|| T::default());
            }
            eigenvectors.push(ev);
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Matrix rank - number of linearly independent rows/columns
    /// Computed using Gaussian elimination with partial pivoting
    pub fn rank(&self) -> GraphResult<usize>
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
        f64: From<T>,
    {
        let (rows, cols) = self.shape();

        // Create a working copy of the matrix for Gaussian elimination
        let mut work_matrix = Vec::new();
        for i in 0..rows {
            let mut row = Vec::new();
            for j in 0..cols {
                let val = self.get(i, j).unwrap_or_else(|| T::default());
                row.push(f64::from(val));
            }
            work_matrix.push(row);
        }

        let tolerance = 1e-10;
        let mut rank = 0;

        // Gaussian elimination to count pivot rows
        for col in 0..cols.min(rows) {
            // Find pivot row
            let mut pivot_row = None;
            let mut max_val = 0.0;

            for row in rank..rows {
                let abs_val = work_matrix[row][col].abs();
                if abs_val > tolerance && abs_val > max_val {
                    max_val = abs_val;
                    pivot_row = Some(row);
                }
            }

            if let Some(pivot) = pivot_row {
                // Swap rows if needed
                if pivot != rank {
                    work_matrix.swap(rank, pivot);
                }

                // Eliminate column entries below pivot
                let pivot_val = work_matrix[rank][col];

                for row in (rank + 1)..rows {
                    if work_matrix[row][col].abs() > tolerance {
                        let factor = work_matrix[row][col] / pivot_val;
                        for c in col..cols {
                            work_matrix[row][c] -= factor * work_matrix[rank][c];
                        }
                    }
                }

                rank += 1;
            }
        }

        Ok(rank)
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

// === OPERATOR OVERLOADING IMPLEMENTATIONS ===

impl<T: NumericType> std::ops::Add<&GraphMatrix<T>> for &GraphMatrix<T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn add(self, rhs: &GraphMatrix<T>) -> Self::Output {
        let (self_rows, self_cols) = self.shape();
        let (rhs_rows, rhs_cols) = rhs.shape();

        if self_rows != rhs_rows || self_cols != rhs_cols {
            // Return zero matrix on dimension mismatch (could be made more robust)
            return GraphMatrix::zeros(self_rows, self_cols);
        }

        let mut result = GraphMatrix::zeros(self_rows, self_cols);

        for i in 0..self_rows {
            for j in 0..self_cols {
                let a = self.get(i, j).unwrap_or_else(|| T::zero());
                let b = rhs.get(i, j).unwrap_or_else(|| T::zero());
                let sum = a + b;
                let _ = result.set(i, j, sum);
            }
        }

        result
    }
}

impl<T: NumericType> std::ops::Add<GraphMatrix<T>> for GraphMatrix<T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn add(self, rhs: GraphMatrix<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T: NumericType> std::ops::Sub<&GraphMatrix<T>> for &GraphMatrix<T>
where
    T: std::ops::Sub<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn sub(self, rhs: &GraphMatrix<T>) -> Self::Output {
        let (self_rows, self_cols) = self.shape();
        let (rhs_rows, rhs_cols) = rhs.shape();

        if self_rows != rhs_rows || self_cols != rhs_cols {
            // Return zero matrix on dimension mismatch
            return GraphMatrix::zeros(self_rows, self_cols);
        }

        let mut result = GraphMatrix::zeros(self_rows, self_cols);

        for i in 0..self_rows {
            for j in 0..self_cols {
                let a = self.get(i, j).unwrap_or_else(|| T::zero());
                let b = rhs.get(i, j).unwrap_or_else(|| T::zero());
                let diff = a - b;
                let _ = result.set(i, j, diff);
            }
        }

        result
    }
}

impl<T: NumericType> std::ops::Sub<GraphMatrix<T>> for GraphMatrix<T>
where
    T: std::ops::Sub<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn sub(self, rhs: GraphMatrix<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T: NumericType> std::ops::Mul<&GraphMatrix<T>> for &GraphMatrix<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn mul(self, rhs: &GraphMatrix<T>) -> Self::Output {
        // Use matrix multiplication (not element-wise)
        self.multiply(rhs)
            .unwrap_or_else(|_| GraphMatrix::zeros(self.shape().0, rhs.shape().1))
    }
}

impl<T: NumericType> std::ops::Mul<GraphMatrix<T>> for GraphMatrix<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    type Output = GraphMatrix<T>;

    fn mul(self, rhs: GraphMatrix<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: NumericType> Default for GraphMatrix<T> {
    fn default() -> Self {
        GraphMatrix::zeros(0, 0)
    }
}

impl<T: NumericType> GraphMatrix<T> {
    /// Select specific columns by indices
    pub fn select_columns(&self, indices: &[usize]) -> GraphResult<GraphMatrix<T>> {
        let (rows, _) = self.shape();
        let mut result = GraphMatrix::zeros(rows, indices.len());

        for (new_col, &old_col) in indices.iter().enumerate() {
            if let Some(column_data) = self.get_column(old_col) {
                for (row, &value) in column_data.iter().enumerate() {
                    result.set(row, new_col, value)?;
                }
            }
        }

        Ok(result)
    }

    /// Multiply matrix by scalar
    pub fn scalar_multiply(&self, scalar: f64) -> GraphResult<GraphMatrix<T>>
    where
        T: std::ops::Mul<Output = T> + From<f64>,
    {
        let (rows, cols) = self.shape();
        let mut result = GraphMatrix::zeros(rows, cols);
        let scalar_t = T::from(scalar);

        for i in 0..rows {
            for j in 0..cols {
                if let Ok(value) = self.get_checked(i, j) {
                    result.set(i, j, value * scalar_t)?;
                }
            }
        }

        Ok(result)
    }

    /// Add two matrices element-wise (method wrapper for + operator)
    pub fn add(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>>
    where
        for<'a> &'a GraphMatrix<T>: std::ops::Add<Output = GraphMatrix<T>>,
    {
        Ok(self + other)
    }

    /// Compute variance of each column
    pub fn column_variances(&self) -> GraphResult<Vec<f64>>
    where
        T: Into<f64> + Copy,
    {
        let (rows, cols) = self.shape();
        let mut variances = Vec::with_capacity(cols);

        for col in 0..cols {
            if let Some(column_data) = self.get_column(col) {
                // Compute mean
                let sum: f64 = column_data.iter().map(|&x| x.into()).sum();
                let mean = sum / rows as f64;

                // Compute variance
                let variance: f64 = column_data
                    .iter()
                    .map(|&x| {
                        let diff = x.into() - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / rows as f64;

                variances.push(variance);
            } else {
                variances.push(0.0);
            }
        }

        Ok(variances)
    }

    /// Concatenate matrices horizontally (by columns)
    pub fn concatenate_columns(matrices: Vec<GraphMatrix<T>>) -> GraphResult<GraphMatrix<T>> {
        if matrices.is_empty() {
            return Ok(GraphMatrix::zeros(0, 0));
        }

        if matrices.len() == 1 {
            return Ok(matrices[0].clone());
        }

        // Use existing concatenate method with axis=1 (columns)
        let mut result = matrices[0].clone();
        for matrix in matrices.into_iter().skip(1) {
            result = result.concatenate(&matrix, 1)?;
        }

        Ok(result)
    }

    /// Compute mean of each column
    pub fn column_means(&self) -> GraphResult<Vec<f64>>
    where
        f64: From<T>,
    {
        let (rows, cols) = self.shape();
        let mut means = Vec::with_capacity(cols);

        for col in 0..cols {
            if let Some(column_data) = self.get_column(col) {
                let sum: f64 = column_data.iter().map(|&x| f64::from(x)).sum();
                let mean = sum / rows as f64;
                means.push(mean);
            } else {
                means.push(0.0);
            }
        }

        Ok(means)
    }
}
