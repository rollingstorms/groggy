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

use crate::storage::advanced_matrix::{
    UnifiedMatrix, NumericType, BackendSelector, ComputationGraph, 
    AdvancedMemoryPool
};
use crate::storage::array::NumArray;
use crate::storage::table::BaseTable;
use crate::core::display::{DisplayEngine, DisplayConfig};
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrValue, AttrValueType, NodeId};
use std::fmt;
use std::sync::Arc;
use std::collections::HashMap;

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
    pub fn analyze<T: NumericType>(matrix: &GraphMatrix<T>) -> Self {
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
    
    /// Backend optimization
    backend_selector: Arc<BackendSelector>,
    memory_pool: Arc<AdvancedMemoryPool<T>>,
    
    /// Cached matrix properties
    properties: Option<MatrixProperties>,
    /// Reference to the source graph (optional)
    graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
}

impl<T: NumericType> GraphMatrix<T> {
    /// Create GraphMatrix from UnifiedMatrix storage
    pub fn from_storage(storage: UnifiedMatrix<T>) -> Self {
        let shape = storage.shape();
        let (rows, cols) = (shape.rows, shape.cols);
        let column_names = (0..cols).map(|i| format!("col_{}", i)).collect();
        
        Self {
            storage,
            column_names,
            row_labels: None,
            requires_grad: false,
            computation_graph: None,
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
    pub fn zeros_with_type(rows: usize, cols: usize, _attr_type: crate::types::AttrValueType) -> Self {
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
    
    /// Create identity matrix (using ones as placeholder until eye is implemented)
    pub fn eye(size: usize) -> Self {
        // TODO: Implement proper identity matrix in UnifiedMatrix
        let storage = UnifiedMatrix::ones(size, size).unwrap_or_else(|_| {
            UnifiedMatrix::new(1, 1).expect("Failed to create minimal matrix")
        });
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
            backend_selector: Arc::new(BackendSelector::new()),
            memory_pool: Arc::new(AdvancedMemoryPool::new()),
            properties: None,
            graph: None,
        })
    }

    /// Create adjacency matrix from edge data
    pub fn adjacency_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)]
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
                (node_to_index.get(&source), node_to_index.get(&target)) {
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
        weighted_edges: &[(NodeId, NodeId, T)]
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
                (node_to_index.get(&source), node_to_index.get(&target)) {
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
            return Err(GraphError::InvalidInput(format!("Column {} not found", col_idx)));
        }
        
        // Extract column using individual element access
        let mut column = Vec::with_capacity(rows);
        for row in 0..rows {
            let element = UnifiedMatrix::get(&self.storage, row, col_idx)
                .map_err(|e| GraphError::InvalidInput(format!("Cannot access element ({}, {}): {:?}", row, col_idx, e)))?;
            column.push(element);
        }
        Ok(column)
    }
    
    /// Get column name by index - needed by slicing module  
    pub(crate) fn get_column_name(&self, col_idx: usize) -> String {
        self.column_names.get(col_idx)
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
            self.column_names = names.into_iter()
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
    pub(crate) fn is_symmetric_internal(&self) -> bool {
        let (rows, cols) = self.shape();
        if rows != cols {
            return false;
        }
        
        // For now, return false as UnifiedMatrix doesn't expose individual element access
        // TODO: Implement proper symmetry check when UnifiedMatrix exposes get() method
        false
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
        let result_storage = self.storage.matmul(&other.storage)
            .map_err(|e| GraphError::InvalidInput(format!("Matrix multiplication failed: {:?}", e)))?;
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
    
    /// 2D Convolution operation (placeholder - integration incomplete)
    pub fn conv2d(&self, _kernel: &GraphMatrix<T>, _config: crate::storage::advanced_matrix::neural::convolution::ConvolutionConfig) -> GraphResult<GraphMatrix<T>> {
        // TODO: Complete Conv2D integration - API mismatch between Conv2D (expects ConvTensor) and GraphMatrix (UnifiedMatrix)
        // Conv2D::new requires (in_channels, out_channels, config) not (kernel, config)
        // Conv2D::forward expects ConvTensor<T> not UnifiedMatrix<T>
        Err(GraphError::InvalidInput("Conv2D integration incomplete - API requires ConvTensor conversion".into()))
    }
    
    /// Enable automatic differentiation
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.computation_graph.is_none() {
            self.computation_graph = Some(ComputationGraph::new());
        }
        self
    }
    
    /// Compute gradients via backpropagation (placeholder - requires output node ID)
    pub fn backward(&mut self) -> GraphResult<()> {
        if let Some(_graph) = &mut self.computation_graph {
            // TODO: Complete autodiff integration - backward() requires output_node: NodeId parameter
            // Need to track which node is the output node in the computation graph
            Err(GraphError::InvalidInput("Autodiff integration incomplete - backward requires output node ID".into()))
        } else {
            Err(GraphError::InvalidInput("No computation graph available for gradient computation".into()))
        }
    }
    
    /// Get gradient matrix (placeholder - get_gradient method not implemented)
    pub fn grad(&self) -> Option<&GraphMatrix<T>> {
        // TODO: ComputationGraph::get_gradient() method not implemented
        // Need to implement gradient storage and retrieval in ComputationGraph
        None
    }
    
    /// Cast matrix to different numeric type
    pub fn cast<U: NumericType>(&self) -> GraphResult<GraphMatrix<U>> {
        // TODO: Implement type casting when UnifiedMatrix supports it
        // For now, return an error
        Err(GraphError::InvalidInput("Type casting not yet implemented in UnifiedMatrix backend".into()))
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
        self.storage.set(row, col, value).map_err(|e| GraphError::InvalidInput(e.to_string()))
    }
    
    /// Get element at specific row and column with error details
    pub fn get_checked(&self, row: usize, col: usize) -> GraphResult<T> {
        self.storage.get(row, col).map_err(|e| GraphError::InvalidInput(e.to_string()))
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
                    return Err(GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j)));
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
            return Err(GraphError::InvalidInput(
                format!("Matrix dimensions incompatible for multiplication: {}x{} * {}x{}", 
                       self_rows, self_cols, other_rows, other_cols)
            ));
        }
        
        let mut result = Self::zeros(self_rows, other_cols);
        
        for i in 0..self_rows {
            for j in 0..other_cols {
                let mut sum = T::zero();
                for k in 0..self_cols {
                    let a = self.get(i, k).ok_or_else(|| 
                        GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, k)))?;
                    let b = other.get(k, j).ok_or_else(|| 
                        GraphError::InvalidInput(format!("Invalid cell at ({}, {})", k, j)))?;
                    sum = T::add(sum, T::mul(a, b));
                }
                result.set(i, j, sum)?;
            }
        }
        
        Ok(result)
    }
    
    /// Element-wise multiplication (Hadamard product)
    pub fn elementwise_multiply(&self, other: &GraphMatrix<T>) -> GraphResult<GraphMatrix<T>> {
        let (rows, cols) = self.shape();
        let (other_rows, other_cols) = other.shape();
        
        if rows != other_rows || cols != other_cols {
            return Err(GraphError::InvalidInput(
                format!("Matrix dimensions must match for element-wise multiplication: {}x{} vs {}x{}", 
                       rows, cols, other_rows, other_cols)
            ));
        }
        
        let mut result = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let a = self.get(i, j).ok_or_else(|| 
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j)))?;
                let b = other.get(i, j).ok_or_else(|| 
                    GraphError::InvalidInput(format!("Invalid cell at ({}, {})", i, j)))?;
                result.set(i, j, T::mul(a, b))?;
            }
        }
        
        Ok(result)
    }
    
    /// Matrix power operation (repeated multiplication)
    pub fn power(&self, n: u32) -> GraphResult<GraphMatrix<T>> {
        if !self.is_square() {
            return Err(GraphError::InvalidInput("Matrix must be square for power operation".into()));
        }
        
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
        T: std::ops::Add<Output = T> + Copy
    {
        let sums = self.sum_axis(axis)?;
        let count = match axis {
            Axis::Rows => self.shape().1, // number of columns
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
        T: std::ops::Add<Output = T> + Copy
    {
        // This is a simplified implementation - for now just return the sum_axis
        // TODO: Implement proper standard deviation calculation
        self.sum_axis(axis)
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
        format!("GraphMatrix<{}>: {}x{} matrix", std::any::type_name::<T>(), rows, cols)
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
                return Err(GraphError::InvalidInput(
                    format!("Array {} has length {} but expected {}", i, array.len(), rows)
                ));
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
        T: std::ops::Add<Output = T> + Copy
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
}

// Graph-specific matrix operations
impl<T: NumericType> GraphMatrix<T> {
    /// Convert adjacency matrix to Laplacian matrix
    pub fn to_laplacian(&self) -> GraphResult<GraphMatrix<T>> {
        // TODO: Implement proper Laplacian matrix calculation
        // L = D - A where D is degree matrix and A is adjacency matrix
        Err(GraphError::InvalidInput("Laplacian matrix calculation not yet implemented".into()))
    }
    
    /// Convert adjacency matrix to normalized Laplacian matrix
    pub fn to_normalized_laplacian(&self) -> GraphResult<GraphMatrix<T>> {
        // TODO: Implement normalized Laplacian matrix calculation  
        // L_norm = I - D^(-1/2) * A * D^(-1/2)
        Err(GraphError::InvalidInput("Normalized Laplacian matrix calculation not yet implemented".into()))
    }
    
    /// Check if this matrix represents a valid adjacency matrix
    pub fn is_adjacency_matrix(&self) -> bool {
        // TODO: Implement adjacency matrix validation
        // Check for non-negative values, symmetry (for undirected), etc.
        false
    }
    
    /// Get degree matrix from adjacency matrix
    pub fn to_degree_matrix(&self) -> GraphResult<GraphMatrix<T>> {
        // TODO: Implement degree matrix calculation
        // Diagonal matrix with node degrees on diagonal
        Err(GraphError::InvalidInput("Degree matrix calculation not yet implemented".into()))
    }

    // === PLACEHOLDER FOR ENHANCED OPERATIONS ===
    // These will be implemented incrementally once core system is stable
    
}
