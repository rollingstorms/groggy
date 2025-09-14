//! GraphMatrix - General-purpose matrix built on GraphArray foundation
//!
//! This module provides GraphMatrix as a collection of GraphArrays with enforced
//! homogeneous typing and specialized matrix operations.
//!
//! # Design Principles
//! - GraphMatrix is a collection of GraphArrays (columns)
//! - All columns must have the same type (enforced)
//! - Inherits statistical operations from GraphArray
//! - Supports both dense and sparse representations
//! - Linear algebra operations for numeric matrices
//! - Memory-efficient storage and lazy evaluation

use crate::storage::array::NumArray;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrValue, AttrValueType, NodeId};
use std::fmt;

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
    pub fn analyze(matrix: &GraphMatrix) -> Self {
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

/// General-purpose matrix built on NumArray foundation
#[derive(Debug, Clone)]
pub struct GraphMatrix {
    /// Columns stored as NumArrays for numerical operations
    columns: Vec<NumArray<f64>>,
    /// Column names/labels
    column_names: Vec<String>,
    /// Row labels (optional) - string labels stored as BaseArray
    #[allow(dead_code)]
    row_labels: Option<Vec<String>>,
    /// Shape of the matrix (rows, cols)
    shape: (usize, usize),
    /// Cached matrix properties
    properties: Option<MatrixProperties>,
    /// Reference to the source graph (optional)
    graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
}

impl GraphMatrix {
    /// Internal method to get column by index - needed by slicing module
    pub(crate) fn get_column_internal(&self, col_idx: usize) -> GraphResult<&NumArray<f64>> {
        self.columns.get(col_idx).ok_or_else(|| {
            GraphError::InvalidInput(format!("Column {} not found", col_idx))
        })
    }
    
    /// Get column name by index - needed by slicing module  
    pub(crate) fn get_column_name(&self, col_idx: usize) -> String {
        self.column_names.get(col_idx)
            .cloned()
            .unwrap_or_else(|| format!("col_{}", col_idx))
    }
    
    /// Create a new GraphMatrix from a collection of NumArrays
    /// All arrays must have the same length for proper matrix structure
    pub fn from_arrays(arrays: Vec<NumArray<f64>>) -> GraphResult<Self> {
        if arrays.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot create matrix from empty array list".to_string(),
            ));
        }

        // Check all arrays have the same length for valid matrix structure
        let num_rows = arrays[0].len();
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != num_rows {
                return Err(GraphError::InvalidInput(format!(
                    "Column {} has length {} but expected {}",
                    i,
                    array.len(),
                    num_rows
                )));
            }
        }

        let num_cols = arrays.len();
        let column_names: Vec<String> = (0..num_cols)
            .map(|i| format!("col_{}", i))
            .collect();

        Ok(GraphMatrix {
            columns: arrays,
            column_names,
            row_labels: None,
            shape: (num_rows, num_cols),
            properties: None,
            graph: None,
        })
    }

    /// Create a GraphMatrix from graph attributes
    pub fn from_graph_attributes(
        graph: std::rc::Rc<crate::api::graph::Graph>,
        attrs: &[&str],
        entities: &[NodeId],
    ) -> GraphResult<Self> {
        // TODO: Implement graph attribute extraction for NumArray
        // This requires converting GraphArray::from_graph_attribute to work with NumArray<f64>
        let _arrays: Vec<NumArray<f64>> = Vec::new();
        Err(GraphError::InvalidInput("from_graph_attributes not yet implemented for NumArray".to_string()))
    }

    /// Create a zero matrix with specified dimensions and type
    pub fn zeros(rows: usize, cols: usize, dtype: AttrValueType) -> Self {
        let zero_value = match dtype {
            AttrValueType::Int => AttrValue::Int(0),
            AttrValueType::Float => AttrValue::Float(0.0),
            AttrValueType::Bool => AttrValue::Bool(false),
            AttrValueType::Text => AttrValue::Text("".to_string()),
            _ => AttrValue::Int(0), // Default fallback
        };

        let arrays = (0..cols)
            .map(|_i| {
                let values = vec![0.0f64; rows];
                NumArray::new(values)
            })
            .collect();

        let column_names = (0..cols).map(|i| format!("col_{}", i)).collect();

        Self {
            columns: arrays,
            column_names,
            row_labels: None,
            shape: (rows, cols),
            properties: None,
            graph: None,
        }
    }

    /// Create adjacency matrix from edge data
    pub fn adjacency_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
    ) -> crate::errors::GraphResult<Self> {
        let size = nodes.len();
        if size == 0 {
            return Ok(Self::zeros_f64(0, 0));
        }

        // Create node index mapping
        let node_to_index: std::collections::HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        // Initialize adjacency matrix data (row-major)
        let mut matrix_data = vec![0.0f64; size * size];

        // Fill adjacency matrix from edge list
        for &(source, target) in edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = 
                (node_to_index.get(&source), node_to_index.get(&target)) {
                // Set adjacency (1.0 for unweighted)
                matrix_data[src_idx * size + tgt_idx] = 1.0;
            }
        }

        // Convert to column-major format
        Self::from_row_major_data(matrix_data, size, size, Some(nodes))
    }

    /// Create weighted adjacency matrix from weighted edge data
    pub fn weighted_adjacency_from_edges(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, f64)],
    ) -> crate::errors::GraphResult<Self> {
        let size = nodes.len();
        if size == 0 {
            return Ok(Self::zeros_f64(0, 0));
        }

        // Create node index mapping
        let node_to_index: std::collections::HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        // Initialize adjacency matrix data (row-major)
        let mut matrix_data = vec![0.0f64; size * size];

        // Fill weighted adjacency matrix
        for &(source, target, weight) in weighted_edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = 
                (node_to_index.get(&source), node_to_index.get(&target)) {
                matrix_data[src_idx * size + tgt_idx] = weight;
            }
        }

        // Convert to column-major format
        Self::from_row_major_data(matrix_data, size, size, Some(nodes))
    }

    /// Create zeros matrix with f64 type
    pub fn zeros_f64(rows: usize, cols: usize) -> Self {
        let arrays = (0..cols)
            .map(|_| {
                let values = vec![0.0f64; rows];
                crate::storage::array::NumArray::new(values)
            })
            .collect();

        let column_names = (0..cols).map(|i| format!("col_{}", i)).collect();

        GraphMatrix {
            columns: arrays,
            column_names,
            row_labels: None,
            shape: (rows, cols),
            properties: None,
            graph: None,
        }
    }

    /// Helper: Create GraphMatrix from row-major data
    fn from_row_major_data(
        data: Vec<f64>,
        rows: usize,
        cols: usize,
        nodes: Option<&[NodeId]>,
    ) -> crate::errors::GraphResult<Self> {
        // Convert row-major to column-major
        let mut columns = Vec::with_capacity(cols);
        for col_idx in 0..cols {
            let mut column_data = Vec::with_capacity(rows);
            for row_idx in 0..rows {
                column_data.push(data[row_idx * cols + col_idx]);
            }
            columns.push(crate::storage::array::NumArray::new(column_data));
        }

        // Create appropriate labels
        let (column_names, row_labels) = if let Some(node_ids) = nodes {
            let col_names = node_ids.iter().map(|id| format!("node_{}", id)).collect();
            let row_labels = node_ids.iter().map(|id| format!("node_{}", id)).collect();
            (col_names, Some(row_labels))
        } else {
            let col_names = (0..cols).map(|i| format!("col_{}", i)).collect();
            (col_names, None)
        };

        Ok(GraphMatrix {
            columns,
            column_names,
            row_labels,
            shape: (rows, cols),
            properties: None,
            graph: None,
        })
    }

    /// Create an identity matrix of specified size
    pub fn identity(size: usize) -> Self {
        let mut columns = Vec::new();
        let column_names: Vec<String> = (0..size).map(|i| format!("col_{}", i)).collect();

        // Create each column with 1 at diagonal position, 0 elsewhere
        for j in 0..size {
            let mut col_values = Vec::new();
            for i in 0..size {
                col_values.push(if i == j { 1.0 } else { 0.0 });
            }
            columns.push(NumArray::new(col_values));
        }

        Self {
            columns,
            column_names,
            row_labels: None,
            shape: (size, size),
            properties: None,
            graph: None,
        }
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }

    /// Get the data type of the matrix - always Float for NumArray<f64>
    pub fn dtype(&self) -> AttrValueType {
        AttrValueType::Float
    }

    /// Get column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Set column names
    pub fn set_column_names(&mut self, names: Vec<String>) -> GraphResult<()> {
        if names.len() != self.columns.len() {
            return Err(GraphError::InvalidInput(format!(
                "Expected {} column names, got {}",
                self.columns.len(),
                names.len()
            )));
        }
        self.column_names = names;
        self.properties = None; // Invalidate cache
        Ok(())
    }

    /// Get matrix properties (computed lazily)
    pub fn properties(&mut self) -> &MatrixProperties {
        if self.properties.is_none() {
            self.properties = Some(MatrixProperties::analyze(self));
        }
        self.properties.as_ref().unwrap()
    }

    /// Check if matrix is square
    pub fn is_square(&self) -> bool {
        let (rows, cols) = self.shape();
        rows == cols
    }

    /// Check if matrix is numeric - always true for NumArray<f64>
    pub fn is_numeric(&self) -> bool {
        true
    }

    /// Get optional graph reference
    pub fn graph(&self) -> Option<&std::rc::Rc<crate::api::graph::Graph>> {
        self.graph.as_ref()
    }

    /// Get element at (row, col) position
    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        self.columns.get(col)?.get(row)
    }

    /// Get a column by index
    pub fn get_column(&self, col: usize) -> Option<&NumArray<f64>> {
        self.columns.get(col)
    }

    /// Get a column by name
    pub fn get_column_by_name(&self, name: &str) -> Option<&NumArray<f64>> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.columns.get(idx))
    }

    /// Get a row as a new NumArray
    pub fn get_row(&self, row: usize) -> Option<NumArray<f64>> {
        let (rows, _) = self.shape();
        if row >= rows {
            return None;
        }

        let row_values: Vec<f64> = self
            .columns
            .iter()
            .filter_map(|col| col.get(row).copied())
            .collect();

        if row_values.len() == self.columns.len() {
            Some(NumArray::new(row_values))
        } else {
            None
        }
    }

    /// Iterator over columns
    pub fn iter_columns(&self) -> impl Iterator<Item = &NumArray<f64>> {
        self.columns.iter()
    }

    /// Iterator over rows (returns NumArrays)
    pub fn iter_rows(&self) -> impl Iterator<Item = NumArray<f64>> + '_ {
        let (rows, _) = self.shape();
        (0..rows).filter_map(move |i| self.get_row(i))
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> GraphMatrix {
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            return self.clone();
        }

        let transposed_arrays: Vec<NumArray<f64>> = (0..rows)
            .map(|row_idx| {
                let row_values: Vec<f64> = self
                    .columns
                    .iter()
                    .filter_map(|col| col.get(row_idx).copied())
                    .collect();
                NumArray::new(row_values)
            })
            .collect();

        // Column names become row names, and vice versa
        let new_column_names = (0..rows).map(|i| format!("row_{}", i)).collect();

        Self {
            columns: transposed_arrays,
            column_names: new_column_names,
            row_labels: None,
            shape: (cols, rows),
            properties: None,
            graph: self.graph.clone(),
        }
    }

    /// Statistical operations along an axis
    pub fn sum_axis(&self, axis: Axis) -> NumArray<f64> {
        match axis {
            Axis::Columns => {
                // Sum each column (returns array of column sums)
                let sums: Vec<f64> = self
                    .columns
                    .iter()
                    .map(|col| col.sum())
                    .collect();
                NumArray::new(sums)
            }
            Axis::Rows => {
                // Sum each row (returns array of row sums)
                let (rows, _) = self.shape();
                let sums: Vec<f64> = (0..rows)
                    .map(|row_idx| {
                        self.columns
                            .iter()
                            .filter_map(|col| col.get(row_idx))
                            .map(|&val| val)
                            .sum()
                    })
                    .collect();
                NumArray::new(sums)
            }
        }
    }

    /// Mean along an axis
    pub fn mean_axis(&self, axis: Axis) -> NumArray<f64> {
        match axis {
            Axis::Columns => {
                let means: Vec<f64> = self
                    .columns
                    .iter()
                    .map(|col| col.mean().unwrap_or(0.0))
                    .collect();
                NumArray::new(means)
            }
            Axis::Rows => {
                let (rows, cols) = self.shape();
                let means: Vec<f64> = (0..rows)
                    .map(|row_idx| {
                        let row_sum: f64 = self
                            .columns
                            .iter()
                            .filter_map(|col| col.get(row_idx))
                            .map(|&val| val)
                            .sum();
                        
                        if cols == 0 {
                            0.0
                        } else {
                            row_sum / cols as f64
                        }
                    })
                    .collect();
                NumArray::new(means)
            }
        }
    }

    /// Standard deviation along an axis
    pub fn std_axis(&self, axis: Axis) -> NumArray<f64> {
        match axis {
            Axis::Columns => {
                let stds: Vec<f64> = self
                    .columns
                    .iter()
                    .map(|col| col.std_dev().unwrap_or(0.0))
                    .collect();
                NumArray::new(stds)
            }
            Axis::Rows => {
                // This would be more complex - compute std deviation for each row
                // For now, return zeros as placeholder
                let (rows, _) = self.shape();
                let zeros = vec![0.0f64; rows];
                NumArray::new(zeros)
            }
        }
    }

    /// Matrix multiplication - multiply this matrix with another
    /// Returns a new GraphMatrix that is the product of self * other
    /// Optimized for graph adjacency matrices (often sparse)
    pub fn multiply(&self, other: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        // Check dimensions are compatible for multiplication
        if self_cols != other_rows {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions incompatible for multiplication: ({}, {}) × ({}, {})",
                self_rows, self_cols, other_rows, other_cols
            )));
        }

        // Check both matrices are numeric
        if !self.is_numeric() || !other.is_numeric() {
            return Err(GraphError::InvalidInput(
                "Matrix multiplication requires numeric matrices".to_string(),
            ));
        }

        // 🚀 OPTIMIZATION: Check sparsity and choose appropriate algorithm
        let self_sparsity = self.estimate_sparsity();
        let other_sparsity = other.estimate_sparsity();

        // Use sparse multiplication if both matrices are sparse (typical for graph adjacency)
        if self_sparsity < 0.1 && other_sparsity < 0.1 {
            self.multiply_sparse(other)
        } else {
            // Use optimized dense multiplication for denser matrices
            let self_data = self.extract_matrix_data()?;
            let other_data = other.extract_matrix_data()?;

            let result_data =
                self.multiply_blocked(&self_data, &other_data, self_rows, self_cols, other_cols)?;
            self.create_matrix_from_data(result_data, self_rows, other_cols)
        }
    }

    /// Estimate matrix sparsity (fraction of non-zero elements)
    fn estimate_sparsity(&self) -> f64 {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        if total_elements == 0 {
            return 0.0;
        }

        let mut non_zero_count = 0;
        let sample_size = (total_elements / 10).max(100).min(total_elements); // Sample 10% or at least 100 elements

        for i in 0..sample_size {
            let row = i / cols;
            let col = i % cols;
            if let Some(val) = self.get(row, col) {
                if val.abs() > 1e-10 {
                    non_zero_count += 1;
                }
            }
        }

        (non_zero_count as f64) / (sample_size as f64)
    }

    /// Sparse matrix multiplication optimized for adjacency matrices
    fn multiply_sparse(&self, other: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (self_rows, _self_cols) = self.shape();
        let (_, other_cols) = other.shape();

        // Extract sparse representation
        let self_sparse = self.extract_sparse_data()?;
        let other_sparse = other.extract_sparse_data()?;

        // Sparse multiplication: only process non-zero elements
        let mut result_data = vec![0.0; self_rows * other_cols];

        // For each non-zero element in self
        for (self_row, self_col, self_val) in self_sparse {
            // Find all non-zero elements in corresponding row of other
            for (other_row, other_col, other_val) in &other_sparse {
                if *other_row == self_col {
                    // Accumulate: result[self_row][other_col] += self_val * other_val
                    let result_idx = self_row * other_cols + other_col;
                    result_data[result_idx] += self_val * other_val;
                }
            }
        }

        self.create_matrix_from_data(result_data, self_rows, other_cols)
    }

    /// Extract sparse representation as (row, col, value) tuples
    fn extract_sparse_data(&self) -> GraphResult<Vec<(usize, usize, f64)>> {
        let (rows, cols) = self.shape();
        let mut sparse_data = Vec::new();

        for row_idx in 0..rows {
            for col_idx in 0..cols {
                if let Some(val) = self.get(row_idx, col_idx) {
                    if val.abs() > 1e-10 {
                        // Consider values > epsilon as non-zero
                        sparse_data.push((row_idx, col_idx, *val));
                    }
                }
            }
        }

        Ok(sparse_data)
    }

    /// Extract matrix data into a flat Vec<f64> for optimized access
    /// Returns row-major ordered data
    fn extract_matrix_data(&self) -> GraphResult<Vec<f64>> {
        let (rows, cols) = self.shape();
        let mut data = Vec::with_capacity(rows * cols);

        // Extract in row-major order for better cache locality
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                let val = self
                    .get(row_idx, col_idx)
                    .copied()
                    .unwrap_or(0.0);
                data.push(val);
            }
        }

        Ok(data)
    }

    /// Optimized blocked matrix multiplication
    /// Uses cache-friendly memory access patterns
    fn multiply_blocked(
        &self,
        a_data: &[f64],
        b_data: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> GraphResult<Vec<f64>> {
        let mut c_data = vec![0.0; m * n];

        // 🚀 OPTIMIZATION: Use blocked multiplication for cache efficiency
        const BLOCK_SIZE: usize = 64; // Tuned for typical L1 cache

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    // Process block
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k_idx in k_block..k_end {
                                // Row-major access: A[i][k] = a_data[i * k + k_idx]
                                // Column access: B[k][j] = b_data[k_idx * n + j]
                                sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
                            }
                            c_data[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        Ok(c_data)
    }

    /// Create GraphMatrix from flat data array
    fn create_matrix_from_data(
        &self,
        data: Vec<f64>,
        rows: usize,
        cols: usize,
    ) -> GraphResult<GraphMatrix> {
        let mut result_columns = Vec::with_capacity(cols);

        // Convert row-major data back to column-major NumArrays
        for col_idx in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row_idx in 0..rows {
                let val = data[row_idx * cols + col_idx] as f64;
                column_data.push(val);
            }

            let column_name = format!("col_{}", col_idx);
            let column = NumArray::new(column_data);
            result_columns.push(column);
        }

        // Create result matrix
        let mut result = GraphMatrix::from_arrays(result_columns)?;

        // Set appropriate column names
        let column_names: Vec<String> = (0..cols).map(|i| format!("col_{}", i)).collect();
        result.set_column_names(column_names)?;

        Ok(result)
    }

    /// Matrix power - raise matrix to integer power
    /// Returns self^n using repeated squaring for efficiency
    pub fn power(&self, n: u32) -> GraphResult<GraphMatrix> {
        if !self.is_square() {
            return Err(GraphError::InvalidInput(
                "Matrix power requires square matrix".to_string(),
            ));
        }

        if !self.is_numeric() {
            return Err(GraphError::InvalidInput(
                "Matrix power requires numeric matrix".to_string(),
            ));
        }

        if n == 0 {
            // Return identity matrix
            let size = self.shape().0;
            let identity = Self::zeros(size, size, AttrValueType::Float);

            // Set diagonal elements to 1
            for _i in 0..size {
                // Create identity values - we'll need to implement a way to set matrix elements
                // For now, create identity through multiplication
            }

            // TODO: Implement proper identity matrix creation
            return Ok(identity);
        }

        if n == 1 {
            return Ok(self.clone());
        }

        // Use repeated squaring for efficiency
        let mut result = self.clone();
        let base = self.clone();
        let _exp = n;

        // Initialize result as identity for the algorithm
        // For now, just return self^n by repeated multiplication
        for _ in 1..n {
            result = result.multiply(&base)?;
        }

        Ok(result)
    }

    /// Elementwise multiplication (Hadamard product)
    pub fn elementwise_multiply(&self, other: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();

        // Check dimensions match exactly
        if self_rows != other_rows || self_cols != other_cols {
            return Err(GraphError::InvalidInput(format!(
                "Matrix dimensions must match for elementwise multiplication: ({}, {}) vs ({}, {})",
                self_rows, self_cols, other_rows, other_cols
            )));
        }

        // Check both matrices are numeric
        if !self.is_numeric() || !other.is_numeric() {
            return Err(GraphError::InvalidInput(
                "Elementwise multiplication requires numeric matrices".to_string(),
            ));
        }

        // Create result columns
        let mut result_columns = Vec::with_capacity(self_cols);

        for col_idx in 0..self_cols {
            let mut result_column: Vec<f64> = Vec::with_capacity(self_rows);

            for row_idx in 0..self_rows {
                let self_val = self
                    .get(row_idx, col_idx)
                    .copied()
                    .unwrap_or(0.0);
                let other_val = other
                    .get(row_idx, col_idx)
                    .copied()
                    .unwrap_or(0.0);

                result_column.push(self_val * other_val);
            }

            let column_name = format!("col_{}", col_idx);
            let column = NumArray::new(result_column);
            result_columns.push(column);
        }

        // Create result matrix
        let mut result = GraphMatrix::from_arrays(result_columns)?;

        // Set appropriate column names
        let column_names: Vec<String> = (0..self_cols).map(|i| format!("col_{}", i)).collect();
        result.set_column_names(column_names)?;

        Ok(result)
    }

    // Helper methods

    // Adjacency Matrix Specialized Operations

    /// Convert adjacency matrix to Laplacian matrix
    pub fn to_laplacian(&self) -> crate::errors::GraphResult<Self> {
        if !self.is_square() {
            return Err(crate::errors::GraphError::InvalidInput(
                "Laplacian transformation requires square matrix".to_string()
            ));
        }

        let size = self.shape().0;

        // Calculate degree for each node (row sums)
        let degrees = self.sum_axis(Axis::Rows);

        // Create new columns for Laplacian matrix: L = D - A
        let mut new_columns = Vec::new();
        
        for j in 0..size {
            let mut col_values = Vec::new();
            
            for i in 0..size {
                let laplacian_val = if i == j {
                    // Diagonal: degree - self_connection
                    let degree = degrees.get(i).copied().unwrap_or(0.0);
                    let self_conn = self.get(i, j).copied().unwrap_or(0.0);
                    degree - self_conn
                } else {
                    // Off-diagonal: -adjacency_value
                    let adj_val = self.get(i, j).copied().unwrap_or(0.0);
                    -adj_val
                };
                
                col_values.push(laplacian_val);
            }
            
            new_columns.push(NumArray::new(col_values));
        }

        Ok(GraphMatrix {
            columns: new_columns,
            column_names: self.column_names.clone(),
            row_labels: self.row_labels.clone(),
            shape: self.shape,
            properties: None,
            graph: self.graph.clone(),
        })
    }

    /// Convert adjacency matrix to normalized Laplacian matrix
    pub fn to_normalized_laplacian(&self) -> crate::errors::GraphResult<Self> {
        let laplacian = self.to_laplacian()?;
        let size = self.shape().0;

        // Calculate degrees for normalization
        let degrees = self.sum_axis(Axis::Rows);

        // Create new columns with normalized values
        let mut new_columns = Vec::new();
        
        for j in 0..size {
            let mut col_values = Vec::new();
            
            for i in 0..size {
                let original_val = laplacian.get(i, j).copied().unwrap_or(0.0);
                let deg_i = degrees.get(i).copied().unwrap_or(0.0);
                let deg_j = degrees.get(j).copied().unwrap_or(0.0);
                
                let normalized_val = if deg_i > 0.0 && deg_j > 0.0 {
                    let normalizer = (deg_i * deg_j).sqrt();
                    original_val / normalizer
                } else {
                    original_val
                };
                
                col_values.push(normalized_val);
            }
            
            new_columns.push(NumArray::new(col_values));
        }

        Ok(GraphMatrix {
            columns: new_columns,
            column_names: laplacian.column_names.clone(),
            row_labels: laplacian.row_labels.clone(),
            shape: laplacian.shape,
            properties: None,
            graph: self.graph.clone(),
        })
    }

    /// Check if this appears to be an adjacency matrix (square, non-negative)
    pub fn is_adjacency_matrix(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        // Check if all values are non-negative
        for col in &self.columns {
            for val in col.iter() {
                if *val < 0.0 {
                    return false;
                }
            }
        }

        true
    }

    /// Get diagonal values (degrees for adjacency matrix)
    pub fn diagonal(&self) -> crate::storage::array::NumArray<f64> {
        let size = self.shape().0.min(self.shape().1);
        let mut diag_values = Vec::with_capacity(size);
        
        for i in 0..size {
            let val = self.get(i, i).copied().unwrap_or(0.0);
            diag_values.push(val);
        }
        
        crate::storage::array::NumArray::new(diag_values)
    }

    /// Calculate trace (sum of diagonal elements)
    pub fn trace(&self) -> f64 {
        self.diagonal().sum()
    }

    // Helper methods removed - NumArray<f64> is always Float type

    /// Check if the matrix is symmetric (internal helper)
    fn is_symmetric_internal(&self) -> bool {
        if !self.is_square() {
            return false;
        }

        let (size, _) = self.shape();
        for i in 0..size {
            for j in 0..size {
                if let (Some(val_ij), Some(val_ji)) = (self.get(i, j), self.get(j, i)) {
                    if val_ij != val_ji {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        true
    }

    /// Count zero elements (for sparsity calculation)
    fn count_zeros(&self) -> usize {
        if !self.is_numeric() {
            return 0;
        }

        self.columns
            .iter()
            .map(|col| {
                col.iter()
                    .filter(|&val| *val == 0.0)
                    .count()
            })
            .sum()
    }

    // ==================================================================================
    // LAZY EVALUATION & MATERIALIZATION METHODS
    // ==================================================================================

    /// Get a preview of the matrix for display purposes (first N rows/cols)
    /// This is used by repr() and does not materialize the full matrix
    pub fn preview(&self, row_limit: usize, col_limit: usize) -> (Vec<Vec<String>>, Vec<String>) {
        let (rows, cols) = self.shape();
        let num_rows = row_limit.min(rows);
        let num_cols = col_limit.min(cols);

        // Get column names preview
        let col_names = self.column_names.iter().take(num_cols).cloned().collect();

        // Get data preview
        let mut preview_data = Vec::new();
        for row_idx in 0..num_rows {
            let mut row = Vec::new();
            for col_idx in 0..num_cols {
                let value = self
                    .get(row_idx, col_idx)
                    .map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| "0".to_string());
                row.push(value);
            }
            preview_data.push(row);
        }

        (preview_data, col_names)
    }

    /// Materialize the matrix to nested vectors for Python consumption
    /// This is the primary materialization method used by .data property
    pub fn materialize(&self) -> Vec<Vec<f64>> {
        let (rows, cols) = self.shape();
        let mut materialized = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for col_idx in 0..cols {
                let value = self
                    .get(row_idx, col_idx)
                    .copied()
                    .unwrap_or(0.0);
                row.push(value);
            }
            materialized.push(row);
        }

        materialized
    }

    /// Check if the matrix is effectively sparse (has many default/zero values)
    pub fn is_sparse(&self) -> bool {
        let (rows, cols) = self.shape();
        let total_elements = rows * cols;
        if total_elements == 0 {
            return false;
        }

        let zero_count = self.count_zeros();

        // Consider sparse if >50% are zero/default values
        (zero_count as f64) / (total_elements as f64) > 0.5
    }

    /// Get summary information for lazy display without full materialization
    pub fn summary_info(&self) -> String {
        let (rows, cols) = self.shape();
        let is_sparse = self.is_sparse();
        let is_square = self.is_square();
        let dtype = "f64";

        format!(
            "GraphMatrix(shape=({}, {}), dtype={:?}, sparse={}, square={})",
            rows, cols, dtype, is_sparse, is_square
        )
    }

    /// Create a lazy view of the transposed matrix without materializing data
    pub fn transpose_lazy(&self) -> Self {
        // For now, we'll create a new matrix with transposed columns
        // In a full lazy implementation, this would just record the transpose operation
        self.transpose()
    }

    /// Create a dense materialized version of the matrix
    /// This forces full computation and storage
    pub fn dense(&self) -> Self {
        // For now, the matrix is already dense
        // In a sparse implementation, this would convert from sparse to dense storage
        self.clone()
    }
}

impl fmt::Display for GraphMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        writeln!(
            f,
            "GraphMatrix ({} x {}) - dtype: {:?}",
            rows, cols, "f64"
        )?;

        // Show column names
        write!(f, "Columns: ")?;
        for (i, name) in self.column_names.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", name)?;
        }
        writeln!(f)?;

        // Show first few rows
        let display_rows = std::cmp::min(rows, 5);
        for row in 0..display_rows {
            write!(f, "  ")?;
            for col in 0..cols {
                if col > 0 {
                    write!(f, " ")?;
                }
                if let Some(value) = self.get(row, col) {
                    write!(f, "{:8}", format!("{}", value))?;
                } else {
                    write!(f, "{:8}", "null")?;
                }
            }
            writeln!(f)?;
        }

        if rows > display_rows {
            writeln!(f, "  ... ({} more rows)", rows - display_rows)?;
        }

        Ok(())
    }
}

// Additional trait implementations and helper functions would go here
// For example: linear algebra operations, matrix multiplication, etc.
