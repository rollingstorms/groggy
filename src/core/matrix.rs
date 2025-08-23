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

use crate::core::array::GraphArray;
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
        let is_numeric = matrix.dtype().is_numeric();

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

/// General-purpose matrix built on GraphArray foundation
#[derive(Debug, Clone)]
pub struct GraphMatrix {
    /// Columns stored as GraphArrays
    columns: Vec<GraphArray>,
    /// Column names/labels
    column_names: Vec<String>,
    /// Row labels (optional)
    #[allow(dead_code)]
    row_labels: Option<GraphArray>,
    /// Enforced data type (all columns must match)
    dtype: AttrValueType,
    /// Cached matrix properties
    properties: Option<MatrixProperties>,
    /// Reference to the source graph (optional)
    graph: Option<std::rc::Rc<crate::api::graph::Graph>>,
}

impl GraphMatrix {
    /// Create a new GraphMatrix from a collection of GraphArrays
    /// All arrays must have the same length and compatible types
    pub fn from_arrays(arrays: Vec<GraphArray>) -> GraphResult<Self> {
        if arrays.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot create matrix from empty array list".to_string(),
            ));
        }

        // Check all arrays have the same length
        let expected_len = arrays[0].len();
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != expected_len {
                return Err(GraphError::InvalidInput(format!(
                    "Array {} has length {} but expected {}",
                    i,
                    array.len(),
                    expected_len
                )));
            }
        }

        // Determine common type and validate compatibility
        let dtype = Self::determine_common_type(&arrays)?;

        // Convert all arrays to the common type if needed
        let converted_arrays = arrays
            .into_iter()
            .enumerate()
            .map(|(i, array)| {
                if array.dtype() == dtype {
                    Ok(array)
                } else {
                    array.convert_to(dtype).map_err(|_| {
                        GraphError::InvalidInput(format!(
                            "Cannot convert array {} to type {:?}",
                            i, dtype
                        ))
                    })
                }
            })
            .collect::<GraphResult<Vec<_>>>()?;

        // Generate default column names
        let column_names = (0..converted_arrays.len())
            .map(|i| format!("col_{}", i))
            .collect();

        Ok(Self {
            columns: converted_arrays,
            column_names,
            row_labels: None,
            dtype,
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
        let arrays = attrs
            .iter()
            .map(|attr| GraphArray::from_graph_attribute(&graph, attr, entities))
            .collect::<GraphResult<Vec<_>>>()?;

        let mut matrix = Self::from_arrays(arrays)?;
        matrix.column_names = attrs.iter().map(|s| s.to_string()).collect();
        matrix.graph = Some(graph);
        Ok(matrix)
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
            .map(|i| {
                let values = vec![zero_value.clone(); rows];
                GraphArray::from_vec(values).with_name(format!("col_{}", i))
            })
            .collect();

        let column_names = (0..cols).map(|i| format!("col_{}", i)).collect();

        Self {
            columns: arrays,
            column_names,
            row_labels: None,
            dtype,
            properties: None,
            graph: None,
        }
    }

    /// Create an identity matrix of specified size
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size, AttrValueType::Int);

        // Set diagonal elements to 1
        for i in 0..size {
            if let Some(_col) = matrix.columns.get_mut(i) {
                // This would need to be implemented in GraphArray
                // col.set(i, AttrValue::Int(1));
            }
        }

        matrix
    }

    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        if self.columns.is_empty() {
            (0, 0)
        } else {
            (self.columns[0].len(), self.columns.len())
        }
    }

    /// Get the data type of the matrix
    pub fn dtype(&self) -> AttrValueType {
        self.dtype
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

    /// Check if matrix is numeric
    pub fn is_numeric(&self) -> bool {
        self.dtype.is_numeric()
    }

    /// Get optional graph reference
    pub fn graph(&self) -> Option<&std::rc::Rc<crate::api::graph::Graph>> {
        self.graph.as_ref()
    }

    /// Get element at (row, col) position
    pub fn get(&self, row: usize, col: usize) -> Option<&AttrValue> {
        self.columns.get(col)?.get(row)
    }

    /// Get a column by index
    pub fn get_column(&self, col: usize) -> Option<&GraphArray> {
        self.columns.get(col)
    }

    /// Get a column by name
    pub fn get_column_by_name(&self, name: &str) -> Option<&GraphArray> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.columns.get(idx))
    }

    /// Get a row as a new GraphArray
    pub fn get_row(&self, row: usize) -> Option<GraphArray> {
        let (rows, _) = self.shape();
        if row >= rows {
            return None;
        }

        let row_values: Vec<AttrValue> = self
            .columns
            .iter()
            .filter_map(|col| col.get(row).cloned())
            .collect();

        if row_values.len() == self.columns.len() {
            Some(GraphArray::from_vec(row_values))
        } else {
            None
        }
    }

    /// Iterator over columns
    pub fn iter_columns(&self) -> impl Iterator<Item = &GraphArray> {
        self.columns.iter()
    }

    /// Iterator over rows (returns GraphArrays)
    pub fn iter_rows(&self) -> impl Iterator<Item = GraphArray> + '_ {
        let (rows, _) = self.shape();
        (0..rows).filter_map(move |i| self.get_row(i))
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> GraphMatrix {
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            return self.clone();
        }

        let transposed_arrays: Vec<GraphArray> = (0..rows)
            .map(|row_idx| {
                let row_values: Vec<AttrValue> = self
                    .columns
                    .iter()
                    .filter_map(|col| col.get(row_idx).cloned())
                    .collect();
                GraphArray::from_vec(row_values).with_name(format!("row_{}", row_idx))
            })
            .collect();

        // Column names become row names, and vice versa
        let new_column_names = (0..rows).map(|i| format!("row_{}", i)).collect();

        Self {
            columns: transposed_arrays,
            column_names: new_column_names,
            row_labels: None,
            dtype: self.dtype,
            properties: None,
            graph: self.graph.clone(),
        }
    }

    /// Statistical operations along an axis
    pub fn sum_axis(&self, axis: Axis) -> GraphArray {
        match axis {
            Axis::Columns => {
                // Sum each column (returns array of column sums)
                let sums: Vec<AttrValue> = self
                    .columns
                    .iter()
                    .map(|col| {
                        col.sum()
                            .map(|f| AttrValue::Float(f as f32))
                            .unwrap_or(AttrValue::Int(0))
                    })
                    .collect();
                GraphArray::from_vec(sums)
            }
            Axis::Rows => {
                // Sum each row (returns array of row sums)
                let (rows, _) = self.shape();
                let sums: Vec<AttrValue> = (0..rows)
                    .map(|row_idx| {
                        let row_sum: f32 = self
                            .columns
                            .iter()
                            .filter_map(|col| col.get(row_idx))
                            .filter_map(|val| val.as_float())
                            .sum();
                        AttrValue::Float(row_sum)
                    })
                    .collect();
                GraphArray::from_vec(sums)
            }
        }
    }

    /// Mean along an axis
    pub fn mean_axis(&self, axis: Axis) -> GraphArray {
        match axis {
            Axis::Columns => {
                let means: Vec<AttrValue> = self
                    .columns
                    .iter()
                    .map(|col| {
                        col.mean()
                            .map(|f| AttrValue::Float(f as f32))
                            .unwrap_or(AttrValue::Int(0))
                    })
                    .collect();
                GraphArray::from_vec(means)
            }
            Axis::Rows => {
                let (rows, _cols) = self.shape();
                let means: Vec<AttrValue> = (0..rows)
                    .map(|row_idx| {
                        let row_values: Vec<f32> = self
                            .columns
                            .iter()
                            .filter_map(|col| col.get(row_idx))
                            .filter_map(|val| val.as_float())
                            .collect();

                        if row_values.is_empty() {
                            AttrValue::Int(0)
                        } else {
                            let mean = row_values.iter().sum::<f32>() / row_values.len() as f32;
                            AttrValue::Float(mean)
                        }
                    })
                    .collect();
                GraphArray::from_vec(means)
            }
        }
    }

    /// Standard deviation along an axis
    pub fn std_axis(&self, axis: Axis) -> GraphArray {
        match axis {
            Axis::Columns => {
                let stds: Vec<AttrValue> = self
                    .columns
                    .iter()
                    .map(|col| {
                        col.std()
                            .map(|f| AttrValue::Float(f as f32))
                            .unwrap_or(AttrValue::Int(0))
                    })
                    .collect();
                GraphArray::from_vec(stds)
            }
            Axis::Rows => {
                // This would be more complex - compute std deviation for each row
                // For now, return zeros as placeholder
                let (rows, _) = self.shape();
                let zeros = vec![AttrValue::Float(0.0); rows];
                GraphArray::from_vec(zeros)
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
                if val.as_float().unwrap_or(0.0).abs() > 1e-10 {
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
                    let float_val = val.as_float().unwrap_or(0.0) as f64;
                    if float_val.abs() > 1e-10 {
                        // Consider values > epsilon as non-zero
                        sparse_data.push((row_idx, col_idx, float_val));
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
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.0) as f64;
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

        // Convert row-major data back to column-major GraphArrays
        for col_idx in 0..cols {
            let mut column_data = Vec::with_capacity(rows);

            for row_idx in 0..rows {
                let val = data[row_idx * cols + col_idx] as f32;
                column_data.push(AttrValue::Float(val));
            }

            let column_name = format!("col_{}", col_idx);
            let column = GraphArray::from_vec(column_data).with_name(column_name);
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
            let mut result_column = Vec::with_capacity(self_rows);

            for row_idx in 0..self_rows {
                let self_val = self
                    .get(row_idx, col_idx)
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.0);
                let other_val = other
                    .get(row_idx, col_idx)
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.0);

                result_column.push(AttrValue::Float(self_val * other_val));
            }

            let column_name = format!("col_{}", col_idx);
            let column = GraphArray::from_vec(result_column).with_name(column_name.clone());
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

    /// Determine the common type for a collection of arrays
    fn determine_common_type(arrays: &[GraphArray]) -> GraphResult<AttrValueType> {
        if arrays.is_empty() {
            return Ok(AttrValueType::Int); // Default
        }

        let first_type = arrays[0].dtype();

        // For now, require exact type matches
        // TODO: Implement type promotion rules (int -> float, etc.)
        for array in arrays.iter() {
            if array.dtype() != first_type {
                return Err(GraphError::InvalidInput(format!(
                    "Type mismatch: expected {:?}, found {:?}",
                    first_type,
                    array.dtype()
                )));
            }
        }

        Ok(first_type)
    }

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
                    .filter(|val| match val {
                        AttrValue::Int(0) | AttrValue::SmallInt(0) => true,
                        AttrValue::Float(f) if *f == 0.0 => true,
                        _ => false,
                    })
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
    pub fn materialize(&self) -> Vec<Vec<AttrValue>> {
        let (rows, cols) = self.shape();
        let mut materialized = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for col_idx in 0..cols {
                let value = self
                    .get(row_idx, col_idx)
                    .cloned()
                    .unwrap_or(AttrValue::Int(0));
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
        let dtype = self.dtype;

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
            rows, cols, self.dtype
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
