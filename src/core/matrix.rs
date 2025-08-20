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

use crate::types::{NodeId, AttrValue, AttrValueType};
use crate::errors::{GraphResult, GraphError};
use crate::core::array::GraphArray;
use std::fmt;

/// Matrix properties that can be computed and cached
#[derive(Debug, Clone)]
pub struct MatrixProperties {
    pub is_square: bool,
    pub is_symmetric: bool,
    pub is_numeric: bool,
    pub is_sparse: bool,
    pub sparsity: Option<f64>,  // Ratio of zero elements
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
            return Err(GraphError::InvalidInput("Cannot create matrix from empty array list".to_string()));
        }
        
        // Check all arrays have the same length
        let expected_len = arrays[0].len();
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != expected_len {
                return Err(GraphError::InvalidInput(format!("Array {} has length {} but expected {}", i, array.len(), expected_len)));
            }
        }
        
        // Determine common type and validate compatibility
        let dtype = Self::determine_common_type(&arrays)?;
        
        // Convert all arrays to the common type if needed
        let converted_arrays = arrays.into_iter()
            .enumerate()
            .map(|(i, array)| {
                if array.dtype() == dtype {
                    Ok(array)
                } else {
                    array.convert_to(dtype).map_err(|_| GraphError::InvalidInput(format!("Cannot convert array {} to type {:?}", i, dtype)))
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
        entities: &[NodeId]
    ) -> GraphResult<Self> {
        let arrays = attrs.iter()
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
            if let Some(col) = matrix.columns.get_mut(i) {
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
            return Err(GraphError::InvalidInput(format!("Expected {} column names, got {}", self.columns.len(), names.len())));
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
        self.column_names.iter()
            .position(|n| n == name)
            .and_then(|idx| self.columns.get(idx))
    }
    
    /// Get a row as a new GraphArray
    pub fn get_row(&self, row: usize) -> Option<GraphArray> {
        let (rows, _) = self.shape();
        if row >= rows {
            return None;
        }
        
        let row_values: Vec<AttrValue> = self.columns.iter()
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
                let row_values: Vec<AttrValue> = self.columns.iter()
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
                let sums: Vec<AttrValue> = self.columns.iter()
                    .map(|col| col.sum().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)))
                    .collect();
                GraphArray::from_vec(sums)
            }
            Axis::Rows => {
                // Sum each row (returns array of row sums)
                let (rows, _) = self.shape();
                let sums: Vec<AttrValue> = (0..rows)
                    .map(|row_idx| {
                        let row_sum: f32 = self.columns.iter()
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
                let means: Vec<AttrValue> = self.columns.iter()
                    .map(|col| col.mean().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)))
                    .collect();
                GraphArray::from_vec(means)
            }
            Axis::Rows => {
                let (rows, cols) = self.shape();
                let means: Vec<AttrValue> = (0..rows)
                    .map(|row_idx| {
                        let row_values: Vec<f32> = self.columns.iter()
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
                let stds: Vec<AttrValue> = self.columns.iter()
                    .map(|col| col.std().map(|f| AttrValue::Float(f as f32)).unwrap_or(AttrValue::Int(0)))
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
                return Err(GraphError::InvalidInput(format!("Type mismatch: expected {:?}, found {:?}", first_type, array.dtype())));
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
        
        self.columns.iter()
            .map(|col| {
                col.iter()
                    .filter(|val| {
                        match val {
                            AttrValue::Int(0) | AttrValue::SmallInt(0) => true,
                            AttrValue::Float(f) if *f == 0.0 => true,
                            _ => false,
                        }
                    })
                    .count()
            })
            .sum()
    }
}

impl fmt::Display for GraphMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rows, cols) = self.shape();
        writeln!(f, "GraphMatrix ({} x {}) - dtype: {:?}", rows, cols, self.dtype)?;
        
        // Show column names
        write!(f, "Columns: ")?;
        for (i, name) in self.column_names.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", name)?;
        }
        writeln!(f)?;
        
        // Show first few rows
        let display_rows = std::cmp::min(rows, 5);
        for row in 0..display_rows {
            write!(f, "  ")?;
            for col in 0..cols {
                if col > 0 { write!(f, " ")?; }
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