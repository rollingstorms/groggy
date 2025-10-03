//! Advanced matrix slicing operations
//!
//! This module provides NumPy-style matrix slicing operations for GraphMatrix:
//! - 2D slice notation: matrix[:5, :3], matrix[::2, ::2]
//! - Advanced indexing: matrix[[0, 2], [1, 3]]
//! - Boolean indexing: matrix[row_mask, :]
//! - Mixed indexing: matrix[row_slice, col_list]

use crate::errors::{GraphError, GraphResult};
use crate::storage::BoolArray;
use crate::storage::{
    array::{NumArray, SliceIndex},
    GraphMatrix,
};

/// 2D slice index for matrix operations
#[derive(Debug, Clone)]
pub enum MatrixIndex {
    /// Single index for row or column: matrix[5] or matrix[:, 5]
    Single(i64),
    /// Range slice: matrix[start:end:step]
    Range {
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    },
    /// Integer list: matrix[[0, 2, 5]]
    List(Vec<i64>),
    /// Boolean array: matrix[bool_mask]
    BoolArray(BoolArray),
    /// All elements: matrix[:]
    All,
}

impl MatrixIndex {
    /// Convert to SliceIndex for array operations
    pub fn to_slice_index(&self) -> SliceIndex {
        match self {
            MatrixIndex::Single(idx) => SliceIndex::Single(*idx),
            MatrixIndex::Range { start, stop, step } => SliceIndex::Range {
                start: *start,
                stop: *stop,
                step: *step,
            },
            MatrixIndex::List(indices) => SliceIndex::List(indices.clone()),
            MatrixIndex::BoolArray(mask) => SliceIndex::BoolArray(mask.clone()),
            MatrixIndex::All => SliceIndex::Range {
                start: None,
                stop: None,
                step: None,
            },
        }
    }

    /// Resolve indices for the given dimension length
    pub fn resolve_indices(&self, length: usize) -> GraphResult<Vec<usize>> {
        self.to_slice_index().resolve_indices(length)
    }
}

/// 2D indexing specification for matrices
#[derive(Debug, Clone)]
pub struct MatrixSlice {
    pub row_index: MatrixIndex,
    pub col_index: MatrixIndex,
}

impl MatrixSlice {
    /// Create new matrix slice
    pub fn new(row_index: MatrixIndex, col_index: MatrixIndex) -> Self {
        Self {
            row_index,
            col_index,
        }
    }

    /// Create slice for single row
    pub fn row(row_index: MatrixIndex) -> Self {
        Self::new(row_index, MatrixIndex::All)
    }

    /// Create slice for single column
    pub fn column(col_index: MatrixIndex) -> Self {
        Self::new(MatrixIndex::All, col_index)
    }
}

/// Trait for matrix slicing operations
pub trait MatrixSlicing {
    /// Get submatrix using 2D slice specification
    fn get_submatrix(&self, slice: &MatrixSlice) -> GraphResult<GraphMatrix>;

    /// Get single row as NumArray
    fn get_row(&self, row_idx: usize) -> GraphResult<NumArray<f64>>;

    /// Get single column as NumArray
    fn get_column(&self, col_idx: usize) -> GraphResult<NumArray<f64>>;

    /// Get multiple rows
    fn get_rows(&self, row_indices: &[usize]) -> GraphResult<GraphMatrix>;

    /// Get multiple columns
    fn get_columns(&self, col_indices: &[usize]) -> GraphResult<GraphMatrix>;

    /// Get single cell value
    fn get_cell(&self, row: usize, col: usize) -> GraphResult<f64>;
}

impl MatrixSlicing for GraphMatrix {
    fn get_submatrix(&self, slice: &MatrixSlice) -> GraphResult<GraphMatrix> {
        let (rows, cols) = self.shape();

        // Resolve row and column indices
        let row_indices = slice.row_index.resolve_indices(rows)?;
        let col_indices = slice.col_index.resolve_indices(cols)?;

        if row_indices.is_empty() || col_indices.is_empty() {
            return Err(GraphError::InvalidInput(
                "Cannot create matrix with zero rows or columns".to_string(),
            ));
        }

        // Extract the submatrix data
        let mut new_columns = Vec::new();
        let mut new_column_names = Vec::new();

        for &col_idx in &col_indices {
            if col_idx >= cols {
                return Err(GraphError::InvalidInput(format!(
                    "Column index {} out of bounds",
                    col_idx
                )));
            }

            // Get the full column
            let column = self.get_column_internal(col_idx)?;

            // Extract only the requested rows
            let mut row_data = Vec::new();
            for &row_idx in &row_indices {
                if row_idx >= rows {
                    return Err(GraphError::InvalidInput(format!(
                        "Row index {} out of bounds",
                        row_idx
                    )));
                }
                row_data.push(*column.get(row_idx).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Row {} not found in column", row_idx))
                })?);
            }

            new_columns.push(NumArray::new(row_data));
            new_column_names.push(self.get_column_name(col_idx));
        }

        // Create new matrix
        let mut result = GraphMatrix::from_arrays(new_columns)?;
        result.set_column_names(new_column_names);
        Ok(result)
    }

    fn get_row(&self, row_idx: usize) -> GraphResult<NumArray<f64>> {
        let (rows, cols) = self.shape();
        if row_idx >= rows {
            return Err(GraphError::InvalidInput(format!(
                "Row index {} out of bounds",
                row_idx
            )));
        }

        let mut row_data = Vec::new();
        for col_idx in 0..cols {
            let column = self.get_column_internal(col_idx)?;
            row_data.push(
                *column.get(row_idx).ok_or_else(|| {
                    GraphError::InvalidInput(format!("Row {} not found", row_idx))
                })?,
            );
        }

        Ok(NumArray::new(row_data))
    }

    fn get_column(&self, col_idx: usize) -> GraphResult<NumArray<f64>> {
        let (_, cols) = self.shape();
        if col_idx >= cols {
            return Err(GraphError::InvalidInput(format!(
                "Column index {} out of bounds",
                col_idx
            )));
        }

        Ok(NumArray::new(self.get_column_internal(col_idx)?))
    }

    fn get_rows(&self, row_indices: &[usize]) -> GraphResult<GraphMatrix> {
        let slice = MatrixSlice::new(
            MatrixIndex::List(row_indices.iter().map(|&i| i as i64).collect()),
            MatrixIndex::All,
        );
        self.get_submatrix(&slice)
    }

    fn get_columns(&self, col_indices: &[usize]) -> GraphResult<GraphMatrix> {
        let slice = MatrixSlice::new(
            MatrixIndex::All,
            MatrixIndex::List(col_indices.iter().map(|&i| i as i64).collect()),
        );
        self.get_submatrix(&slice)
    }

    fn get_cell(&self, row: usize, col: usize) -> GraphResult<f64> {
        let (rows, cols) = self.shape();
        if row >= rows {
            return Err(GraphError::InvalidInput(format!(
                "Row index {} out of bounds",
                row
            )));
        }
        if col >= cols {
            return Err(GraphError::InvalidInput(format!(
                "Column index {} out of bounds",
                col
            )));
        }

        let column = self.get_column_internal(col)?;
        column.get(row).copied().ok_or_else(|| {
            GraphError::InvalidInput(format!("Cell at [{}, {}] not found", row, col))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_matrix() -> GraphMatrix {
        let col1 = NumArray::new(vec![1.0, 2.0, 3.0, 4.0]);
        let col2 = NumArray::new(vec![5.0, 6.0, 7.0, 8.0]);
        let col3 = NumArray::new(vec![9.0, 10.0, 11.0, 12.0]);

        GraphMatrix::from_arrays(vec![col1, col2, col3]).unwrap()
    }

    #[test]
    fn test_single_cell_access() {
        let matrix = create_test_matrix();
        assert_eq!(matrix.get_cell(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get_cell(1, 2).unwrap(), 10.0);
        assert_eq!(matrix.get_cell(3, 1).unwrap(), 8.0);
    }

    #[test]
    fn test_row_access() {
        let matrix = create_test_matrix();
        let row = matrix.get_row(1).unwrap();
        assert_eq!(row.to_vec(), vec![2.0, 6.0, 10.0]);
    }

    #[test]
    fn test_column_access() {
        let matrix = create_test_matrix();
        let col = matrix.get_column(1).unwrap();
        assert_eq!(col.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_submatrix_slicing() {
        let matrix = create_test_matrix();

        // Select rows [0, 2] and columns [0, 2]
        let slice = MatrixSlice::new(MatrixIndex::List(vec![0, 2]), MatrixIndex::List(vec![0, 2]));

        let submatrix = matrix.get_submatrix(&slice).unwrap();
        let (rows, cols) = submatrix.shape();

        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(submatrix.get_cell(0, 0).unwrap(), 1.0); // Original [0, 0]
        assert_eq!(submatrix.get_cell(0, 1).unwrap(), 9.0); // Original [0, 2]
        assert_eq!(submatrix.get_cell(1, 0).unwrap(), 3.0); // Original [2, 0]
        assert_eq!(submatrix.get_cell(1, 1).unwrap(), 11.0); // Original [2, 2]
    }

    #[test]
    fn test_range_slicing() {
        let matrix = create_test_matrix();

        // Select first 2 rows and first 2 columns
        let slice = MatrixSlice::new(
            MatrixIndex::Range {
                start: Some(0),
                stop: Some(2),
                step: Some(1),
            },
            MatrixIndex::Range {
                start: Some(0),
                stop: Some(2),
                step: Some(1),
            },
        );

        let submatrix = matrix.get_submatrix(&slice).unwrap();
        let (rows, cols) = submatrix.shape();

        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        assert_eq!(submatrix.get_cell(0, 0).unwrap(), 1.0);
        assert_eq!(submatrix.get_cell(1, 1).unwrap(), 6.0);
    }
}
