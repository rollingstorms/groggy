//! Python matrix indexing utilities for 2D slicing operations
//!
//! This module handles the conversion of Python 2D indexing operations to matrix slicing.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyList, PyTuple};
use groggy::storage::matrix::{MatrixIndex, MatrixSlice};
use crate::ffi::storage::bool_array::PyBoolArray;
use crate::ffi::utils::indexing::python_index_to_slice_index;

/// Convert a single Python index to MatrixIndex
pub fn python_to_matrix_index(py: Python, index: &PyAny) -> PyResult<MatrixIndex> {
    // Handle single integer
    if let Ok(int_val) = index.extract::<i64>() {
        return Ok(MatrixIndex::Single(int_val));
    }
    
    // Handle Python slice object
    if let Ok(slice) = index.downcast::<PySlice>() {
        let indices = slice.indices(i64::MAX)?;
        return Ok(MatrixIndex::Range {
            start: if indices.start == 0 && indices.stop == i64::MAX as isize { 
                None 
            } else { 
                Some(indices.start as i64) 
            },
            stop: if indices.stop == i64::MAX as isize { 
                None 
            } else { 
                Some(indices.stop as i64) 
            },
            step: if indices.step == 1 { 
                None 
            } else { 
                Some(indices.step as i64) 
            },
        });
    }
    
    // Handle list of integers
    if let Ok(list) = index.downcast::<PyList>() {
        let mut int_indices = Vec::new();
        for item in list.iter() {
            if let Ok(int_val) = item.extract::<i64>() {
                int_indices.push(int_val);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "List indices must be integers"
                ));
            }
        }
        return Ok(MatrixIndex::List(int_indices));
    }
    
    // Handle tuple of integers (treat as list)
    if let Ok(tuple) = index.downcast::<PyTuple>() {
        let mut int_indices = Vec::new();
        for item in tuple.iter() {
            if let Ok(int_val) = item.extract::<i64>() {
                int_indices.push(int_val);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Tuple indices must be integers"
                ));
            }
        }
        return Ok(MatrixIndex::List(int_indices));
    }
    
    // Handle BoolArray
    if let Ok(bool_array) = index.extract::<PyRef<PyBoolArray>>() {
        return Ok(MatrixIndex::BoolArray(bool_array.inner.clone()));
    }
    
    // Handle Python list of booleans
    if let Ok(list) = index.downcast::<PyList>() {
        let mut all_bool = true;
        let mut bool_values = Vec::new();
        
        for item in list.iter() {
            if let Ok(bool_val) = item.extract::<bool>() {
                bool_values.push(bool_val);
            } else {
                all_bool = false;
                break;
            }
        }
        
        if all_bool {
            let bool_array = groggy::storage::BoolArray::new(bool_values);
            return Ok(MatrixIndex::BoolArray(bool_array));
        }
    }
    
    // Handle ellipsis or None as "all"
    if index.is_none() || index.to_string() == "..." {
        return Ok(MatrixIndex::All);
    }
    
    Err(pyo3::exceptions::PyTypeError::new_err(
        format!("Invalid matrix index type: expected int, slice, list of ints, or BoolArray, got {}", 
                index.get_type().name()?)
    ))
}

/// Convert Python 2D matrix indexing to MatrixSlice
pub fn python_to_matrix_slice(py: Python, key: &PyAny) -> PyResult<MatrixSlice> {
    // Handle tuple (row_index, col_index)
    if let Ok(tuple) = key.downcast::<PyTuple>() {
        match tuple.len() {
            2 => {
                let row_index = python_to_matrix_index(py, tuple.get_item(0)?)?;
                let col_index = python_to_matrix_index(py, tuple.get_item(1)?)?;
                Ok(MatrixSlice::new(row_index, col_index))
            }
            1 => {
                // Single element tuple - treat as row selection
                let row_index = python_to_matrix_index(py, tuple.get_item(0)?)?;
                Ok(MatrixSlice::row(row_index))
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Matrix indexing tuple must have 1 or 2 elements"
            ))
        }
    }
    // Handle single index - treat as row selection with all columns
    else {
        let row_index = python_to_matrix_index(py, key)?;
        Ok(MatrixSlice::row(row_index))
    }
}

/// Parse advanced matrix indexing syntax
pub enum MatrixIndexResult {
    /// Single cell: matrix[row, col] -> f64
    Cell(usize, usize),
    /// Row: matrix[row] -> NumArray  
    Row(usize),
    /// Column by name: matrix["col_name"] -> NumArray
    ColumnByName(String),
    /// Advanced slicing: matrix[slice] -> GraphMatrix
    Slice(MatrixSlice),
}

/// Parse Python matrix index into appropriate result type
pub fn parse_matrix_index(py: Python, key: &PyAny) -> PyResult<MatrixIndexResult> {
    // Multi-index access (row, col) -> single cell value
    if let Ok(indices) = key.extract::<(usize, usize)>() {
        let (row, col) = indices;
        return Ok(MatrixIndexResult::Cell(row, col));
    }

    // Single integer -> row access  
    if let Ok(row_index) = key.extract::<usize>() {
        return Ok(MatrixIndexResult::Row(row_index));
    }

    // String -> column access
    if let Ok(col_name) = key.extract::<String>() {
        return Ok(MatrixIndexResult::ColumnByName(col_name));
    }

    // Advanced slicing
    let matrix_slice = python_to_matrix_slice(py, key)?;
    Ok(MatrixIndexResult::Slice(matrix_slice))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_single_integer_to_matrix_index() {
        Python::with_gil(|py| {
            let index = 5i64.to_object(py);
            let matrix_index = python_to_matrix_index(py, index.as_ref(py)).unwrap();
            
            match matrix_index {
                MatrixIndex::Single(val) => assert_eq!(val, 5),
                _ => panic!("Expected Single index"),
            }
        });
    }
    
    #[test]
    fn test_tuple_to_matrix_slice() {
        Python::with_gil(|py| {
            let indices = (2i64, 3i64).to_object(py);
            let result = parse_matrix_index(py, indices.as_ref(py)).unwrap();
            
            match result {
                MatrixIndexResult::Cell(row, col) => {
                    assert_eq!(row, 2);
                    assert_eq!(col, 3);
                }
                _ => panic!("Expected Cell result"),
            }
        });
    }
}