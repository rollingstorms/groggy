//! Python indexing utilities for converting Python index objects to Rust SliceIndex
//!
//! This module handles the conversion of Python indexing operations to our unified SliceIndex system.

use pyo3::prelude::*;
use pyo3::types::{PySlice, PyList, PyTuple};
use groggy::storage::array::{SliceIndex, BoolArray};
// PyBoolArray functionality integrated into unified NumArray

/// Convert Python indexing object to SliceIndex
pub fn python_index_to_slice_index(py: Python, index: &PyAny) -> PyResult<SliceIndex> {
    // Handle single integer
    if let Ok(int_val) = index.extract::<i64>() {
        return Ok(SliceIndex::Single(int_val));
    }
    
    // Handle Python slice object
    if let Ok(slice) = index.downcast::<PySlice>() {
        let indices = slice.indices(i64::MAX)?;
        return Ok(SliceIndex::Range {
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
        return Ok(SliceIndex::List(int_indices));
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
        return Ok(SliceIndex::List(int_indices));
    }
    
    // Handle NumArray with bool dtype (replaces BoolArray)
    if let Ok(num_array) = index.extract::<PyRef<crate::ffi::storage::num_array::PyNumArray>>() {
        if num_array.get_dtype() == "bool" {
            // Extract bool data from unified NumArray
            let bool_values: Vec<bool> = num_array.get_list(py)?.extract(py)?;
            let bool_array = groggy::storage::BoolArray::new(bool_values);
            return Ok(SliceIndex::BoolArray(bool_array));
        }
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
            let bool_array = BoolArray::new(bool_values);
            return Ok(SliceIndex::BoolArray(bool_array));
        }
    }
    
    Err(pyo3::exceptions::PyTypeError::new_err(
        format!("Invalid index type: expected int, slice, list of ints, or BoolArray, got {}", 
                index.get_type().name()?)
    ))
}

/// Convert Python slice indices with length constraint
pub fn python_slice_to_slice_index(py: Python, slice: &PySlice, length: usize) -> PyResult<SliceIndex> {
    let indices = slice.indices(length as i64)?;
    Ok(SliceIndex::Range {
        start: if indices.start == 0 && indices.stop == length as isize { 
            None 
        } else { 
            Some(indices.start as i64) 
        },
        stop: if indices.stop == length as isize { 
            None 
        } else { 
            Some(indices.stop as i64) 
        },
        step: if indices.step == 1 { 
            None 
        } else { 
            Some(indices.step as i64) 
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_single_integer_conversion() {
        Python::with_gil(|py| {
            let index = 5i64.to_object(py);
            let slice_index = python_index_to_slice_index(py, index.as_ref(py)).unwrap();
            
            match slice_index {
                SliceIndex::Single(val) => assert_eq!(val, 5),
                _ => panic!("Expected Single index"),
            }
        });
    }
    
    #[test]
    fn test_list_conversion() {
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::new(py, &[0i64, 2i64, 4i64]);
            let slice_index = python_index_to_slice_index(py, list).unwrap();
            
            match slice_index {
                SliceIndex::List(indices) => assert_eq!(indices, vec![0, 2, 4]),
                _ => panic!("Expected List index"),
            }
        });
    }
}