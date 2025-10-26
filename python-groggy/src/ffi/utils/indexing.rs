//! Python indexing utilities for converting Python index objects to Rust SliceIndex
//!
//! This module handles the conversion of Python indexing operations to our unified SliceIndex system.

use groggy::storage::array::{BoolArray, SliceIndex};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};
use std::os::raw::c_long;
// PyBoolArray functionality integrated into unified NumArray

/// Convert Python indexing object to SliceIndex
pub fn python_index_to_slice_index(_py: Python, index: &PyAny) -> PyResult<SliceIndex> {
    // Handle single integer
    if let Ok(int_val) = index.extract::<i64>() {
        return Ok(SliceIndex::Single(int_val));
    }

    // Handle Python slice object
    if let Ok(slice) = index.downcast::<PySlice>() {
        let indices = slice.indices(c_long::MAX)?;
        return Ok(SliceIndex::Range {
            start: if indices.start == 0 && indices.stop == c_long::MAX as isize {
                None
            } else {
                Some(indices.start as i64)
            },
            stop: if indices.stop == c_long::MAX as isize {
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
                    "List indices must be integers",
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
                    "Tuple indices must be integers",
                ));
            }
        }
        return Ok(SliceIndex::List(int_indices));
    }

    // Handle NumArray with bool dtype (replaces BoolArray)
    if let Ok(num_array) = index.extract::<PyRef<crate::ffi::storage::num_array::PyNumArray>>() {
        if num_array.get_dtype() == "bool" {
            // Extract bool data from unified NumArray
            let bool_values = num_array.to_bool_vec();
            let bool_array = groggy::storage::BoolArray::new(bool_values);
            return Ok(SliceIndex::BoolArray(bool_array));
        } else {
            // Handle NumArray with numeric dtype for fancy indexing
            let int_values = num_array.to_int64_vec();
            return Ok(SliceIndex::List(int_values));
        }
    }

    // Handle BaseArray for indexing
    if let Ok(base_array) = index.extract::<PyRef<crate::ffi::storage::array::PyBaseArray>>() {
        // Try to extract as integers for fancy indexing
        let values: Vec<_> = base_array.inner.iter().cloned().collect();
        
        // Check if all values are integers
        let mut int_indices = Vec::new();
        let mut is_all_int = true;
        for val in values.iter() {
            match val {
                groggy::types::AttrValue::SmallInt(i) => int_indices.push(*i as i64),
                groggy::types::AttrValue::Int(i) => int_indices.push(*i),
                groggy::types::AttrValue::Bool(_) => {
                    is_all_int = false;
                    break;
                }
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "BaseArray indexing requires integer or boolean values",
                    ));
                }
            }
        }
        
        if is_all_int {
            return Ok(SliceIndex::List(int_indices));
        }
        
        // Check if all values are booleans
        let mut bool_values = Vec::new();
        for val in values.iter() {
            match val {
                groggy::types::AttrValue::Bool(b) => bool_values.push(*b),
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "BaseArray must contain all integers or all booleans for indexing",
                    ));
                }
            }
        }
        
        let bool_array = groggy::storage::BoolArray::new(bool_values);
        return Ok(SliceIndex::BoolArray(bool_array));
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

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Invalid index type: expected int, slice, list of ints, or BoolArray, got {}",
        index.get_type().name()?
    )))
}

/// Convert Python slice indices with length constraint
///
/// # Future Feature
///
/// Designed for advanced slicing operations. Currently unused as basic
/// indexing patterns handle most use cases. Will be integrated when
/// NumPy-style advanced indexing is added to the Python API.
#[allow(dead_code)]
pub fn python_slice_to_slice_index(
    _py: Python,
    slice: &PySlice,
    length: usize,
) -> PyResult<SliceIndex> {
    let indices = slice.indices(length as c_long)?;
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
            let list = pyo3::types::PyList::new(py, [0i64, 2i64, 4i64]);
            let slice_index = python_index_to_slice_index(py, list).unwrap();

            match slice_index {
                SliceIndex::List(indices) => assert_eq!(indices, vec![0, 2, 4]),
                _ => panic!("Expected List index"),
            }
        });
    }
}
