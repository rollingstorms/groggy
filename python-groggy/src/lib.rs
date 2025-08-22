//! Groggy Python Bindings
//!
//! Fast graph library with statistical operations and memory-efficient processing.

use pyo3::prelude::*;

// Import all FFI modules
mod ffi;
mod module;

// Re-export main types
pub use ffi::api::graph::PyGraph;
pub use ffi::api::graph_version::PyHistoricalView;
pub use ffi::core::accessors::{PyEdgesAccessor, PyNodesAccessor};
pub use ffi::core::array::PyGraphArray;
pub use ffi::core::history::{PyBranchInfo, PyCommit, PyHistoryStatistics};
pub use ffi::core::matrix::PyGraphMatrix;
pub use ffi::core::query::{PyAttributeFilter, PyEdgeFilter, PyNodeFilter};
pub use ffi::core::subgraph::PySubgraph;
pub use ffi::core::table::{PyGraphTable, PyGroupBy};
pub use ffi::core::traversal::{
    PyAggregationResult, PyGroupedAggregationResult, PyTraversalResult,
};
pub use ffi::core::views::{PyEdgeView, PyNodeView};
pub use ffi::types::{PyAttrValue, PyAttributeCollection, PyResultHandle};

// ====================================================================
// UNIFIED BUILDER PATTERNS
// ====================================================================

/// Create a GraphArray from a Python list or array-like object
///
/// Examples:
///   gr.array([1, 2, 3, 4])
///   gr.array(['a', 'b', 'c'])
///   gr.array([1.0, 2.5, 3.7])
#[pyfunction]
#[pyo3(signature = (values))]
fn array(values: Vec<PyObject>) -> PyResult<PyGraphArray> {
    PyGraphArray::from_py_objects(values)
}

/// Create a GraphMatrix from arrays or nested lists
///
/// Examples:
///   gr.matrix([[1, 2], [3, 4]])  # From nested lists
///   gr.matrix([arr1, arr2])      # From GraphArrays
///   gr.matrix.zeros(3, 3)        # Factory methods available on class
#[pyfunction]
#[pyo3(signature = (data))]
fn matrix(py: Python, data: PyObject) -> PyResult<PyGraphMatrix> {
    // Check if data is a list of lists (nested structure)
    if let Ok(nested_lists) = data.extract::<Vec<Vec<PyObject>>>(py) {
        // Convert nested lists to core GraphArrays directly
        let mut core_arrays = Vec::new();
        for row in nested_lists {
            let py_array = PyGraphArray::from_py_objects(row)?;
            core_arrays.push(py_array.inner);
        }

        // Create core GraphMatrix directly
        let matrix = groggy::core::matrix::GraphMatrix::from_arrays(core_arrays).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create matrix: {:?}",
                e
            ))
        })?;

        Ok(PyGraphMatrix { inner: matrix })
    }
    // Check if data is a flat list (single row)
    else if let Ok(flat_list) = data.extract::<Vec<PyObject>>(py) {
        // Convert single list to single-row matrix
        let py_array = PyGraphArray::from_py_objects(flat_list)?;
        let core_arrays = vec![py_array.inner];

        // Create core GraphMatrix directly
        let matrix = groggy::core::matrix::GraphMatrix::from_arrays(core_arrays).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create matrix: {:?}",
                e
            ))
        })?;

        Ok(PyGraphMatrix { inner: matrix })
    }
    // Check if data is a list of GraphArrays
    else if let Ok(array_list) = data.extract::<Vec<Py<PyGraphArray>>>(py) {
        PyGraphMatrix::new(py, array_list)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "matrix() expects nested lists [[1,2],[3,4]], flat list [1,2,3,4], or list of GraphArrays"
        ))
    }
}

/// Create a GraphTable from dictionary or arrays with column names
///
/// Examples:
///   gr.table({'name': ['Alice', 'Bob'], 'age': [25, 30]})
///   gr.table([arr1, arr2], columns=['col1', 'col2'])
#[pyfunction]
#[pyo3(signature = (data, columns = None))]
fn table(py: Python, data: PyObject, columns: Option<Vec<String>>) -> PyResult<PyGraphTable> {
    use pyo3::types::PyDict;

    // Check if data is a dictionary
    if let Ok(dict) = data.extract::<&PyDict>(py) {
        let mut core_arrays = Vec::new();
        let mut column_names = Vec::new();

        // Convert dictionary to core arrays directly
        for (key, value) in dict.iter() {
            let col_name: String = key.extract()?;
            let values: Vec<PyObject> = value.extract()?;
            let py_array = PyGraphArray::from_py_objects(values)?;

            core_arrays.push(py_array.inner);
            column_names.push(col_name);
        }

        // Create core GraphTable directly
        let table =
            groggy::core::table::GraphTable::from_arrays(core_arrays, Some(column_names), None)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to create table: {:?}",
                        e
                    ))
                })?;

        Ok(PyGraphTable { inner: table })
    }
    // Check if data is a list of lists (nested structure for columns)
    else if let Ok(nested_lists) = data.extract::<Vec<Vec<PyObject>>>(py) {
        // Convert nested lists to core arrays directly
        let mut core_arrays = Vec::new();
        for column in nested_lists {
            let py_array = PyGraphArray::from_py_objects(column)?;
            core_arrays.push(py_array.inner);
        }

        // Create core GraphTable directly
        let table = groggy::core::table::GraphTable::from_arrays(core_arrays, columns, None)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create table: {:?}",
                    e
                ))
            })?;

        Ok(PyGraphTable { inner: table })
    }
    // Check if data is a list of arrays
    else if let Ok(array_list) = data.extract::<Vec<Py<PyGraphArray>>>(py) {
        PyGraphTable::new(py, array_list, columns)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "table() expects dictionary {'col': [values]}, nested lists [[col1_values], [col2_values]], or list of GraphArrays"
        ))
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _groggy(py: Python, m: &PyModule) -> PyResult<()> {
    // Register core graph components
    m.add_class::<PyGraph>()?;
    m.add_class::<PySubgraph>()?;

    // Register array and matrix types
    m.add_class::<PyGraphArray>()?;
    m.add_class::<PyGraphMatrix>()?;
    m.add_class::<PyGraphTable>()?;
    m.add_class::<PyGroupBy>()?;

    // Register accessor and view types
    m.add_class::<PyNodesAccessor>()?;
    m.add_class::<PyEdgesAccessor>()?;
    m.add_class::<PyNodeView>()?;
    m.add_class::<PyEdgeView>()?;

    // Register type system
    m.add_class::<PyAttrValue>()?;
    m.add_class::<PyResultHandle>()?;
    m.add_class::<PyAttributeCollection>()?;

    // Register query and filter system
    m.add_class::<PyAttributeFilter>()?;
    m.add_class::<PyNodeFilter>()?;
    m.add_class::<PyEdgeFilter>()?;

    // Register version control system
    m.add_class::<PyCommit>()?;
    m.add_class::<PyBranchInfo>()?;
    m.add_class::<PyHistoryStatistics>()?;
    m.add_class::<PyHistoricalView>()?;

    // Register traversal and aggregation results
    m.add_class::<PyTraversalResult>()?;
    m.add_class::<PyAggregationResult>()?;
    m.add_class::<PyGroupedAggregationResult>()?;

    // Add aliases for Python imports - these are already added with correct names

    // Register display functions
    ffi::display::register_display_functions(py, m)?;

    // Add unified builder functions
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(matrix, m)?)?;
    m.add_function(wrap_pyfunction!(table, m)?)?;

    // Use the module registration function
    module::register_classes(py, m)?;

    Ok(())
}
