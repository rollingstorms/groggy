//! Groggy Python Bindings
//! 
//! Fast graph library with statistical operations and memory-efficient processing.

use pyo3::prelude::*;

// Import all FFI modules
mod ffi;
mod module;

// Re-export main types
pub use ffi::api::graph::PyGraph;
pub use ffi::core::subgraph::PySubgraph;
pub use ffi::core::array::PyGraphArray;
pub use ffi::core::matrix::PyGraphMatrix;
pub use ffi::core::accessors::{PyNodesAccessor, PyEdgesAccessor};
pub use ffi::core::views::{PyNodeView, PyEdgeView};
pub use ffi::core::table::PyGraphTable;
pub use ffi::types::{PyAttrValue, PyResultHandle, PyAttributeCollection};
pub use ffi::core::query::{PyAttributeFilter, PyNodeFilter, PyEdgeFilter};
pub use ffi::core::history::{PyCommit, PyBranchInfo, PyHistoryStatistics};
pub use ffi::api::graph_version::{PyHistoricalView};
pub use ffi::core::traversal::{PyTraversalResult, PyAggregationResult, PyGroupedAggregationResult};

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
        // Convert nested lists to GraphArrays
        let mut arrays = Vec::new();
        for row in nested_lists {
            let array = PyGraphArray::from_py_objects(row)?;
            arrays.push(Py::new(py, array)?);
        }
        PyGraphMatrix::new(py, arrays)
    } 
    // Check if data is a list of GraphArrays
    else if let Ok(array_list) = data.extract::<Vec<Py<PyGraphArray>>>(py) {
        PyGraphMatrix::new(py, array_list)
    }
    else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "matrix() expects either nested lists [[1,2],[3,4]] or list of GraphArrays"
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
        let mut arrays = Vec::new();
        let mut column_names = Vec::new();
        
        // Convert dictionary to arrays
        for (key, value) in dict.iter() {
            let col_name: String = key.extract()?;
            let values: Vec<PyObject> = value.extract()?;
            let array = PyGraphArray::from_py_objects(values)?;
            
            arrays.push(Py::new(py, array)?);
            column_names.push(col_name);
        }
        
        PyGraphTable::new(py, arrays, Some(column_names))
    }
    // Check if data is a list of arrays
    else if let Ok(array_list) = data.extract::<Vec<Py<PyGraphArray>>>(py) {
        PyGraphTable::new(py, array_list, columns)
    }
    else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "table() expects either a dictionary {'col': [values]} or list of GraphArrays with column names"
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