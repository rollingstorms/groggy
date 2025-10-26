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
pub use ffi::api::graph_version::{PyBranchInfo, PyCommit, PyHistoryStatistics};
// Re-enabled accessor exports for table integration
pub use ffi::storage::accessors::{PyEdgesAccessor, PyNodesAccessor};
pub use ffi::storage::array::{PyBaseArray, PyEdgesArray, PyMetaNodeArray, PyNodesArray};
pub use ffi::storage::array_array::PyArrayArray;
pub use ffi::storage::num_array::{
    PyNumArray, PyNumArrayIterator, PyStatsArray, PyStatsArrayIterator,
};
// BoolArray and SimpleStatsArray functionality integrated into unified NumArray
pub use ffi::storage::components::PyComponentsArray;
pub use ffi::storage::edges_array::{PyEdgesArray as PyEdgesArrayNew, PyEdgesArrayIterator};
pub use ffi::storage::matrix::PyGraphMatrix;
pub use ffi::storage::matrix_array::{PyMatrixArray, PyMatrixArrayIterator};
pub use ffi::storage::nodes_array::{PyNodesArray as PyNodesArrayNew, PyNodesArrayIterator};
pub use ffi::storage::subgraph_array::{PySubgraphArray, PySubgraphArrayIterator};
pub use ffi::storage::table_array::{
    PyTableArray, PyTableArrayChainIterator, PyTableArrayIterator,
};
pub use ffi::storage::table_array_core::{PyTableArrayCore, PyTableArrayCoreIterator};
pub use ffi::subgraphs::component::PyComponentSubgraph;
pub use ffi::subgraphs::neighborhood::{
    PyNeighborhoodArray, PyNeighborhoodResult, PyNeighborhoodStats, PyNeighborhoodSubgraph,
};
// pub use ffi::storage::path_result::PyPathResult; // Unused
pub use ffi::query::query::{PyAttributeFilter, PyEdgeFilter, PyNodeFilter};
pub use ffi::query::query_parser::{parse_edge_query, parse_node_query};
pub use ffi::subgraphs::subgraph::PySubgraph;
// Re-enabled table exports for Phase 5 completion
pub use ffi::query::traversal::{PyAggregationResult, PyGroupedAggregationResult};
pub use ffi::storage::table::{PyBaseTable, PyEdgesTable, PyGraphTable, PyNodesTable};
// pub use ffi::storage::views::{PyEdgeView, PyNodeView}; // Temporarily disabled
pub use ffi::types::{PyAttrValue, PyAttributeCollection, PyResultHandle};

// Entity system - trait-based wrappers
pub use ffi::entities::{PyEdge, PyMetaEdge, PyMetaNode, PyNode};

// Hierarchical subgraph types
pub use ffi::subgraphs::hierarchical::PyAggregationFunction;

// MetaGraph Composer types
pub use ffi::subgraphs::composer::{
    PyComposerPreview, PyEdgeStrategy, PyMetaNodePlan, PyMetaNodePlanExecutor,
};

// Display system exports
pub use ffi::display::{PyDisplayConfig, PyTableFormatter};

// Viz accessor for .viz property functionality
pub use ffi::viz_accessor::VizAccessor;

// ====================================================================
// UNIFIED BUILDER PATTERNS
// ====================================================================

/// Create a NumArray for numerical operations from a Python list
///
/// Examples:
///   gr.num_array([1, 2, 3, 4])
///   gr.num_array([1.0, 2.5, 3.7])
#[pyfunction]
#[pyo3(signature = (values))]
#[allow(dead_code)]
fn num_array(values: Vec<f64>) -> PyResult<PyNumArray> {
    Ok(PyNumArray::new(values))
}

/// Create a BaseArray or specialized array from a Python list
///
/// This function intelligently handles:
/// - Primitive values (int, float, str, bool) -> BaseArray
/// - Groggy Subgraph objects -> SubgraphArray
/// - Groggy NeighborhoodSubgraph objects -> SubgraphArray
///
/// Examples:
///   gr.array([1, 2, 3, 4])
///   gr.array(['a', 'b', 'c'])
///   gr.array([1.0, 2.5, 3.7])
///   gr.array(neighborhood.neighborhoods)  # List of NeighborhoodSubgraph
///   gr.array([subgraph1, subgraph2])      # List of Subgraph
#[pyfunction]
#[pyo3(signature = (values))]
fn array(values: Vec<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if values.is_empty() {
            // Empty array -> return empty BaseArray
            return Ok(PyBaseArray {
                inner: ::groggy::storage::array::BaseArray::new(vec![]),
            }
            .into_py(py));
        }

        // Check the type of the first element to determine what kind of array to create
        let first = &values[0];

        // Try to extract as Subgraph
        if let Ok(_) = first.extract::<crate::ffi::subgraphs::subgraph::PySubgraph>(py) {
            let subgraphs: Result<Vec<crate::ffi::subgraphs::subgraph::PySubgraph>, _> =
                values.iter().map(|v| v.extract::<crate::ffi::subgraphs::subgraph::PySubgraph>(py)).collect();
            
            if let Ok(subgraphs) = subgraphs {
                return Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(subgraphs).into_py(py));
            }
        }

        // Try to extract as NeighborhoodSubgraph
        if let Ok(_neighborhood_subgraph) = first.extract::<crate::ffi::subgraphs::neighborhood::PyNeighborhoodSubgraph>(py) {
            // Convert NeighborhoodSubgraph to regular Subgraph
            let subgraphs: Result<Vec<crate::ffi::subgraphs::subgraph::PySubgraph>, _> = values
                .iter()
                .map(|v| {
                    let ns = v.extract::<crate::ffi::subgraphs::neighborhood::PyNeighborhoodSubgraph>(py)?;
                    // Get the underlying subgraph
                    ns.subgraph(py)
                })
                .collect();

            if let Ok(subgraphs) = subgraphs {
                return Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(subgraphs).into_py(py));
            }
        }

        // Default: Convert to AttrValues for BaseArray
        let mut attr_values = Vec::new();
        for value in values {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(value.as_ref(py))?;
            attr_values.push(attr_value);
        }
        Ok(PyBaseArray {
            inner: ::groggy::storage::array::BaseArray::new(attr_values),
        }
        .into_py(py))
    })
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
        // Convert nested lists to core NumArrays for GraphMatrix
        let mut core_arrays = Vec::new();
        for row in nested_lists {
            // Convert PyObjects to f64 values
            let mut f64_values = Vec::new();
            for py_obj in row {
                if let Ok(f_val) = py_obj.extract::<f64>(py) {
                    f64_values.push(f_val);
                } else if let Ok(i_val) = py_obj.extract::<i64>(py) {
                    f64_values.push(i_val as f64);
                } else if let Ok(b_val) = py_obj.extract::<bool>(py) {
                    f64_values.push(if b_val { 1.0 } else { 0.0 });
                } else {
                    f64_values.push(0.0); // Default for non-numeric
                }
            }
            // Create NumArray<f64> directly
            let num_array = ::groggy::storage::array::NumArray::new(f64_values);
            core_arrays.push(num_array);
        }

        // Create core GraphMatrix directly
        let matrix = ::groggy::storage::GraphMatrix::from_arrays(core_arrays).map_err(|e| {
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
        let mut f64_values = Vec::new();
        for py_obj in flat_list {
            if let Ok(f_val) = py_obj.extract::<f64>(py) {
                f64_values.push(f_val);
            } else if let Ok(i_val) = py_obj.extract::<i64>(py) {
                f64_values.push(i_val as f64);
            } else if let Ok(b_val) = py_obj.extract::<bool>(py) {
                f64_values.push(if b_val { 1.0 } else { 0.0 });
            } else {
                f64_values.push(0.0); // Default for non-numeric
            }
        }
        let num_array = ::groggy::storage::array::NumArray::new(f64_values);
        let core_arrays = vec![num_array];

        // Create core GraphMatrix directly
        let matrix = ::groggy::storage::GraphMatrix::from_arrays(core_arrays).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create matrix: {:?}",
                e
            ))
        })?;

        Ok(PyGraphMatrix { inner: matrix })
    }
    // Check if data is a list of BaseArrays
    else if let Ok(array_list) = data.extract::<Vec<Py<PyBaseArray>>>(py) {
        // Convert PyBaseArrays to NumArrays for GraphMatrix
        let mut core_arrays = Vec::new();
        for py_array_ref in array_list.iter() {
            let py_array = py_array_ref.borrow(py);
            // Convert BaseArray<AttrValue> to NumArray<f64>
            let mut f64_values = Vec::new();
            for attr_value in py_array.inner.iter() {
                match attr_value {
                    ::groggy::AttrValue::Float(f) => f64_values.push(*f as f64),
                    ::groggy::AttrValue::Int(i) => f64_values.push(*i as f64),
                    ::groggy::AttrValue::SmallInt(i) => f64_values.push(*i as f64),
                    ::groggy::AttrValue::Bool(b) => f64_values.push(if *b { 1.0 } else { 0.0 }),
                    _ => f64_values.push(0.0),
                }
            }
            let num_array = ::groggy::storage::array::NumArray::new(f64_values);
            core_arrays.push(num_array);
        }

        // Create core GraphMatrix
        let matrix = ::groggy::storage::GraphMatrix::from_arrays(core_arrays).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create matrix: {:?}",
                e
            ))
        })?;

        Ok(PyGraphMatrix::from_graph_matrix(matrix))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "matrix() expects nested lists [[1,2],[3,4]], flat list [1,2,3,4], or list of BaseArrays"
        ))
    }
}

/// Create a GraphTable from dictionary or arrays with column names
///
/// Examples:
///   gr.table({'name': ['Alice', 'Bob'], 'age': [25, 30]})
///   gr.table([arr1, arr2], columns=['col1', 'col2'])
/*
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
            ::groggy::storage::GraphTable::from_arrays(core_arrays, Some(column_names), None)
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
        let table = ::groggy::storage::GraphTable::from_arrays(core_arrays, columns, None)
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
*/
/// Merge multiple graphs into a single new graph
///
/// Examples:
///   gr.merge([g1, g2, g3])  # Merge multiple graphs
///   gr.merge([g1, g2])      # Merge two graphs
#[pyfunction]
fn merge(py: Python, graphs: Vec<Py<PyGraph>>) -> PyResult<PyObject> {
    if graphs.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot merge empty list of graphs",
        ));
    }

    // Create a new empty graph with the same directionality as the first graph
    let first_graph = graphs[0].borrow(py);
    let directed = first_graph.inner.borrow().graph_type() == ::groggy::types::GraphType::Directed;
    drop(first_graph); // Release borrow

    // Use the constructor through Python class instantiation
    let result_py = py.get_type::<PyGraph>().call1((directed,))?;
    let mut result: PyRefMut<PyGraph> = result_py.extract::<PyRefMut<PyGraph>>()?;

    // Add each graph to the result
    for graph_py in graphs {
        let graph = graph_py.borrow(py);
        result.add_graph(py, &graph)?;
    }

    // Return the PyObject
    Ok(result_py.to_object(py))
}

/// Create a BaseTable from a Python list of dictionaries or other data
///
/// Examples:
///   gr.table([{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}])
///   gr.table({"node_id": [1, 2, 3], "name": ["A", "B", "C"]})
#[pyfunction]
fn table(py: Python, data: &PyAny) -> PyResult<PyObject> {
    use ::groggy::storage::array::BaseArray;
    use ::groggy::storage::table::BaseTable;
    use ::groggy::AttrValue;
    use pyo3::types::{PyDict, PyList};
    use std::collections::HashMap;

    // Handle dict-like input: {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    if let Ok(dict) = data.downcast::<PyDict>() {
        let mut columns = HashMap::new();

        for (key, value) in dict.iter() {
            let column_name = key.extract::<String>()?;

            // Convert value to BaseArray
            if let Ok(list) = value.downcast::<PyList>() {
                // Convert list to AttrValues and create BaseArray
                let mut attr_values = Vec::new();
                for item in list.iter() {
                    let attr_val = crate::ffi::types::PyAttrValue::from_py_value(item)?;
                    attr_values.push(attr_val.inner);
                }

                let base_array = BaseArray::from_attr_values(attr_values);
                columns.insert(column_name, base_array);
            }
        }

        // Create BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let py_table = PyBaseTable::from_table(base_table);
        return Ok(py_table.into_py(py));
    }

    // Handle list-like input: [{"name": "Alice", "age": 25}, ...]
    if let Ok(list) = data.downcast::<PyList>() {
        if list.is_empty() {
            // Empty table
            let empty_table = BaseTable::from_columns(HashMap::new())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let py_table = PyBaseTable::from_table(empty_table);
            return Ok(py_table.into_py(py));
        }

        // Get column names from first row
        let first_row = list.get_item(0)?;
        if let Ok(row_dict) = first_row.downcast::<PyDict>() {
            let column_names: Vec<String> = row_dict
                .keys()
                .iter()
                .map(|k| k.extract::<String>())
                .collect::<Result<Vec<_>, _>>()?;

            // Collect data for each column
            let mut columns_data: HashMap<String, Vec<AttrValue>> = HashMap::new();
            for col_name in &column_names {
                columns_data.insert(col_name.clone(), Vec::new());
            }

            // Extract values from each row
            for row_py in list.iter() {
                if let Ok(row_dict) = row_py.downcast::<PyDict>() {
                    for col_name in &column_names {
                        if let Some(value) = row_dict.get_item(col_name)? {
                            let attr_val = crate::ffi::types::PyAttrValue::from_py_value(value)?;
                            columns_data.get_mut(col_name).unwrap().push(attr_val.inner);
                        }
                    }
                }
            }

            // Create BaseArrays from collected data
            let mut columns = HashMap::new();
            for (col_name, values) in columns_data {
                let base_array = BaseArray::from_attr_values(values);
                columns.insert(col_name, base_array);
            }

            // Create BaseTable
            let base_table = BaseTable::from_columns(columns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let py_table = PyBaseTable::from_table(base_table);
            return Ok(py_table.into_py(py));
        }
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "table() expects a dict of lists or a list of dicts",
    ))
}

/// Create NumArray with bool dtype (replaces BoolArray constructor)
#[pyfunction]
#[pyo3(name = "bool_array")]
#[allow(dead_code)]
fn bool_array_factory(values: Vec<bool>) -> PyNumArray {
    PyNumArray::new_bool(values)
}

/// Create NumArray with bool dtype filled with True values
#[pyfunction]
#[pyo3(name = "ones_bool")]
fn ones_bool_factory(size: usize) -> PyNumArray {
    let values = vec![true; size];
    PyNumArray::new_bool(values)
}

/// Create NumArray with bool dtype filled with False values
#[pyfunction]
#[pyo3(name = "zeros_bool")]
fn zeros_bool_factory(size: usize) -> PyNumArray {
    let values = vec![false; size];
    PyNumArray::new_bool(values)
}

/// Integer array factory function (backward compatibility)
#[pyfunction]
#[pyo3(name = "int_array")]
#[allow(dead_code)]
fn int_array_factory(values: Vec<i64>) -> PyNumArray {
    PyNumArray::new_int64(values)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _groggy(py: Python, m: &PyModule) -> PyResult<()> {
    // Register core graph components
    m.add_class::<PyGraph>()?;
    m.add_class::<PySubgraph>()?;

    // Register array and matrix types
    m.add_class::<PyBaseArray>()?;
    m.add_class::<PyArrayArray>()?;
    // TODO: Add PyArrayArrayIterator once we implement proper PyBaseArray conversion
    // m.add_class::<ffi::storage::array_array::PyArrayArrayIterator>()?;
    m.add_class::<PyNodesArray>()?;
    m.add_class::<PyEdgesArray>()?;
    m.add_class::<PyMetaNodeArray>()?;
    m.add_class::<PyComponentsArray>()?;
    m.add_class::<ffi::storage::components::PyComponentsArrayIterator>()?;
    m.add_class::<PyGraphMatrix>()?;

    // Register specialized arrays (Phase 2 - Unified Delegation Architecture)
    m.add_class::<PySubgraphArray>()?;
    m.add_class::<PySubgraphArrayIterator>()?;
    // Unified Numerical array API (supports all numeric types with dtype)
    m.add_class::<PyNumArray>()?;
    m.add_class::<PyNumArrayIterator>()?;
    m.add_class::<PyTableArray>()?;
    m.add_class::<PyTableArrayIterator>()?;
    m.add_class::<PyTableArrayChainIterator>()?;
    m.add_class::<PyTableArrayCore>()?;
    m.add_class::<PyTableArrayCoreIterator>()?;
    m.add_class::<PyNodesArrayNew>()?;
    m.add_class::<PyNodesArrayIterator>()?;
    m.add_class::<PyEdgesArrayNew>()?;
    m.add_class::<PyEdgesArrayIterator>()?;
    m.add_class::<PyMatrixArray>()?;
    m.add_class::<PyMatrixArrayIterator>()?;
    // m.add_class::<PyPathResult>()?; // Unused
    m.add_class::<PyGraphTable>()?; // Re-enabled for Phase 5 completion
                                    // m.add_class::<PyGroupBy>()?; // Still disabled
    m.add_class::<PyBaseTable>()?; // Re-enabled for Phase 5 completion

    // Register pipeline submodule
    let pipeline_mod = PyModule::new(py, "pipeline")?;
    crate::ffi::api::pipeline::pipeline(py, pipeline_mod)?;
    m.add_submodule(pipeline_mod)?;
    m.add_class::<PyNodesTable>()?; // Re-enabled for Phase 5 completion
    m.add_class::<PyEdgesTable>()?; // Re-enabled for Phase 5 completion

    // Register display system types
    m.add_class::<PyDisplayConfig>()?;
    m.add_class::<PyTableFormatter>()?;

    // Register accessor and view types
    m.add_class::<ffi::storage::accessors::PyNodesAccessor>()?; // Re-enabled for table integration
    m.add_class::<ffi::storage::accessors::PyEdgesAccessor>()?; // Re-enabled for table integration
                                                                // m.add_class::<PyNodeView>()?; // Still disabled - not essential for current functionality
                                                                // m.add_class::<PyEdgeView>()?; // Still disabled - not essential for current functionality

    // Register viz accessor
    m.add_class::<VizAccessor>()?;

    // Register type system
    m.add_class::<PyAttrValue>()?;
    m.add_class::<PyResultHandle>()?;
    m.add_class::<PyAttributeCollection>()?;

    // Register query and filter system
    m.add_class::<PyAttributeFilter>()?;
    m.add_class::<PyNodeFilter>()?;
    m.add_class::<PyEdgeFilter>()?;

    // Register core query functions (eliminates circular dependency)
    m.add_function(wrap_pyfunction!(parse_node_query, m)?)?;
    m.add_function(wrap_pyfunction!(parse_edge_query, m)?)?;

    // Register version control system
    // Register version control classes
    m.add_class::<PyCommit>()?;
    m.add_class::<PyBranchInfo>()?;
    m.add_class::<PyHistoryStatistics>()?;
    m.add_class::<ffi::temporal::PyTemporalSnapshot>()?;
    m.add_class::<ffi::temporal::PyExistenceIndex>()?;
    m.add_class::<ffi::temporal::PyTemporalIndex>()?;
    m.add_class::<ffi::temporal::PyIndexStatistics>()?;
    m.add_class::<ffi::temporal::PyTemporalScope>()?;
    m.add_class::<ffi::temporal::PyTemporalDelta>()?;
    m.add_class::<ffi::temporal::PyChangedEntities>()?;
    m.add_class::<PyHistoricalView>()?;

    // Register specialized entity types
    m.add_class::<PyComponentSubgraph>()?;

    // Register trait-based entity system
    m.add_class::<ffi::entities::PyNode>()?;
    m.add_class::<ffi::entities::PyEdge>()?;
    m.add_class::<ffi::entities::PyMetaNode>()?;
    m.add_class::<ffi::entities::PyMetaEdge>()?;

    // Register neighborhood sampling system
    m.add_class::<PyNeighborhoodSubgraph>()?;
    m.add_class::<PyNeighborhoodArray>()?;
    m.add_class::<PyNeighborhoodResult>()?;
    m.add_class::<PyNeighborhoodStats>()?;

    // Register traversal and aggregation results
    m.add_class::<PyAggregationResult>()?;
    m.add_class::<PyGroupedAggregationResult>()?;

    // Register hierarchical subgraph types
    m.add_class::<ffi::subgraphs::hierarchical::PyAggregationFunction>()?;
    m.add_class::<ffi::entities::PyMetaNode>()?;

    // Register MetaGraph Composer types
    m.add_class::<ffi::subgraphs::composer::PyEdgeStrategy>()?;
    m.add_class::<ffi::subgraphs::composer::PyComposerPreview>()?;
    m.add_class::<ffi::subgraphs::composer::PyMetaNodePlan>()?;
    m.add_class::<ffi::subgraphs::composer::PyMetaNodePlanExecutor>()?;

    // Add aliases for Python imports - these are already added with correct names

    // Register display functions
    ffi::display::register_display_functions(py, m)?;

    // Add explicit aliases for NumArray registration
    m.add("NumArray", py.get_type::<PyNumArray>())?;
    m.add("StatsArray", py.get_type::<PyNumArray>())?;

    // Add unified builder functions
    m.add_function(wrap_pyfunction!(num_array, m)?)?;
    m.add_function(wrap_pyfunction!(array, m)?)?;
    m.add_function(wrap_pyfunction!(matrix, m)?)?;
    m.add_function(wrap_pyfunction!(table, m)?)?; // Re-enabled for Phase 5 completion
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    // Boolean array builder functions
    // Array factory functions now use unified NumArray with appropriate dtypes
    m.add_function(wrap_pyfunction!(bool_array_factory, m)?)?;
    m.add_function(wrap_pyfunction!(ones_bool_factory, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_bool_factory, m)?)?;
    m.add_function(wrap_pyfunction!(int_array_factory, m)?)?;

    // Use the module registration function (currently empty)
    // module::register_classes(py, m)?;

    // Register neural network submodule
    let neural_module = PyModule::new(py, "neural")?;
    ffi::neural::neural(py, neural_module)?;
    m.add_submodule(neural_module)?;

    Ok(())
}
