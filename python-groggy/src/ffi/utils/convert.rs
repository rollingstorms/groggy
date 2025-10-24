//! FFI bindings for graph conversion functionality
//!
//! This module provides Python bindings for converting graphs and subgraphs
//! to NetworkX format and other external representations.

use groggy::utils::convert::{NetworkXGraph, NetworkXValue};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert a NetworkXGraph to a Python NetworkX graph object
///
/// This function creates an actual NetworkX graph in Python from our
/// internal NetworkXGraph representation.
pub fn networkx_graph_to_python(py: Python, nx_graph: &NetworkXGraph) -> PyResult<PyObject> {
    // Import NetworkX - try both common import patterns
    let networkx = py
        .import("networkx")
        .or_else(|_| py.import("nx"))
        .map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "NetworkX is not installed. Please install it with 'pip install networkx'",
            )
        })?;

    // Create the appropriate NetworkX graph type
    let graph = if nx_graph.directed {
        networkx.getattr("DiGraph")?.call0()?
    } else {
        networkx.getattr("Graph")?.call0()?
    };

    // Add nodes with attributes
    for node in &nx_graph.nodes {
        if node.attributes.is_empty() {
            // No attributes - just pass node ID
            graph.call_method1("add_node", (node.id,))?;
        } else {
            // Pass node ID and attributes as keyword arguments
            let kwargs = PyDict::new(py);
            for (key, value) in &node.attributes {
                let py_value = networkx_value_to_python(py, value)?;
                kwargs.set_item(key, py_value)?;
            }
            graph.call_method("add_node", (node.id,), Some(kwargs))?;
        }
    }

    // Add edges with attributes
    for edge in &nx_graph.edges {
        let edge_attrs = PyDict::new(py);

        for (key, value) in &edge.attributes {
            let py_value = networkx_value_to_python(py, value)?;
            edge_attrs.set_item(key, py_value)?;
        }

        // Only pass edge attributes if they exist
        if !edge_attrs.is_empty() {
            graph.call_method("add_edge", (edge.source, edge.target), Some(edge_attrs))?;
        } else {
            graph.call_method1("add_edge", (edge.source, edge.target))?;
        }
    }

    // Set graph-level attributes
    let graph_dict = graph.getattr("graph")?;
    for (key, value) in &nx_graph.graph_attrs {
        let py_value = networkx_value_to_python(py, value)?;
        graph_dict.set_item(key, py_value)?;
    }

    Ok(graph.to_object(py))
}

/// Convert a NetworkXValue to a Python object
fn networkx_value_to_python(py: Python, value: &NetworkXValue) -> PyResult<PyObject> {
    match value {
        NetworkXValue::String(s) => Ok(s.to_object(py)),
        NetworkXValue::Integer(i) => Ok(i.to_object(py)),
        NetworkXValue::Float(f) => Ok(f.to_object(py)),
        NetworkXValue::Boolean(b) => Ok(b.to_object(py)),
        NetworkXValue::Null => Ok(py.None()),
    }
}

/// Convert a Python NetworkX graph to our internal NetworkXGraph representation
///
/// # Future Feature
///
/// This function is designed for bidirectional NetworkX conversion support,
/// allowing users to import NetworkX graphs into Groggy. Currently unused as
/// the primary use case is exporting Groggy graphs to NetworkX format.
///
/// Planned for future release when import functionality is added to the Python API.
#[allow(dead_code)]
pub fn python_to_networkx_graph(py: Python, py_graph: &PyAny) -> PyResult<NetworkXGraph> {
    // Check if it's a NetworkX graph
    let networkx = py
        .import("networkx")
        .or_else(|_| py.import("nx"))
        .map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "NetworkX is not installed. Please install it with 'pip install networkx'",
            )
        })?;

    // Check if the object is a NetworkX graph
    let is_graph = networkx.getattr("is_directed")?.call1((py_graph,))?;
    let directed = is_graph.extract::<bool>()?;

    // Get nodes with attributes
    let mut nodes = Vec::new();
    let nodes_data = py_graph.call_method1("nodes", (true,))?; // nodes(data=True)

    for item in nodes_data.iter()? {
        let (node_id, attrs_dict) = item?.extract::<(usize, &PyDict)>()?;
        let mut attributes = std::collections::HashMap::new();

        for (key, value) in attrs_dict {
            let key_str = key.extract::<String>()?;
            let nx_value = python_to_networkx_value(value)?;
            attributes.insert(key_str, nx_value);
        }

        nodes.push(groggy::utils::convert::NetworkXNode {
            id: node_id,
            attributes,
        });
    }

    // Get edges with attributes
    let mut edges = Vec::new();
    let edges_data = py_graph.call_method1("edges", (true,))?; // edges(data=True)

    for item in edges_data.iter()? {
        let (source, target, attrs_dict) = item?.extract::<(usize, usize, &PyDict)>()?;
        let mut attributes = std::collections::HashMap::new();

        for (key, value) in attrs_dict {
            let key_str = key.extract::<String>()?;
            let nx_value = python_to_networkx_value(value)?;
            attributes.insert(key_str, nx_value);
        }

        edges.push(groggy::utils::convert::NetworkXEdge {
            source,
            target,
            attributes,
        });
    }

    // Get graph-level attributes
    let mut graph_attrs = std::collections::HashMap::new();
    let graph_dict = py_graph.getattr("graph")?;

    if let Ok(dict) = graph_dict.downcast::<PyDict>() {
        for (key, value) in dict {
            let key_str = key.extract::<String>()?;
            let nx_value = python_to_networkx_value(value)?;
            graph_attrs.insert(key_str, nx_value);
        }
    }

    Ok(NetworkXGraph {
        directed,
        nodes,
        edges,
        graph_attrs,
    })
}

/// Convert a Python value to NetworkXValue
///
/// # Future Feature
///
/// Helper function for bidirectional NetworkX conversion. Currently unused but
/// will be needed when importing NetworkX graphs into Groggy.
#[allow(dead_code)]
fn python_to_networkx_value(py_value: &PyAny) -> PyResult<NetworkXValue> {
    if py_value.is_none() {
        Ok(NetworkXValue::Null)
    } else if let Ok(s) = py_value.extract::<String>() {
        Ok(NetworkXValue::String(s))
    } else if let Ok(i) = py_value.extract::<i64>() {
        Ok(NetworkXValue::Integer(i))
    } else if let Ok(f) = py_value.extract::<f64>() {
        Ok(NetworkXValue::Float(f))
    } else if let Ok(b) = py_value.extract::<bool>() {
        Ok(NetworkXValue::Boolean(b))
    } else {
        // For other types, convert to string representation
        let string_repr = py_value.str()?.extract::<String>()?;
        Ok(NetworkXValue::String(string_repr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_networkx_value_conversion() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Test various NetworkXValue types
            let string_val = NetworkXValue::String("test".to_string());
            let py_val = networkx_value_to_python(py, &string_val).unwrap();
            assert_eq!(py_val.extract::<String>(py).unwrap(), "test");

            let int_val = NetworkXValue::Integer(42);
            let py_val = networkx_value_to_python(py, &int_val).unwrap();
            assert_eq!(py_val.extract::<i64>(py).unwrap(), 42);

            let float_val = NetworkXValue::Float(std::f64::consts::PI);
            let py_val = networkx_value_to_python(py, &float_val).unwrap();
            assert!(
                (py_val.extract::<f64>(py).unwrap() - std::f64::consts::PI).abs() < f64::EPSILON
            );

            let bool_val = NetworkXValue::Boolean(true);
            let py_val = networkx_value_to_python(py, &bool_val).unwrap();
            assert!(py_val.extract::<bool>(py).unwrap());

            let null_val = NetworkXValue::Null;
            let py_val = networkx_value_to_python(py, &null_val).unwrap();
            assert!(py_val.is_none(py));
        });
    }
}
