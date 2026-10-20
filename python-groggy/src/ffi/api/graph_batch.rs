//! Translation of the Python batch description into the Rust write plan.

use crate::ffi::utils::python_value_to_attr_value;
use groggy::{BatchEdgeRef, BatchNodeRef, GraphBatch};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

fn node_ref(value: &PyAny) -> PyResult<BatchNodeRef> {
    if let Ok(id) = value.extract::<usize>() {
        return Ok(BatchNodeRef::Existing(id));
    }
    if let Ok(reference) = value.downcast::<PyDict>() {
        if let Some(position) = reference.get_item("added")? {
            return Ok(BatchNodeRef::Added(position.extract()?));
        }
    }
    Err(PyTypeError::new_err(
        "node reference must be an existing ID or {'added': position}",
    ))
}

fn edge_ref(value: &PyAny) -> PyResult<BatchEdgeRef> {
    if let Ok(id) = value.extract::<usize>() {
        return Ok(BatchEdgeRef::Existing(id));
    }
    if let Ok(reference) = value.downcast::<PyDict>() {
        if let Some(position) = reference.get_item("added")? {
            return Ok(BatchEdgeRef::Added(position.extract()?));
        }
    }
    Err(PyTypeError::new_err(
        "edge reference must be an existing ID or {'added': position}",
    ))
}

fn pairs<'py>(value: &'py PyAny, context: &str) -> PyResult<Vec<(&'py PyAny, &'py PyAny)>> {
    let items = value.downcast::<PyList>().map_err(|_| {
        PyTypeError::new_err(format!("{context} must be a list of two-item pairs"))
    })?;
    items
        .iter()
        .enumerate()
        .map(|(position, item)| {
            let pair = item.downcast::<PyTuple>().map_err(|_| {
                PyTypeError::new_err(format!("{context}[{position}] must be a pair"))
            })?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "{context}[{position}] must have exactly two items"
                )));
            }
            Ok((pair.get_item(0)?, pair.get_item(1)?))
        })
        .collect()
}

pub(crate) fn parse_batch(batch: &PyAny) -> PyResult<GraphBatch> {
    let owned;
    let description = if let Ok(dict) = batch.downcast::<PyDict>() {
        dict
    } else {
        owned = batch.call_method0("to_dict")?;
        owned.downcast::<PyDict>().map_err(|_| {
            PyTypeError::new_err("GraphBatch.to_dict() must return a dictionary")
        })?
    };
    let mut result = GraphBatch::default();
    if let Some(count) = description.get_item("add_nodes")? {
        result.add_nodes = count.extract()?;
    }
    if let Some(edges) = description.get_item("add_edges")? {
        for (source, target) in pairs(edges, "add_edges")? {
            result.add_edges.push((node_ref(source)?, node_ref(target)?));
        }
    }
    if let Some(attrs) = description.get_item("node_attrs")? {
        for (name, values) in attrs.downcast::<PyDict>()?.iter() {
            let name: String = name.extract()?;
            let mut assignments = Vec::new();
            for (reference, value) in pairs(values, &format!("node_attrs.{name}"))? {
                assignments.push((node_ref(reference)?, python_value_to_attr_value(value)?));
            }
            result.node_attrs.insert(name, assignments);
        }
    }
    if let Some(attrs) = description.get_item("edge_attrs")? {
        for (name, values) in attrs.downcast::<PyDict>()?.iter() {
            let name: String = name.extract()?;
            let mut assignments = Vec::new();
            for (reference, value) in pairs(values, &format!("edge_attrs.{name}"))? {
                assignments.push((edge_ref(reference)?, python_value_to_attr_value(value)?));
            }
            result.edge_attrs.insert(name, assignments);
        }
    }
    if let Some(nodes) = description.get_item("remove_nodes")? {
        for reference in nodes.downcast::<PyList>()?.iter() {
            result.remove_nodes.push(node_ref(reference)?);
        }
    }
    if let Some(edges) = description.get_item("remove_edges")? {
        for reference in edges.downcast::<PyList>()?.iter() {
            result.remove_edges.push(edge_ref(reference)?);
        }
    }
    Ok(result)
}
